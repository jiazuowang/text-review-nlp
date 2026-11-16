"""Verify this self-contained package and optionally reproduce its experiments."""
import argparse
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reproduce',action='store_true',help='Recompute this module, including training where relevant')
    ap.add_argument('--include-cv',action='store_true',help='Also recompute five-fold X cross-validation (C or FULL)')
    ap.add_argument('--compare-existing',action='store_true',help='With --reproduce and --output, compare a prior run without rerunning its completed commands')
    ap.add_argument('--output',type=Path,help='A NEW directory for logs and generated results')
    args=ap.parse_args()
    meta=json.loads((ROOT/'PACKAGE_INFO.json').read_text(encoding='utf-8'))
    module=meta['module']
    out=(args.output or ROOT/'runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    if args.compare_existing:
        if not args.reproduce or not args.output:raise ValueError('--compare-existing requires --reproduce and an existing --output')
        prior=json.loads((out/'summary.json').read_text(encoding='utf-8'))
        if prior['module']!=module:raise ValueError('Prior run belongs to another module')
    else:out.mkdir(parents=True,exist_ok=False)
    result_name='comparison_summary.json' if args.compare_existing else 'summary.json'
    report={'module':module,'status':'running','mode':'compare_existing' if args.compare_existing else 'reproduce' if args.reproduce else 'verify',
            'python':sys.version,'platform':platform.platform(),'root':str(ROOT),'steps':[],
            'input_policy':'Frozen inputs per stage; no automatic substitution of newly trained B/C models into D/E/F.'}
    def call(name,*arguments):
        if args.compare_existing:
            if name not in prior['steps'] or not (out/(name+'.log')).is_file():raise RuntimeError('No evidence of a completed prior command: '+name)
            print('Reusing completed command: '+name,flush=True)
            report['steps'].append(name)
            return
        print('Running '+name,flush=True)
        with (out/(name+'.log')).open('w',encoding='utf-8') as log:
            p=subprocess.run([sys.executable,*map(str,arguments)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if p.returncode:raise RuntimeError(f'{name} failed; see {out/(name+".log")}')
        report['steps'].append(name)
    def compare_csv(expected,actual,atol=1e-9,rtol=1e-8,ignore=()):
        import pandas as pd
        import numpy as np
        a=pd.read_csv(expected);b=pd.read_csv(actual)
        timing=[c for c in a if 'seconds' in c or c.endswith('_ms') or c in ignore]
        a=a.drop(columns=timing);b=b.drop(columns=timing)
        pd.testing.assert_frame_equal(a,b,check_exact=False,rtol=rtol,atol=atol)
        differences={c:float((a[c]-b[c]).abs().max()) for c in a.select_dtypes('number') if float((a[c]-b[c]).abs().max())>1e-10}
        if differences:report.setdefault('numerical_differences',[]).append({'file':str(actual.relative_to(out)),'max_absolute_by_column':differences,'atol':atol,'rtol':rtol})
        for score in ['ai_score','X_score']:
            if score in a:
                np.testing.assert_array_equal(a[score].to_numpy()>=.5,b[score].to_numpy()>=.5)
    def compare_selected(original,new,filename='run.json'):
        old=json.loads((original/filename).read_text())['selected']
        fresh=json.loads((new/filename).read_text())['selected']
        for key,value in old.items():
            if key in {'seconds','fit_predict_seconds','iterations'}:continue
            if isinstance(value,float):
                if abs(value-fresh[key])>1e-9:raise ValueError('Selected model metric/config changed: '+key)
            elif fresh[key]!=value:raise ValueError('Selected model config changed: '+key)
    def compare_folder(original,new,atol=1e-9):
        for p in sorted(original.glob('*.csv')):
            compare_csv(p,new/p.name,atol=atol,rtol=0 if atol>1e-9 else 1e-8)
    try:
        manifest=json.loads((ROOT/'PACKAGE_CHECKSUMS.json').read_text(encoding='utf-8'))
        for name,digest in manifest.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed or missing package file: '+name)
        report['checked_files']=len(manifest)
        versions={}
        for line in (ROOT/'requirements.txt').read_text().splitlines():
            if not line.strip() or line.startswith('#'):continue
            name,expected=line.split('==');versions[name]=importlib.metadata.version(name)
            if versions[name]!=expected:raise ValueError(f'Install requirements.txt: {name} expected {expected}, found {versions[name]}')
        report['dependencies']=versions
        import src
        if not Path(src.__file__).resolve().is_relative_to(ROOT):raise RuntimeError('Imported code from outside this package')
        call('dependency_check','-m','pip','check')
        if module=='FULL':
            call('full_project_verification','-m','scripts.team_reproduce','--output',out/'whole_project')
        else:
            call('unit_tests','-m','unittest','discover','-s','tests','-v')
        import pandas as pd
        import numpy as np
        import joblib
        if module=='A':call('frozen_data_validation','-m','src.data.validate')
        if module in ['B','C']:
            from src.common import load_development_data
            _,val,_=load_development_data(ROOT)
            pairs=[('module_b','pipeline.joblib')] if module=='B' else [('module_c','pipeline.joblib'),('extension_x','pipeline.joblib')]
            for folder,model_name in pairs:
                model=joblib.load(ROOT/'artifacts'/folder/model_name)
                scores=model.predict_proba(val.text.tolist())[:,list(model.classes_).index(1)]
                file='full_validation_predictions.csv' if folder=='module_c' else 'validation_predictions.csv'
                reference=pd.read_csv(ROOT/'artifacts'/folder/file).set_index('sample_id')
                np.testing.assert_allclose(scores,reference.loc[val.sample_id,'ai_score'].to_numpy(),atol=1e-9,rtol=0)
                report.setdefault('frozen_validation',{})[folder]={'texts':len(val),'accuracy':float(np.mean((scores>=.5)==val.label.to_numpy()))}
        if module=='D':
            d=pd.read_csv(ROOT/'artifacts/module_d/paired_scores.csv')
            assert len(d)==1820
            report['archived_edit_pairs']=len(d)
        if module=='E':
            from src.evaluation.run import load_inputs
            frame,_,_=load_inputs(ROOT);frame=frame.loc[frame.main_task].sort_values('sample_id')
            reference=pd.read_csv(ROOT/'artifacts/module_e/predictions.csv').set_index('sample_id')
            assert len(frame)==506
            for key in ['B','C']:
                model=joblib.load(ROOT/f'resources/detectors/{key}/pipeline.joblib')
                scores=model.predict_proba(frame.text.tolist())[:,list(model.classes_).index(1)]
                np.testing.assert_allclose(scores,reference.loc[frame.sample_id,key+'_score'].to_numpy(),atol=1e-9,rtol=0)
                report.setdefault('frozen_test',{})[key]={'texts':len(frame),'accuracy':float(np.mean((scores>=.5)==frame.label.to_numpy()))}
        if module=='F':
            call('fixtures_exports','-m','scripts.verify_module_f','--output',out/'fixture_checks.json')
            call('batch','-m','src.app.cli','examples/batch.csv',out/'batch_results.csv','--batch','--model','Compare All')
        if args.reproduce:
            modules=['A','B','C','D','E','F'] if module=='FULL' else [module]
            for key in modules:
                if key=='A':
                    target=out/'A';call('A_prepare','-m','src.data.prepare','--output',target)
                    call('A_validate_new','-m','src.data.validate','--input',target/'texts.parquet')
                    for name in ['texts.parquet','train_main.parquet','val_main.parquet','test_main.parquet']:
                        pd.testing.assert_frame_equal(pd.read_parquet(ROOT/'data/processed'/name),pd.read_parquet(target/name))
                    report['A_partition_counts']={s:len(pd.read_parquet(target/f'{s}_main.parquet')) for s in ['train','val','test']}
                if key=='B':
                    target=out/'B';call('B_train','-m','src.models.tfidf_lr','--output',target)
                    compare_csv(ROOT/'artifacts/module_b/validation_predictions.csv',target/'validation_predictions.csv')
                    compare_selected(ROOT/'artifacts/module_b',target)
                    compare_csv(ROOT/'artifacts/module_b/validation_search.csv',target/'validation_search.csv',atol=1e-4,rtol=0,ignore=['iterations'])
                    report['B_search_note']='Candidate solver iterations and timings are diagnostics. Candidate ranking metrics allow absolute 1e-4; selected-model configuration and metrics are checked separately at 1e-9.'
                if key=='C':
                    target=out/'C';call('C_train','-m','src.models.train_style','--with-pos','--output',target)
                    compare_csv(ROOT/'artifacts/module_c/ablation_summary.csv',target/'ablation_summary.csv')
                    compare_selected(ROOT/'artifacts/module_c',target)
                    for p in (ROOT/'artifacts/module_c').glob('*validation_predictions.csv'):compare_csv(p,target/p.name)
                    x=out/'X';call('X_train','-m','src.models.extension','--output',x)
                    compare_csv(ROOT/'artifacts/extension_x/validation_predictions.csv',x/'validation_predictions.csv',atol=1e-5,rtol=0)
                    compare_selected(ROOT/'artifacts/extension_x',x,'selection_frozen.json')
                    call('X_evaluate','-m','src.evaluation.extension_eval','--model-dir',x,'--output',out/'X_evaluation')
                    compare_folder(ROOT/'artifacts/extension_x_evaluation',out/'X_evaluation',atol=1e-5)
                    report['X_retraining_note']='Newly trained X scores allow absolute 1e-5 with identical default-threshold classes. Selected configuration and validation metrics remain checked at 1e-9. Frozen-model inference above retains its strict 1e-9 tolerance.'
                    if args.include_cv:
                        call('X_cv','-m','src.evaluation.extension_cv','--model-dir',x,'--output',out/'X_cv')
                        compare_csv(ROOT/'artifacts/extension_x_cv/fold_metrics.csv',out/'X_cv/fold_metrics.csv',atol=1e-5,rtol=0)
                        compare_csv(ROOT/'artifacts/extension_x_cv/out_of_fold_predictions.csv',out/'X_cv/out_of_fold_predictions.csv',atol=1e-5,rtol=0)
                if key=='D':
                    target=out/'D';call('D_analysis','-m','src.edits.analyze','--output',target)
                    compare_folder(ROOT/'artifacts/module_d',target)
                    call('D_review_summary','-m','src.edits.review','--analysis',target,'--output',out/'D_quality')
                    compare_folder(ROOT/'artifacts/module_d_quality',out/'D_quality')
                if key=='E':
                    target=out/'E';call('E_evaluation','-m','src.evaluation.run','--output',target)
                    compare_folder(ROOT/'artifacts/module_e',target)
                if key=='F' and module=='FULL':
                    call('F_exports','-m','scripts.verify_module_f','--output',out/'fixture_checks.json')
            report['reproduced_modules']=modules
            report['comparison']='Recomputed tables/predictions matched archived results within documented numerical tolerance; timing columns excluded.'
        report['status']='passed'
        if args.compare_existing:report['verification_basis']='Current file/dependency checks and fresh comparisons of actual outputs from previously completed commands. No commands or training rerun; original summary and logs retained.'
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        (out/result_name).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
        print('Saved '+str(out/result_name),flush=True)
    print('PASS '+module+' '+report['mode'],flush=True)

if __name__=='__main__':main()
