"""Verify package hashes, archived validation scores, and document exports."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from src.app.service import Detector
from src.app.document_input import read_document
from src.app.export import export_report

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    manifest=ROOT/'CHECKSUMS_F.json'
    if manifest.exists():
        for name,expected in json.loads(manifest.read_text()).items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected, name
    detector=Detector();checks=0
    cases=json.loads((ROOT/'examples/verification_cases.json').read_text(encoding='utf-8'))
    report=None
    for case in cases:
        text=(ROOT/'examples'/case['file']).read_text(encoding='utf-8')
        report=detector.run(text,'Compare All')
        for r in report['results']:
            assert abs(r['score']-case['scores'][r['model']])<1e-10,(case['file'],r['model'])
            checks+=1
    with tempfile.TemporaryDirectory() as temp:
        for extension in ['docx','pdf','json']:
            path=Path(temp)/('result.'+extension);export_report(report,path)
            if extension!='json':assert 'Text Analysis Report' in read_document(path)
            else:assert json.loads(path.read_text(encoding='utf-8'))==report
    result={'status':'passed','frozen_score_checks':checks,'export_formats':['docx','pdf','json']}
    if args.output:args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':main()
