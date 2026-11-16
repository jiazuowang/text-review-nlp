"""Reproduce frozen inference on another computer; never train or overwrite evidence."""
import argparse
from datetime import datetime
import json
import platform
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New output directory; must not exist')
    args = parser.parse_args()
    output = args.output or ROOT / 'team_results' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    summary = {'status': 'running', 'python': sys.version, 'platform': platform.platform(),
               'executable': sys.executable, 'scope': 'Frozen-model inference, not retraining',
               'output': str(output)}

    def run(name, arguments):
        print('Running: ' + name, flush=True)
        with (output / (name + '.log')).open('w', encoding='utf-8') as stream:
            process = subprocess.run([sys.executable, *arguments], cwd=ROOT,
                                     stdout=stream, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f'{name} failed; read {output / (name + ".log")}')

    try:
        run('dependencies', ['-m', 'pip', 'freeze'])
        run('pip_check', ['-m', 'pip', 'check'])
        run('submission', ['verify_submission.py'])
        run('fixtures_and_exports', ['-m', 'scripts.verify_module_f', '--output', str(output / 'fixture_checks.json')])
        run('unit_tests', ['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
        run('batch', ['-m', 'src.app.cli', 'examples/batch.csv', str(output / 'batch_results.csv'), '--batch', '--model', 'Compare All'])

        import numpy as np
        import pandas as pd
        from src.app.service import Detector, NAMES
        from src.evaluation.run import load_inputs
        from src.evaluation.metrics import metrics

        frame, _, _ = load_inputs(ROOT)
        main_test = frame.loc[frame.main_task].sort_values('sample_id').copy()
        if len(main_test) != 506:
            raise ValueError('Expected 506 frozen original test documents')
        baseline = pd.read_csv(ROOT / 'artifacts/module_e/predictions.csv').set_index('sample_id')
        combined = pd.read_csv(ROOT / 'artifacts/extension_x_evaluation/predictions.csv').set_index('sample_id')
        detector = Detector()
        scores = main_test[['sample_id', 'label']].copy()
        results = []
        expected_correct = {'B': 374, 'C': 356, 'X': 392}
        for name, key in NAMES.items():
            print(f'Predicting all 506 original test texts: {name}', flush=True)
            model = detector.load(key)
            prediction = model.predict_proba(main_test.text.tolist())[:, list(model.classes_).index(1)]
            reference = combined if key == 'X' else baseline
            archived = reference.loc[main_test.sample_id, key + '_score'].to_numpy()
            np.testing.assert_allclose(prediction, archived, atol=1e-9, rtol=0)
            correct = int(np.sum((prediction >= 0.5) == main_test.label.to_numpy()))
            if correct != expected_correct[key]:
                raise ValueError(f'Unexpected correct count: {key} {correct}')
            scores[key + '_score'] = prediction
            result = {'model': name, 'key': key, 'n': len(main_test), 'correct': correct,
                      'accuracy': correct / len(main_test), 'threshold': 0.5,
                      'max_absolute_score_difference': float(np.max(np.abs(prediction - archived))),
                      'metrics': metrics(main_test.label, prediction, 0.5)}
            results.append(result)
            print(f'  Matched archived predictions; accuracy {100 * correct / len(main_test):.2f}%', flush=True)
        scores.to_csv(output / 'main_test_predictions.csv', index=False)
        summary.update(status='passed', main_test_results=results,
                       fixture_score_checks=6,
                       notes='Combined Features is a follow-up estimate on the previously used test set. Full training, editing analysis and bootstrap reproduction are separate tasks.')
    except Exception as exc:
        summary.update(status='failed', error=str(exc))
        raise
    finally:
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        print('Evidence saved to: ' + str(output), flush=True)
    print('PASS: fixtures, exports, unit tests, batch and 506-text frozen inference.', flush=True)


if __name__ == '__main__':
    main()
