"""Verified, read-only inference for the desktop and command-line interfaces."""
import csv
import hashlib
import json
from pathlib import Path
import threading
import joblib
from src.app.analysis import analyze
from src.features.style_features import WORD

ROOT = Path(__file__).resolve().parents[2]
NAMES = {'Combined Features': 'X', 'Lexical Features': 'B', 'Writing Style': 'C'}

class Cancelled(Exception):
    pass

def check_cancel(event):
    if event is not None and event.is_set():
        raise Cancelled('Analysis cancelled.')

def validate_text(text):
    count = len(WORD.findall(text))
    if count == 0:
        raise ValueError('Enter non-empty English text containing words.')
    if any('\u4e00' <= c <= '\u9fff' for c in text):
        raise ValueError('This model supports English documents. Please provide an English text.')
    return count

def read_batch(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'id', 'text'} <= set(reader.fieldnames or []):
            raise ValueError('CSV requires id and text columns.')
        rows = list(reader)
    if not rows or len(rows) > 1000:
        raise ValueError('Import a CSV containing 1 to 1,000 documents.')
    ids = [r['id'] for r in rows]
    if any(not x or not x.strip() for x in ids) or len(ids) != len(set(ids)):
        raise ValueError('Each document needs a unique, non-empty id.')
    if any(r.get('text') is None for r in rows):
        raise ValueError('Every CSV row needs a text field.')
    return [{'id': r['id'], 'text': r['text']} for r in rows]

class Detector:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        self.config = json.loads((self.root/'configs/integration_models.json').read_text(encoding='utf-8'))
        self.models = {}
        self.lock = threading.RLock()

    def load(self, key):
        with self.lock:
            if key not in self.models:
                meta = self.config['models'][key]
                path = self.root / meta['path']
                if hashlib.sha256(path.read_bytes()).hexdigest() != meta['sha256']:
                    raise ValueError('Model integrity check failed. Restore the original application package.')
                self.models[key] = joblib.load(path)
            return self.models[key]

    def run(self, text, selection='Combined Features', mode='Default', local=True, cancel=None):
        validate_text(text)
        if selection not in {*NAMES, 'Compare All'}:
            raise ValueError('Unknown model selection.')
        if mode not in {'Default', 'Low False Positive'}:
            raise ValueError('Unknown threshold mode.')
        names = list(NAMES) if selection == 'Compare All' else [selection]
        results = []
        with self.lock:
            for name in names:
                check_cancel(cancel)
                key = NAMES[name]
                model = self.load(key)
                threshold = self.config['models'][key]['thresholds']['default' if mode == 'Default' else 'low_FPR']
                if local:
                    result = analyze(text, model, threshold, cancel=cancel)
                else:
                    score = float(model.predict_proba([text])[0, list(model.classes_).index(1)])
                    result = dict(score=score, threshold=threshold, prediction='AI origin' if score >= threshold else 'Human origin',
                                  word_count=len(WORD.findall(text)), status='ok', sentences=[], suggestions=[])
                check_cancel(cancel)
                result.update(model=name, threshold_mode=mode, model_sha256=self.config['models'][key]['sha256'])
                results.append(result)
        return {'text': text, 'selection': selection, 'threshold_mode': mode, 'results': results}

    def batch(self, rows, selection, mode, cancel=None, progress=None):
        output = []
        for index, row in enumerate(rows):
            check_cancel(cancel)
            try:
                report = self.run(row['text'], selection, mode, local=False, cancel=cancel)
                output.append({'id': row['id'], 'status': 'Complete', **report})
            except ValueError as exc:
                output.append({'id': row['id'], 'text': row['text'], 'status': 'Invalid input', 'error': str(exc), 'results': []})
            if progress:
                progress(index+1, len(rows))
        return output
