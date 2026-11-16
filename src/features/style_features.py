"""Versioned English style, comma and optional pretrained POS features."""
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

ROOT = Path(__file__).resolve().parents[2]
WORD = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*")
TOKEN = re.compile(r"[A-Za-z]+(?:['’\-][A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]")
BOUNDARY = re.compile(r'''[.!?]+["')\]]*(?:\s+|$)|\n+''')
VERSION = '1.0.0'
BASE = {
 'word_count': 'English word count', 'sentence_count': 'Number of segments containing English words',
 'sentence_words_mean': 'Mean word count per sentence', 'sentence_words_std': 'Population standard deviation of sentence word counts',
 'word_length_mean': 'Mean characters per word including internal apostrophes and hyphens',
 'type_token_ratio': 'Lowercase word types divided by word count', 'hapax_ratio': 'Singleton lowercase word types divided by word count',
 'repeated_bigram_ratio': 'One minus distinct within-sentence lowercase bigrams divided by all within-sentence bigrams',
 'uppercase_ratio': 'ASCII uppercase letters divided by ASCII letters',
 'semicolon_per_100_words': 'semicolon count per 100 words', 'colon_per_100_words': 'colon count per 100 words',
 'question_per_100_words': 'question count per 100 words', 'exclamation_per_100_words': 'exclamation count per 100 words',
 'paragraph_count': 'Non-empty paragraphs separated by blank lines', 'paragraph_words_mean': 'Word count divided by paragraph count',
 'list_line_ratio': 'Bullet or numbered lines divided by non-empty lines'
}
COMMA = {
 'comma_per_100_words': 'comma count per 100 words',
 'comma_sentence_ratio': 'Sentences containing a comma divided by sentence count',
 'comma_sentence_rate_mean': 'Mean of comma count divided by word count within each sentence',
 'comma_relative_position_mean': 'Mean words before a comma divided by words in its sentence',
 'comma_relative_position_std': 'Population standard deviation of relative comma positions'
}
POS = {f'pos_diversity_{n}': f'Distinct within-sentence POS {n}-grams divided by total within-sentence POS {n}-grams' for n in range(1, 6)}
POS.update({f'pos_{name}_ratio': f'{name} POS count divided by word and number tokens'
            for name in ['noun', 'verb', 'adjective', 'adverb', 'pronoun']})
COMMA_POS = {'comma_pos_pair_diversity': 'Distinct adjacent POS pairs around commas divided by valid comma count'}
LENGTH = {'word_count', 'sentence_count', 'sentence_words_mean', 'sentence_words_std',
          'paragraph_count', 'paragraph_words_mean'}
GROUPS = {'base': list(BASE), 'base_comma': [*BASE, *COMMA],
          'base_pos': [*BASE, *POS], 'full': [*BASE, *COMMA, *POS, *COMMA_POS]}
GROUPS['base_comma_no_length'] = [f for f in GROUPS['base_comma'] if f not in LENGTH]
GROUPS['full_no_length'] = [f for f in GROUPS['full'] if f not in LENGTH]


def schema():
    return {'version': VERSION, 'definitions': {**BASE, **COMMA, **POS, **COMMA_POS},
            'groups': GROUPS, 'direct_length_features': sorted(LENGTH),
            'word_regex': WORD.pattern, 'sentence_boundary_regex': BOUNDARY.pattern,
            'empty_denominator': 0.0, 'POS_resource': 'NLTK averaged_perceptron_tagger_eng',
            'sentence_rules': 'Split on terminal punctuation followed by whitespace/end, and line breaks. Abbreviations such as Dr. split at the period.'}


def ratio(a, b):
    return float(a / b) if b else 0.0


def mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def std(values):
    return float(np.std(values)) if len(values) else 0.0


@lru_cache(maxsize=1)
def load_tagger():
    folder = ROOT / 'resources/pos'
    manifest = folder / 'manifest.json'
    if not manifest.exists():
        raise FileNotFoundError('POS resource missing: run python scripts/setup_pos.py --input OFFICIAL_ZIP')
    meta = json.loads(manifest.read_text(encoding='utf-8'))
    for name, expected in meta['files'].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
            raise ValueError('POS resource checksum mismatch: ' + name)
    from nltk.tag.perceptron import PerceptronTagger
    tagger = PerceptronTagger(load=False)
    model_dir = folder / 'averaged_perceptron_tagger_eng'
    tagger.model.weights = json.loads((model_dir / 'averaged_perceptron_tagger_eng.weights.json').read_text())
    tagger.tagdict = json.loads((model_dir / 'averaged_perceptron_tagger_eng.tagdict.json').read_text())
    tagger.classes = set(json.loads((model_dir / 'averaged_perceptron_tagger_eng.classes.json').read_text()))
    tagger.model.classes = tagger.classes
    return tagger


def extract(text, with_pos=False, tagger=None):
    if not isinstance(text, str):
        raise ValueError('Text input must be a string')
    sentences = [s for s in BOUNDARY.split(text) if WORD.search(s)]
    words_by_sentence = [WORD.findall(s) for s in sentences]
    words = WORD.findall(text)
    counts = Counter(w.lower() for w in words)
    n = len(words)
    bigrams = [tuple(w.lower() for w in ws[i:i+2]) for ws in words_by_sentence for i in range(len(ws)-1)]
    paragraphs = [p for p in re.split(r'\n\s*\n', text.strip()) if p.strip()]
    lines = [line for line in text.splitlines() if line.strip()]
    lengths = [len(ws) for ws in words_by_sentence]
    result = {'word_count': n, 'sentence_count': len(sentences),
        'sentence_words_mean': mean(lengths), 'sentence_words_std': std(lengths),
        'word_length_mean': mean([len(w) for w in words]), 'type_token_ratio': ratio(len(counts), n),
        'hapax_ratio': ratio(sum(v == 1 for v in counts.values()), n),
        'repeated_bigram_ratio': 1 - ratio(len(set(bigrams)), len(bigrams)) if bigrams else 0.0,
        'uppercase_ratio': ratio(len(re.findall('[A-Z]', text)), len(re.findall('[A-Za-z]', text))),
        'paragraph_count': len(paragraphs), 'paragraph_words_mean': ratio(n, len(paragraphs)),
        'list_line_ratio': ratio(sum(bool(re.match(r'^\s*(?:[-*•]|\d+[.)])\s+', l)) for l in lines), len(lines))}
    for name, punct in [('semicolon', ';'), ('colon', ':'), ('question', '?'), ('exclamation', '!')]:
        result[name + '_per_100_words'] = ratio(text.count(punct) * 100, n)
    positions = [ratio(len(WORD.findall(s[:i])), len(WORD.findall(s)))
                 for s in sentences for i, c in enumerate(s) if c == ',']
    result.update({'comma_per_100_words': ratio(text.count(',') * 100, n),
        'comma_sentence_ratio': ratio(sum(',' in s for s in sentences), len(sentences)),
        'comma_sentence_rate_mean': mean([ratio(s.count(','), len(ws)) for s, ws in zip(sentences, words_by_sentence)]),
        'comma_relative_position_mean': mean(positions), 'comma_relative_position_std': std(positions)})
    if with_pos:
        tagger = tagger or load_tagger()
        tagged = [tagger.tag(TOKEN.findall(s)) for s in sentences]
        sequences = [[tag for token, tag in sent if WORD.fullmatch(token) or token[0].isdigit()] for sent in tagged]
        for width in range(1, 6):
            grams = [tuple(seq[i:i+width]) for seq in sequences for i in range(len(seq)-width+1)]
            result[f'pos_diversity_{width}'] = ratio(len(set(grams)), len(grams))
        tags = [tag for seq in sequences for tag in seq]
        for name, prefixes in [('noun', ('NN',)), ('verb', ('VB', 'MD')), ('adjective', ('JJ',)),
                               ('adverb', ('RB',)), ('pronoun', ('PRP', 'WP'))]:
            result[f'pos_{name}_ratio'] = ratio(sum(t.startswith(prefixes) for t in tags), len(tags))
        pairs = [(sent[i-1][1], sent[i+1][1]) for sent in tagged for i in range(1, len(sent)-1) if sent[i][0] == ',']
        result['comma_pos_pair_diversity'] = ratio(len(set(pairs)), len(pairs))
    return result


class StyleTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, group='base_comma'):
        self.group = group

    def fit(self, X, y=None):
        if self.group not in GROUPS:
            raise ValueError('Unknown feature group')
        if self.needs_pos():
            load_tagger()
        return self

    def needs_pos(self):
        return any(f in POS or f in COMMA_POS for f in GROUPS[self.group])

    def transform(self, X):
        if isinstance(X, str):
            raise ValueError('Pass a collection of texts')
        rows = [extract(t, with_pos=self.needs_pos()) for t in X]
        return np.array([[r[f] for f in GROUPS[self.group]] for r in rows], dtype=float).reshape(-1, len(GROUPS[self.group]))

    def get_feature_names_out(self, input_features=None):
        return np.array(GROUPS[self.group], dtype=object)
