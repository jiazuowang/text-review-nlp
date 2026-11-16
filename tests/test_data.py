"""Synthetic engineering fixtures only; never used as research data."""
import json
import random
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from src.data.prepare import expand,parse_edits,near_pairs,prepare,ROOT
from src.data.validate import validate_frame
from src.data.download import fingerprints,verify

def config():
    return json.loads((ROOT/'configs/data.json').read_text())

def fixture(count=40):
    rng=random.Random(19)
    def text():
        return ' '.join(''.join(rng.choices('abcdefghijklmnopqrstuvwxyz',k=8)) for _ in range(65))+'.'
    records=[]
    for i in range(count):
        records.append({'id':i,'prompt_id':f'prompt-{i}','prompt':f'Question number {i}',
             'category':'Open QA' if i%2 else 'Generation','model':'fixture-model',
             'human_output':text(),'model_output':text(),'human_edits':text(),
             'llama-3.1-70b_edits':[{f'P{j}':text() for j in range(1,4)}],
             'gpt-4o_edits':json.dumps([{f'P{j}':text()} for j in range(1,4)])})
    return pd.DataFrame(records)

class DataTests(unittest.TestCase):
    def test_nested_formats(self):
        for data in [{'P1':'hello'},[{'P1':'hello'}], '[{"P1":"hello"}]', "[{'P1':'hello'}]"]:
            self.assertEqual(parse_edits(data),[('P1','hello')])

    def test_reject_code_and_unknown_schema(self):
        for data in ["__import__('os').getcwd()",{'other':'text'},[{'P1':'a'},{'P1':'b'}]]:
            with self.assertRaises(ValueError):parse_edits(data)

    def test_expand_nine_versions(self):
        frame,_=expand(fixture(3),config())
        self.assertEqual(len(frame),27)
        self.assertEqual(frame.variant.value_counts()['llm_edited'],18)
        self.assertEqual(set(frame.loc[frame.variant.eq('human_edited'),'label']),{1})

    def test_required_columns(self):
        with self.assertRaises(ValueError):expand(fixture().drop(columns='prompt_id'),config())

    def test_deterministic_split(self):
        a,_,_,_=prepare(fixture(),config())
        b,_,_,_=prepare(fixture().sample(frac=1,random_state=6),config())
        pd.testing.assert_frame_equal(a,b)
        self.assertTrue(validate_frame(a)['passed'])

    def test_exact_duplicate_groups_merge(self):
        raw=fixture();raw.at[1,'human_output']=raw.at[0,'human_output']
        frame,_,edges,_=prepare(raw,config())
        self.assertEqual(frame.loc[frame.prompt_id.isin(['prompt-0','prompt-1']),'group_id'].nunique(),1)
        self.assertGreater(len(edges),0)

    def test_conflicting_labels_excluded(self):
        raw=fixture();raw.at[0,'model_output']=raw.at[0,'human_output']
        frame,_,_,audit=prepare(raw,config())
        self.assertEqual(audit['label_conflict_hashes'],1)
        self.assertFalse(frame.loc[frame.sample_id.eq('beemo:0:ai_raw'),'eligible'].iloc[0])
        self.assertFalse(frame.loc[frame.parent_id.eq('beemo:0:ai_raw'),'pair_eligible'].any())
        validate_frame(frame)

    def test_missing_parent_retains_audit(self):
        raw=fixture();raw.at[0,'model_output']=None
        frame,excluded,_,_=prepare(raw,config())
        self.assertFalse(frame.loc[frame.source_row_id.eq('0') & frame.parent_id.notna(),'eligible'].any())
        self.assertIn('missing_text',set(excluded.reason))
        validate_frame(frame)

    def test_short_text_not_main_task(self):
        raw=fixture();raw.at[0,'human_output']='Too short.'
        frame,_,_,_=prepare(raw,config())
        row=frame.loc[frame.sample_id.eq('beemo:0:human')].iloc[0]
        self.assertFalse(row.main_task)

    def test_duplicate_human_per_prompt(self):
        raw=fixture();raw.at[1,'prompt_id']=raw.at[0,'prompt_id'];raw.at[1,'prompt']=raw.at[0,'prompt']
        raw.at[1,'human_output']=raw.at[0,'human_output']
        frame,_,_,_=prepare(raw,config())
        part=frame.loc[frame.prompt_id.eq('prompt-0') & frame.variant.eq('human')]
        self.assertEqual(int(part.eligible.sum()),1)

    def test_join_matches_bruteforce(self):
        rng=random.Random(4)
        originals=[' '.join(rng.choices(['alpha','bravo','charlie','delta','echo','foxtrot','golf'],k=70)) for _ in range(15)]
        texts=originals+[s+' alpha' for s in originals]
        actual={tuple(sorted((a,b))) for a,b,_ in near_pairs(texts,.9,50,5)}
        sets=[]
        for text in texts:
            w=text.split();sets.append(set(tuple(w[i:i+5]) for i in range(len(w)-4)))
        expected={(i,j) for i in range(len(sets)) for j in range(i+1,len(sets)) if len(sets[i]&sets[j])/len(sets[i]|sets[j])>=.9}
        self.assertEqual(actual,expected)

    def test_validation_detects_leakage(self):
        frame,_,_,_=prepare(fixture(),config())
        frame.loc[0,'split']='test' if frame.loc[0,'split']!='test' else 'train'
        with self.assertRaises(ValueError):validate_frame(frame)

    def test_validation_detects_text_tamper(self):
        frame,_,_,_=prepare(fixture(),config());frame.loc[0,'text']='changed'
        with self.assertRaises(ValueError):validate_frame(frame)

    def test_parquet_roundtrip(self):
        frame,_,_,_=prepare(fixture(),config())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'data.parquet';frame.to_parquet(path,index=False)
            loaded=pd.read_parquet(path);validate_frame(loaded)
            pd.testing.assert_frame_equal(frame,loaded)

    def test_wrong_file_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'wrong.parquet';path.write_bytes(b'PAR1FAKEPAR1')
            self.assertEqual(fingerprints(path)['bytes'],12)
            with self.assertRaises(ValueError):verify(path,config())

if __name__=='__main__':unittest.main()
