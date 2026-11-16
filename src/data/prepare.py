"""Expand Beemo, group related/duplicate texts, and write reproducible data splits.

Fixture tests do not constitute a completed Beemo experiment. Production CLI
always verifies the pinned official file before processing.
"""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import unicodedata

import numpy as np
import pandas as pd
from .download import ROOT, verify

WORDS = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)*")
REQUIRED = {"id", "category", "model", "prompt_id", "prompt", "model_output",
            "human_output", "human_edits", "llama-3.1-70b_edits", "gpt-4o_edits"}

def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def clean_text(value):
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected text, received {type(value).__name__}")
    text = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return text or None

def parse_edits(value):
    """Support native arrays/dicts and JSON/Python-literal serialized containers.

    Unknown schema is rejected; never use eval or stringify an entire container.
    """
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return []
    if isinstance(value, np.ndarray): value = value.tolist()
    if isinstance(value, str):
        if not value.strip(): return []
        try: value = json.loads(value)
        except json.JSONDecodeError:
            try: value = ast.literal_eval(value)
            except (ValueError, SyntaxError) as exc: raise ValueError("Unrecognized serialized edit field") from exc
    if isinstance(value, dict): value = [value]
    if not isinstance(value, list): raise ValueError("Edit field must be a list or dictionary")
    result = []
    for item in value:
        if not isinstance(item, dict) or not item:
            raise ValueError("Expected nonempty P1/P2/P3 dictionary")
        for key, text in item.items():
            if key not in {"P1", "P2", "P3"}: raise ValueError(f"Unknown edit key: {key}")
            result.append((key, clean_text(text)))
    keys = [k for k, _ in result]
    if len(keys) != len(set(keys)): raise ValueError("Repeated edit prompt key")
    return sorted(result)

class DisjointSet:
    def __init__(self, keys): self.parent = {key:key for key in keys}
    def find(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]; key = self.parent[key]
        return key
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a,b)] = min(a,b)

def near_pairs(texts, threshold=0.9, min_words=50, width=5):
    """Exact token-shingle Jaccard join with frequency-ordered prefix filtering.

    All qualifying pairs are verified, not just nearest-neighbor guesses.
    Prefixes must share an element for similarity >= threshold.
    """
    sets = []
    for text in texts:
        words = WORDS.findall(normalize(text))
        sets.append(set(tuple(words[j:j+width]) for j in range(len(words)-width+1))
                    if len(words) >= min_words else set())
    frequency = Counter(sh for s in sets for sh in s)
    order = {sh:i for i,sh in enumerate(sorted(frequency, key=lambda x:(frequency[x],x)))}
    index = defaultdict(list)
    for i, current in sorted(enumerate(sets), key=lambda item:(len(item[1]),item[0])):
        if not current: continue
        prefix_len = len(current) - math.ceil(threshold*len(current)) + 1
        prefix = sorted(current, key=order.get)[:prefix_len]
        candidates = set()
        for sh in prefix:
            candidates.update(j for j in index[sh] if len(sets[j]) >= threshold*len(current))
        for j in sorted(candidates):
            score = len(current & sets[j]) / len(current | sets[j])
            if score >= threshold: yield j,i,score
        for sh in prefix: index[sh].append(i)

def expand(raw, config):
    missing = REQUIRED - set(raw.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    if raw["id"].isna().any() or raw["id"].astype(str).duplicated().any():
        raise ValueError("Source id must be non-null and unique")
    records, exclusions, prompts = [], [], {}
    for source in raw.to_dict("records"):
        sid = str(source["id"]); pid = clean_text(source["prompt_id"])
        prompt = clean_text(source["prompt"])
        category = clean_text(source["category"]); generator = clean_text(source["model"])
        if not all([pid,prompt,category,generator]): raise ValueError(f"Missing metadata in source row {sid}")
        if pid in prompts and prompts[pid] != normalize(prompt):
            raise ValueError(f"Same prompt_id has different prompt text: {pid}")
        prompts[pid] = normalize(prompt)
        raw_id = f"beemo:{sid}:ai_raw"
        def add(variant, text, editor=None, edit_prompt=None, parent=None):
            text = clean_text(text)
            sample_id = f"beemo:{sid}:{variant}" + (f":{editor}:{edit_prompt}" if edit_prompt else "")
            if text is None:
                exclusions.append({"sample_id":sample_id,"reason":"missing_text"}); return
            records.append({"sample_id":sample_id,"source_row_id":sid,"dataset_revision":config["revision"],
                "prompt_id":pid,"prompt_sha256":digest(normalize(prompt)),"text":text,"variant":variant,
                "label":0 if variant=="human" else 1,"category":category,
                "generator":None if variant=="human" else generator,"editor":editor,"edit_prompt_id":edit_prompt,
                "parent_id":parent,"word_count":len(WORDS.findall(text)),"char_count":len(text),
                "text_sha256":digest(text),"normalized_sha256":digest(normalize(text))})
        add("human",source["human_output"])
        add("ai_raw",source["model_output"])
        add("human_edited",source["human_edits"],editor="expert",parent=raw_id)
        for col in ["llama-3.1-70b_edits","gpt-4o_edits"]:
            try: edits = parse_edits(source[col])
            except ValueError as exc: raise ValueError(f"Row {sid}, {col}: {exc}") from exc
            for key,text in edits: add("llm_edited",text,col.removesuffix("_edits"),key,raw_id)
    if not records: raise ValueError("No text records")
    return pd.DataFrame(records).sort_values("sample_id").reset_index(drop=True), exclusions

def assign_splits(frame, config):
    rng = random.Random(config["seed"])
    strata = defaultdict(list)
    for group, part in frame.groupby("group_id",sort=True):
        strata["|".join(sorted(part.category.unique()))].append(group)
    mapping = {}; names = ["train","val","test"]; ratios=config["split_ratios"]
    if len(ratios)!=3 or any(r<=0 for r in ratios) or not math.isclose(sum(ratios),1):
        raise ValueError("Three positive split ratios must sum to one")
    totals=Counter()
    for stratum in sorted(strata):
        groups=sorted(strata[stratum]);rng.shuffle(groups); n=len(groups)
        counts=[math.floor(n*r) for r in ratios]
        for k in sorted(range(3),key=lambda k:(-(n*ratios[k]-counts[k]),k))[:n-sum(counts)]: counts[k]+=1
        if n>=3:
            for k in [1,2]:
                if counts[k]==0:
                    donor=max(range(3),key=lambda j:counts[j]);counts[donor]-=1;counts[k]+=1
        if n<3:
            for g in groups:
                k=min(range(3),key=lambda j:totals[names[j]]/ratios[j])
                mapping[g]=names[k];totals[names[k]]+=1
        else:
            start=0
            for name,count in zip(names,counts):
                for g in groups[start:start+count]: mapping[g]=name
                totals[name]+=count;start+=count
    return frame["group_id"].map(mapping)

def prepare(raw, config):
    frame, exclusions = expand(raw,config)
    dsu=DisjointSet(frame.prompt_id.unique())
    edges=[]
    for key in ["prompt_sha256","normalized_sha256"]:
        for _,part in frame.groupby(key,sort=True):
            pids=sorted(part.prompt_id.unique())
            for pid in pids[1:]:
                dsu.union(pids[0],pid)
                edges.append({"left_prompt":pids[0],"right_prompt":pid,"kind":key,"similarity":1.0})
    unique=frame.drop_duplicates("normalized_sha256").reset_index(drop=True)
    for i,j,score in near_pairs(unique.text.tolist(),config["near_duplicate_jaccard"],config["near_duplicate_min_words"],config["shingle_words"]):
        a,b=unique.iloc[i],unique.iloc[j]
        if a.prompt_id != b.prompt_id:
            dsu.union(a.prompt_id,b.prompt_id)
            edges.append({"left_prompt":a.prompt_id,"right_prompt":b.prompt_id,"kind":"token_shingle_jaccard","similarity":score})
    frame["group_id"] = frame.prompt_id.map(lambda p:"group:"+digest(dsu.find(p))[:20])
    frame["split"] = assign_splits(frame,config)
    frame["exclusion_reason"]=""
    conflicts=set(frame.groupby("normalized_sha256").label.nunique().loc[lambda x:x>1].index)
    seen_human=set()
    ids=set(frame.sample_id)
    for i,row in frame.iterrows():
        reasons=[]
        if row.normalized_sha256 in conflicts: reasons.append("conflicting_origin_labels")
        if not config["min_words"] <= row.word_count <= config["max_words"]: reasons.append("outside_word_range")
        if pd.notna(row.parent_id) and row.parent_id not in ids: reasons.append("missing_parent")
        if row.variant=="human":
            key=(row.prompt_id,row.normalized_sha256)
            if key in seen_human: reasons.append("repeated_human_for_prompt")
            seen_human.add(key)
        frame.at[i,"exclusion_reason"]=";".join(reasons)
        if reasons: exclusions.append({"sample_id":row.sample_id,"reason":";".join(reasons)})
    frame["eligible"]=frame.exclusion_reason.eq("")
    eligible_ids=set(frame.loc[frame.eligible,"sample_id"])
    frame["pair_eligible"] = frame.eligible & frame.parent_id.isin(eligible_ids)
    frame["main_task"] = frame.eligible & frame.variant.isin(["human","ai_raw"])
    audit={"schema_version":config["schema_version"],"source_rows":len(raw),"expanded_texts":len(frame),
           "prompt_groups":frame.prompt_id.nunique(),"leakage_groups":frame.group_id.nunique(),
           "label_conflict_hashes":len(conflicts),"eligible_texts":int(frame.eligible.sum()),
           "eligible_main_texts":int(frame.main_task.sum()),"eligible_pairs":int(frame.pair_eligible.sum()),
           "variant_counts":frame.variant.value_counts().to_dict(),
           "split_group_counts":frame.groupby("split").group_id.nunique().to_dict(),
           "main_counts":frame.loc[frame.main_task].groupby(["split","label"]).size().reset_index(name="n").to_dict("records"),
           "note":"Normalization/shingle grouping is a defined leakage screen, not a guarantee against all semantic paraphrases."}
    return frame, pd.DataFrame(exclusions,columns=["sample_id","reason"]), pd.DataFrame(edges,columns=["left_prompt","right_prompt","kind","similarity"]),audit

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",type=Path,default=ROOT/"configs/data.json")
    ap.add_argument("--input",type=Path,default=ROOT/"data/raw/dataset.parquet")
    ap.add_argument("--output",type=Path,default=ROOT/"data/processed")
    args=ap.parse_args();cfg=json.loads(args.config.read_text(encoding="utf-8"))
    info=verify(args.input,cfg);raw=pd.read_parquet(args.input)
    frame,excluded,edges,audit=prepare(raw,cfg)
    from .validate import validate_frame
    checks=validate_frame(frame,strict=True)
    args.output.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(args.output/"texts.parquet",index=False)
    frame.drop(columns="text").to_csv(args.output/"split_manifest.csv",index=False,encoding="utf-8-sig")
    excluded.to_csv(args.output/"audit_exclusions.csv",index=False,encoding="utf-8-sig")
    edges.to_csv(args.output/"group_edges.csv",index=False,encoding="utf-8-sig")
    for split in ["train","val","test"]:
        frame.loc[frame.main_task & frame.split.eq(split)].to_parquet(args.output/f"{split}_main.parquet",index=False)
    audit.update({"source":info,"config":cfg,"checks":checks,"status":"official_data_processed",
                  "manifest_sha256":hashlib.sha256((args.output/"split_manifest.csv").read_bytes()).hexdigest()})
    (args.output/"audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(audit,indent=2,ensure_ascii=False))

if __name__=="__main__":main()
