"""Independent checks for the standardized long table; fails closed."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .download import ROOT
from .prepare import digest,normalize,WORDS

def validate_frame(frame,strict=True):
    def require(condition,message):
        if not condition:raise ValueError(message)
    require(len(frame)>0,"Empty data")
    require(frame.sample_id.is_unique,"Duplicate sample_id")
    require(frame.split.isin(["train","val","test"]).all(),"Invalid split")
    require(frame.groupby("group_id").split.nunique().max()==1,"group_id crosses splits")
    require(frame.groupby("prompt_id").split.nunique().max()==1,"prompt_id crosses splits")
    require(frame.groupby("normalized_sha256").split.nunique().max()==1,"Duplicate text crosses splits")
    require(frame.groupby("prompt_sha256").split.nunique().max()==1,"Duplicate prompt crosses splits")
    for row in frame.itertuples():
        require(row.text_sha256==digest(row.text),"Text hash mismatch")
        require(row.normalized_sha256==digest(normalize(row.text)),"Normalized hash mismatch")
        require(row.word_count==len(WORDS.findall(row.text)),"Word count mismatch")
        require(row.label==(0 if row.variant=="human" else 1),"Wrong origin label")
    byid=frame.set_index("sample_id")
    for row in frame.loc[frame.parent_id.notna()].itertuples():
        if row.parent_id not in byid.index:
            require(not row.eligible,"Eligible edit has missing parent");continue
        parent=byid.loc[row.parent_id]
        require(parent.variant=="ai_raw","Edit parent is not AI original")
        require(parent.group_id==row.group_id and parent.split==row.split,"Parent/child grouping mismatch")
        require(bool(row.pair_eligible)==(bool(row.eligible) and bool(parent.eligible)),"Invalid pair eligibility")
    require((frame.main_task == (frame.eligible & frame.variant.isin(["human","ai_raw"]))).all(),"main_task flag mismatch")
    require(frame.loc[frame.eligible].groupby("normalized_sha256").label.nunique().max()<=1,"Conflicting labels in eligible data")
    if strict:
        for split in ["train","val","test"]:
            require(set(frame.loc[frame.main_task & frame.split.eq(split),"label"])=={0,1},f"Both classes required in {split}")
    return {"passed":True,"rows":len(frame),"strict_both_classes":strict}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",type=Path,default=ROOT/"data/processed/texts.parquet")
    args=ap.parse_args();print(json.dumps(validate_frame(pd.read_parquet(args.input)),indent=2))

if __name__=="__main__":main()
