"""Download or import the pinned official Beemo file, verifying bytes before use."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]

def fingerprints(path):
    path = Path(path)
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "git_blob_sha1": hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()}

def verify(path, config):
    info = fingerprints(path)
    if info["bytes"] != config["expected_bytes"] or info["git_blob_sha1"] != config["expected_git_blob_sha1"]:
        raise ValueError("File does not match pinned official data. Do not silently accept another revision.")
    if Path(path).read_bytes()[:4] != b"PAR1":
        raise ValueError("Expected Parquet magic; received a different file type.")
    return info

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/data.json")
    ap.add_argument("--input", type=Path, help="Import a file downloaded using a browser")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    folder = ROOT / "data/raw"; folder.mkdir(parents=True, exist_ok=True)
    dest = folder / cfg["filename"]
    url = f'https://raw.githubusercontent.com/{cfg["repository"]}/{cfg["revision"]}/{cfg["filename"]}'
    if args.input:
        verify(args.input, cfg)
        if args.input.resolve() != dest.resolve():
            shutil.copyfile(args.input, dest)
    elif not dest.exists():
        temp = dest.with_suffix(".download")
        try:
            with urllib.request.urlopen(url, timeout=30) as response, temp.open("wb") as output:
                shutil.copyfileobj(response, output)
            verify(temp, cfg)
            temp.replace(dest)
        finally:
            temp.unlink(missing_ok=True)
    info = verify(dest, cfg)
    info.update({"repository":cfg["repository"], "revision":cfg["revision"], "url":url,
                 "verified_at":datetime.now(timezone.utc).isoformat(), "status":"official_bytes_verified"})
    (folder / "source_manifest.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))

if __name__ == "__main__":
    try: main()
    except Exception as exc:
        raise SystemExit(f"Download/import failed: {exc}\nDownload dataset.parquet in a browser, then run with --input PATH.")
