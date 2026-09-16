import json, sys
from pathlib import Path
sys.path.insert(0, "src/eval")
from metrics import evaluate, print_report

def load(split):
    rows = [json.loads(l) for l in Path(f"data/gold/turkishtweets_{split}.jsonl").open(encoding="utf-8")]
    return [r["source"] for r in rows], [r["target"] for r in rows]

for split in ["dev", "test"]:
    src, tgt = load(split)
    m = evaluate(src, tgt, src)
    print_report(f"COPY baseline — {split}", m)
