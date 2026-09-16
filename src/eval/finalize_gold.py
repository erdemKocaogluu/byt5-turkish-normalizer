import json, re
from pathlib import Path

def clean_target(s):
    s = s.replace("|", " ")
    return re.sub(r"\s+", " ", s).strip()

for split in ["dev", "test"]:
    p = Path(f"data/gold/turkishtweets_{split}.jsonl")
    rows = [json.loads(l) for l in p.open(encoding="utf-8")]
    for r in rows:
        r["source"] = r["pre"]
        r["target"] = clean_target(r["corrected"])
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_split = sum(1 for r in rows if "|" in r["corrected"])
    print(f"{split}: {len(rows)} rows, {n_split} contain a word split (%{100*n_split/len(rows):.1f})")

rows = [json.loads(l) for s in ["dev","test"] for l in Path(f"data/gold/turkishtweets_{s}.jsonl").open(encoding="utf-8")]
same = sum(1 for r in rows if len(r["source"].split()) == len(r["target"].split()))
print(f"\ntoken count unchanged after unpacking: {same}/{len(rows)} (%{100*same/len(rows):.1f})")
print("\nEXAMPLE (source -> target):")
for r in rows[:3]:
    print(f"  IN : {r['source']}")
    print(f"  OUT: {r['target']}\n")
