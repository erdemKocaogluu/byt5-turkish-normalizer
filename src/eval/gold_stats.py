import json, collections
from pathlib import Path

rows = [json.loads(l) for p in ["dev","test"] for l in Path(f"data/gold/turkishtweets_{p}.jsonl").open(encoding="utf-8")]

cat = collections.Counter()
for r in rows:
    for tok_ann in r["ann"].split():
        for sub in tok_ann.split("&"):
            cat[sub] += 1

total = sum(v for k, v in cat.items())
print(f"{'CATEGORY':45s} {'COUNT':>7s} {'%':>7s}")
print("-"*62)
for k, v in cat.most_common(20):
    print(f"{k:45s} {v:7d} {100*v/total:6.1f}%")

print("\n--- REAL NOISE EXAMPLES (8 most-changed tweets) ---")
scored = sorted(rows, key=lambda r: -sum(1 for a,b in zip(r["pre"].split(), r["corrected"].split()) if a!=b))
for r in scored[:8]:
    print(f"  NOISY: {r['pre']}")
    print(f"  CLEAN: {r['corrected']}\n")
