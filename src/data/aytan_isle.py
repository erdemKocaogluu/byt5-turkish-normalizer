import json, re
from pathlib import Path

SRC = Path("data/raw/external/aytan/turkish_spell_validation.txt")
OUT = Path("data/aytan"); OUT.mkdir(parents=True, exist_ok=True)

rows=[]; skipped=0
for ln, line in enumerate(SRC.open(encoding="utf-8")):
    line=line.rstrip("\n")
    if "[ORG]" not in line: skipped+=1; continue
    noisy,clean=line.split("[ORG]",1)
    noisy=re.sub(r"\s+"," ",noisy).strip(); clean=re.sub(r"\s+"," ",clean).strip()
    if not noisy or not clean or len(clean.split())<2: skipped+=1; continue
    rows.append({"source":noisy,"target":clean,"noise_type":"aytan_typo","region":"",
                 "dataset":"aytan_lowercase"})
print(f"read: {len(rows)}  skipped: {skipped}")

empty=sum(1 for r in rows if not r["source"] or not r["target"])
same=sum(1 for r in rows if r["source"]==r["target"])
changed=len(rows)-same
gold=set()
for gp in ["turkishtweets_dev","turkishtweets_test"]:
    for l in open(f"data/gold/{gp}.jsonl",encoding="utf-8"):
        gold.add(json.loads(l)["target"].lower().strip(".,!?"))
leak=sum(1 for r in rows if r["target"].lower().strip(".,!?") in gold)

print(f"empty field: {empty} | source==target: {same} | changed: {changed}")
print(f"GOLD leak: {leak}/{len(rows)}  (should be 0)")

with (OUT/"aytan_train.jsonl").open("w",encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")
print(f"written: {OUT/'aytan_train.jsonl'} ({len(rows)})")
