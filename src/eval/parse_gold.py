import json, csv, random, re, unicodedata
from pathlib import Path

SRC = Path("data/gold/turkishtweets_raw.txt")
OUT = Path("data/gold")

rows, skipped = [], 0
with SRC.open(encoding="utf-8-sig") as f:
    header = f.readline()
    for ln, line in enumerate(f, start=2):
        parts = [p.strip() for p in line.rstrip("\n").split("\t")]
        parts = [p for p in parts if p != ""]
        if len(parts) < 3:
            skipped += 1
            continue
        raw, pre, corrected = parts[0], parts[1], parts[2]
        ann = parts[3] if len(parts) > 3 else ""
        rows.append({"line": ln, "raw": raw, "pre": pre, "corrected": corrected, "ann": ann})

print(f"read: {len(rows)}  skipped: {skipped}")

n_cap_start   = sum(1 for r in rows if r["corrected"][:1].isupper())
n_endpunct    = sum(1 for r in rows if r["corrected"].rstrip()[-1:] in ".!?")
n_identical   = sum(1 for r in rows if r["pre"] == r["corrected"])
n_len_tokmatch= sum(1 for r in rows if len(r["pre"].split()) == len(r["corrected"].split()))
tok_total = sum(len(r["corrected"].split()) for r in rows)
tok_changed = sum(
    sum(1 for a, b in zip(r["pre"].split(), r["corrected"].split()) if a != b)
    for r in rows if len(r["pre"].split()) == len(r["corrected"].split())
)
print(f"""
--- GOLD TARGET ANALYSIS ({len(rows)} tweets) ---
Starts with a capital letter    : {n_cap_start:5d}  (%{100*n_cap_start/len(rows):.1f})
Ends with . ! ?                 : {n_endpunct:5d}  (%{100*n_endpunct/len(rows):.1f})
pre == corrected (unchanged)    : {n_identical:5d}  (%{100*n_identical/len(rows):.1f})
Token count unchanged           : {n_len_tokmatch:5d}  (%{100*n_len_tokmatch/len(rows):.1f})
Total tokens                    : {tok_total}
Changed tokens (aligned rows)   : {tok_changed}  (%{100*tok_changed/tok_total:.1f})
""")

random.seed(42)
random.shuffle(rows)
dev, test = rows[:500], rows[500:]
for name, part in [("dev", dev), ("test", test)]:
    p = OUT / f"turkishtweets_{name}.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in part:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"written: {p}  ({len(part)} rows)")
