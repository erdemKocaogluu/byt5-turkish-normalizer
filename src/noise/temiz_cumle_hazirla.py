import re, json, random
from pathlib import Path

SRC = Path("data/raw/tur_sentences.tsv")
OUT = Path("data/clean_source/tatoeba_temiz.txt")
TR = set("çğıöşüÇĞİÖŞÜ")

def is_suitable(s):
    w = s.split()
    if not (3 <= len(w) <= 25): return False
    if len(s) > 180: return False
    if not (set(s) & TR): return False
    if re.search(r"https?://|www\.|@\w+|\d{4,}", s): return False
    if sum(c.isalpha() for c in s) / max(len(s),1) < 0.6: return False
    if s.count('"') > 2 or "—" in s: return False
    return True

seen = set(); clean = []
with SRC.open(encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3: continue
        s = parts[2].strip()
        key = s.lower()
        if key in seen: continue
        if is_suitable(s):
            seen.add(key); clean.append(s)

random.seed(42); random.shuffle(clean)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(clean), encoding="utf-8")
print(f"Raw: 748810  ->  Clean pool: {len(clean)} sentences")
print("Examples:")
for s in clean[:8]: print("  -", s)
