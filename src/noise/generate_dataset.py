import json, random, sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import bozucu as B

random.seed(2026)
CLEAN = Path("data/clean_source/tatoeba_temiz.txt").read_text(encoding="utf-8").splitlines()
random.shuffle(CLEAN)

TARGET = 250_000
RECIPE = [
    ("keyboard", 0.25), ("dialect", 0.25), ("slang", 0.15),
    ("combination", 0.25), ("identity", 0.10),
]
REGIONS = ["general","general","general","aegean","blacksea","east"]

def generate(clean, kind):
    if kind == "identity":
        return clean, "identity", ""
    if kind == "keyboard":
        return B.keyboard_sentence(clean, rate=0.6), "keyboard", ""
    if kind == "slang":
        s = B.slang_corrupt(clean, rate=0.7); s = B.keyboard_sentence(s, rate=0.15)
        return s, "slang", ""
    if kind == "dialect":
        region = random.choice(REGIONS)
        s = B.dialect_corrupt(clean, region=region, rate=0.85)
        if random.random() < 0.4: s = B.slang_corrupt(s, rate=0.3)
        s = B.keyboard_sentence(s, rate=0.2)
        return s, "dialect", region
    region = random.choice(REGIONS)
    s = B.dialect_corrupt(clean, region=region, rate=0.7)
    s = B.slang_corrupt(s, rate=0.5)
    s = B.keyboard_sentence(s, rate=0.45)
    return s, "combination", region

plan = []
for kind, ratio in RECIPE:
    plan += [kind] * int(TARGET * ratio)
random.shuffle(plan)

out_path = Path("data/synthetic/synthetic_train.jsonl")
out_path.parent.mkdir(parents=True, exist_ok=True)
stats = {}; unchanged = 0; written = 0
ci = 0
with out_path.open("w", encoding="utf-8") as f:
    for kind in plan:
        clean = CLEAN[ci % len(CLEAN)]; ci += 1
        noisy, ntype, region = generate(clean, kind)
        if ntype != "identity" and noisy.strip() == clean.strip():
            noisy = B.keyboard_sentence(clean, rate=0.7)
            ntype = "keyboard"
            unchanged += 1
        rec = {"source": noisy, "target": clean, "noise_type": ntype, "region": region, "dataset": "synthetic"}
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats[ntype] = stats.get(ntype, 0) + 1
        written += 1

print(f"Generated pairs: {written}  ->  {out_path}")
print(f"Fell back to keyboard noise (no-op corruption caught): {unchanged}")
print("\nnoise_type distribution:")
for k,v in sorted(stats.items(), key=lambda x:-x[1]):
    print(f"  {k:12s}: {v:7d}  %{100*v/written:.1f}")
