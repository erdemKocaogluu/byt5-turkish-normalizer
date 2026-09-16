import json, random, sys, os, csv, argparse, collections
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import bozucu as B
import llm_uretim as L

def args():
    p=argparse.ArgumentParser()
    p.add_argument("--sample", action="store_true", help="generate a small sample (20K), distribution only")
    return p.parse_args()

A = args()
random.seed(2026)

CLEAN = (Path("data/clean_source/tatoeba_temiz.txt").read_text(encoding="utf-8").splitlines()
         + Path("data/clean_source/gecturk_temiz.txt").read_text(encoding="utf-8").splitlines())
random.shuffle(CLEAN)

TARGET_RULE = 20_000 if A.sample else 450_000

RECIPE = [
    ("dialect",           0.22),
    ("slang",             0.13),
    ("char",              0.25),
    ("conjunction",       0.08),
    ("question_particle", 0.08),
    ("lexical",           0.08),
    ("spacing",           0.06),
    ("apostrophe",        0.04),
    ("deascii",           0.03),
    ("combination",       0.03),
]
REGIONS = ["general","general","aegean","blacksea","east","central_anatolia"]

def _choose(weights):
    kinds=list(weights); return random.choices(kinds, weights=[weights[t] for t in kinds])[0]

def _combination(clean):
    region=random.choice(REGIONS); s=clean; applied=[]
    s2,nt,_=B.corrupt_type(s,"dialect",region=region)
    if nt: s=s2; applied.append(("dialect",region))
    s2=B.slang_corrupt(s,0.5)
    if s2!=s: s=s2; applied.append(("slang",""))
    slug=_choose(B.CHAR_WEIGHTS)
    s2,cnt,_=B.corrupt_type(s,slug)
    if cnt: s=s2; applied.append((cnt,""))
    if len(applied)>=2:
        region_used = next((u[1] for u in applied if u[0]=="dialect"), "")
        return s,"combination",region_used
    if len(applied)==1: return s,applied[0][0],applied[0][1]
    s,_=B.char_corrupt(clean,"random_keyboard"); return s,"random_keyboard",""

def generate(clean, kind):
    if kind=="char":
        slug=_choose(B.CHAR_WEIGHTS)
        s,nt,_=B.corrupt_type(clean, slug)
        if nt is None:
            s,ap=B.char_corrupt(clean,"random_keyboard")
            nt="random_keyboard" if ap else None
        return s, nt, ""
    if kind=="combination":
        return _combination(clean)
    if kind=="dialect":
        return B.corrupt_type(clean,"dialect", region=random.choice(REGIONS))
    return B.corrupt_type(clean, kind)

POOL_TYPES=["dialect","slang","lexical","conjunction","question_particle","apostrophe","deascii"]
print("Preparing pools...")
pools={t:[] for t in POOL_TYPES}
for s in CLEAN:
    for t in POOL_TYPES:
        if B.fits(s,t): pools[t].append(s)
for t in POOL_TYPES:
    print(f"  pool[{t:18s}]: {len(pools[t]):>7,}")

def from_pool(kind, counters):
    key = kind if kind in pools else ("dialect" if kind in ("dialect","combination") else "all")
    pool = pools.get(key) or CLEAN
    i = counters.get(key,0); counters[key]=i+1
    return pool[i % len(pool)]

plan=[]
for kind,ratio in RECIPE: plan += [kind]*int(TARGET_RULE*ratio)
random.shuffle(plan)
counters={}; records=[]
for kind in plan:
    noisy=nt=region=None
    for _ in range(4):
        clean=from_pool(kind, counters)
        noisy,nt,region=generate(clean,kind)
        if nt is not None and noisy.strip()!=clean.strip():
            break
    if nt is None or noisy.strip()==clean.strip():
        noisy=B.keyboard_sentence(clean,0.7); nt="random_keyboard"; region=""
    records.append({"source":noisy,"target":clean,"noise_type":nt,"region":region,"dataset":"tatoeba_rule"})

for s,t,nt,region in L.all_pairs():
    records.append({"source":s,"target":t,"noise_type":nt,"region":region,"dataset":"llm_manual"})
    for _ in range(3):
        records.append({"source":B.keyboard_sentence(s,0.3),"target":t,
                         "noise_type":nt,"region":region,"dataset":"llm_manual_aug"})

for clean in L.SLANG_CLEAN_SEEDS:
    records.append({"source":clean,"target":clean,"noise_type":"identity","region":"","dataset":"llm_seed"})
    for _ in range(6):
        s=B.slang_corrupt(clean, rate=0.75)
        if random.random()<0.4: s=B.dialect_corrupt(s, rate=0.4)
        s=B.keyboard_sentence(s, 0.3)
        records.append({"source":s,"target":clean,"noise_type":"slang","region":"","dataset":"llm_seed"})

random.shuffle(records)

ntype_counts=collections.Counter(r["noise_type"] for r in records)
dataset_counts=collections.Counter(r["dataset"] for r in records)
print(f"\nTOTAL: {len(records):,}")
print("noise_type distribution:")
for t,v in ntype_counts.most_common(): print(f"  {t:20s} {v:>8,}  %{v/len(records)*100:.1f}")
print("dataset:", dict(dataset_counts))

if A.sample:
    print("\n(--sample: no file written, distribution only)")
    print("\n--- EXAMPLE PAIRS (by type) ---")
    seen=set()
    for r in records:
        if r["noise_type"] not in seen:
            seen.add(r["noise_type"])
            print(f"  [{r['noise_type']:20s}] {r['source']}  ->  {r['target']}")
    sys.exit(0)

val=records[:6000]; train=records[6000:]
Path("data/synthetic").mkdir(parents=True,exist_ok=True)
for name,part in [("train",train),("val",val)]:
    with open(f"data/synthetic/synthetic_{name}.jsonl","w",encoding="utf-8") as f:
        for r in part: f.write(json.dumps(r,ensure_ascii=False)+"\n")
with open("data/synthetic/sample_3000.csv","w",encoding="utf-8-sig",newline="") as f:
    w=csv.writer(f); w.writerow(["noisy (input)","clean (target)","noise_type","region","dataset"])
    for r in records[:3000]:
        w.writerow([r["source"],r["target"],r["noise_type"],r["region"],r["dataset"]])
print(f"\nWritten: synthetic_train ({len(train):,}) + synthetic_val ({len(val):,})")
