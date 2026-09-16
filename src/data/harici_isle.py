import json, re, sys, collections
from pathlib import Path
from datasets import load_dataset

OUT = Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)

def _norm(s):
    return re.sub(r"\s+"," ", re.sub(r"[^\wğüşıöçĞÜŞİÖÇ ]","", s.lower())).strip()

def gold_pool():
    pool=set()
    for p in ["data/gold/turkishtweets_test.jsonl","data/gold/turkishtweets_dev.jsonl"]:
        if Path(p).exists():
            for l in Path(p).open(encoding="utf-8"):
                r=json.loads(l)
                pool.add(_norm(r.get("source",""))); pool.add(_norm(r.get("target","")))
    pool.discard("")
    return pool
GOLD=gold_pool()
print(f"Gold pool: {len(GOLD)} unique normalized sentences (for leak checking)")

def check_convention(t):
    t=t.strip()
    if not t: return False
    starts_ok = t[0].isupper() or not t[0].isalpha()
    ends_ok = t[-1] in ".!?…\"')]"
    return starts_ok, ends_ok

def process(pairs, dataset):
    out=[]; stat=collections.Counter()
    seen=set()
    start_n=end_n=0
    for s,t in pairs:
        s=(s or "").strip(); t=(t or "").strip()
        stat["raw"]+=1
        if not s or not t: stat["empty"]+=1; continue
        if _norm(s) in GOLD or _norm(t) in GOLD: stat["gold_leak"]+=1; continue
        foreign = re.findall(r"[^\x00-\x7fğüşıöçĞÜŞİÖÇ\s]", t)
        if len(foreign) > max(2, 0.05*len(t)): stat["foreign"]+=1; continue
        if len(t.split())>60: stat["too_long"]+=1; continue
        key=(s,t)
        if key in seen: stat["duplicate"]+=1; continue
        seen.add(key)
        b,e=check_convention(t)
        start_n+=b; end_n+=e
        out.append({"source":s,"target":t,"noise_type":"external","region":"","dataset":dataset})
        stat["kept"]+=1
    n=max(1,stat["kept"])
    print(f"\n[{dataset}]")
    for k in ["raw","kept","empty","gold_leak","foreign","too_long","duplicate"]:
        if stat[k]: print(f"    {k:14s}: {stat[k]:,}")
    print(f"    convention    : target %{100*start_n/n:.0f} starts uppercase, %{100*end_n/n:.0f} ends with punctuation")
    identity=sum(1 for r in out if r['source']==r['target'])
    print(f"    identity rate : %{100*identity/n:.0f} (source==target)")
    return out

records=[]

ds=load_dataset("asimokby/Turkish-GPT-GEC")
c=[(r["sentence"],r["correction"]) for s in ds for r in ds[s]]
records += process(c, "turkish_gpt_gec")

with (OUT/"harici_train.jsonl").open("w",encoding="utf-8") as f:
    for r in records: f.write(json.dumps(r,ensure_ascii=False)+"\n")

print("\n"+"="*60)
print(f"TOTAL vetted external pairs: {len(records):,}")
kn=collections.Counter(r["dataset"] for r in records)
for k,v in kn.most_common(): print(f"  {k:22s} {v:>7,}")
print(f"Written: {OUT/'harici_train.jsonl'}")
