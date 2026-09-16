import json, re, sys, collections, statistics
sys.path.insert(0,"src/noise")
from sozlukler import DIALECT_RULES

TRAIN="data/final/train.jsonl"; VAL="data/final/val.jsonl"

def load(p): return [json.loads(l) for l in open(p,encoding="utf-8")]
print("Loading..."); train=load(TRAIN); val=load(VAL)
N=len(train)
print(f"train={N:,}  val={len(val):,}\n")
issues=collections.Counter(); example={}

def flag(code, r):
    issues[code]+=1
    if code not in example: example[code]=r

for r in train:
    s=r.get("source","") or ""; t=r.get("target","") or ""
    if not s.strip() or not t.strip(): flag("empty_field",r); continue
    if "�" in s or "�" in t: flag("encoding_garbage",r)
    same=(s.strip()==t.strip())
    if same and r["noise_type"]!="identity": flag("same_but_not_identity",r)
    if r["noise_type"]=="identity" and not same: flag("identity_but_different",r)
    ls,lt=len(s.split()),len(t.split())
    if lt>0 and (ls/lt>2.5 or ls/lt<0.4): flag("bad_length_ratio",r)
    if r["noise_type"]=="dialect":
        region=r.get("region","general") or "general"
        if region in DIALECT_RULES and not any(re.search(d,t,re.I) for d,_ in DIALECT_RULES[region]):
            flag("dialect_but_no_rule",r)
    for bad in ["birşey","herşey"," yada ","hiçbirşey"]:
        if bad in (" "+t.lower()+" "): flag("target_has_error",r); break
    if lt<2: flag("target_too_short",r)
    if lt>45: flag("target_too_long",r)

print("="*55)
print("  ISSUE SCAN (on train)")
print("="*55)
if not issues: print("  No issues found.")
for k,v in issues.most_common():
    print(f"  {k:26s}: {v:7d}  (%{100*v/N:.2f})")
    e=example[k]
    print(f"       example: {e['source'][:60]!r} -> {e['target'][:60]!r} [{e['noise_type']}]")

train_set=set((r["source"],r["target"]) for r in train)
leak=sum(1 for r in val if (r["source"],r["target"]) in train_set)
print(f"\n  train/val leakage (same pair): {leak}/{len(val)}")

counts=collections.Counter((r["source"],r["target"]) for r in train)
dupes=sum(1 for v in counts.values() if v>1)
print(f"  duplicate unique pairs in train: {dupes}")

missing=sum(1 for r in train if not all(k in r for k in ["source","target","noise_type"]))
print(f"  rows with missing fields: {missing}")
