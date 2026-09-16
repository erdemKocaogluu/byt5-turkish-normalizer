import csv, re, sys, json
from pathlib import Path
csv.field_size_limit(10**9)

FIXES = [
    (r"\byada\b", "ya da"),
    (r"\bbirşey\b", "bir şey"), (r"\bbirşeyler\b", "bir şeyler"),
    (r"\bherşey\b", "her şey"), (r"\bhiçbirşey\b", "hiçbir şey"),
    (r"\bhiç bir\b", "hiçbir"), (r"\bbir çok\b", "birçok"),
    (r"\bfarketmez\b", "fark etmez"), (r"\byanlız\b", "yalnız"),
    (r"\bherkez\b", "herkes"), (r"\bherhangibir\b", "herhangi bir"),
    (r"\bbirsürü\b", "bir sürü"), (r"\bherbir\b", "her bir"),
    (r"\bhiçbirşeyi\b", "hiçbir şeyi"), (r"\bbirşeyi\b", "bir şeyi"),
]
def _case_repl(repl):
    def f(m):
        s = m.group(0)
        return (repl[:1].upper()+repl[1:]) if s[:1].isupper() else repl
    return f
_COMPILED = [(re.compile(p, re.IGNORECASE), _case_repl(r)) for p, r in FIXES]

def fix(t):
    for rx, fn in _COMPILED:
        t = rx.sub(fn, t)
    return re.sub(r"\s+", " ", t).strip()

def clean(in_csv, out_jsonl, min_words=3, max_words=40):
    n=kept=0
    dropped_encoding=dropped_length=dropped_empty=dropped_same=0
    fixed_n=0
    with open(in_csv, encoding="utf-8") as f, open(out_jsonl, "w", encoding="utf-8") as out:
        for r in csv.DictReader(f):
            n+=1
            s=(r.get("sentence") or "").strip()
            c=(r.get("correction") or "").strip()
            if not(s and c): dropped_empty+=1; continue
            if "�" in c or "�" in s: dropped_encoding+=1; continue
            wc=len(c.split())
            if wc<min_words or wc>max_words: dropped_length+=1; continue
            c2=fix(c)
            if c2!=c: fixed_n+=1
            if s==c2: dropped_same+=1; continue
            out.write(json.dumps({"source":s,"target":c2,
                                  "noise_type":"oscar_gec","region":"",
                                  "dataset":"oscar_clean"}, ensure_ascii=False)+"\n")
            kept+=1
    print(f"[{Path(in_csv).name}]")
    print(f"  rows read           : {n:,}")
    print(f"  targets fixed       : {fixed_n:,}")
    print(f"  dropped (encoding)  : {dropped_encoding:,}")
    print(f"  dropped (length)    : {dropped_length:,}")
    print(f"  dropped (empty/same): {dropped_empty+dropped_same:,}")
    print(f"  CLEAN PAIRS WRITTEN : {kept:,}\n")
    return kept

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv)>1 else "data/raw/oscar_gec_val.csv"
    dst = sys.argv[2] if len(sys.argv)>2 else "data/raw/oscar_temiz_val.jsonl"
    clean(src, dst)
