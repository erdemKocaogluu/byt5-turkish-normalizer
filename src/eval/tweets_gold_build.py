import json, re, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT/"data/raw/gold/turkish_tweets/data.txt"
OUT = ROOT/"data/raw/gold/turkish_tweets/tweets_clean.jsonl"

EMOTICON = re.compile(r"^[:;=8xX]['`\-^]?[)(\]\[dDpPoO/\\|3><]+$|^<3+$|^\^_?\^$|^[)(]+$")
def is_emoji(t): return any(unicodedata.category(ch) in ("So","Sk","Cs") or ord(ch) >= 0x1F000 for ch in t)
def is_junk(t): return t.startswith("@") or t.startswith("#") or is_emoji(t) or bool(EMOTICON.match(t))

def main():
    n_in = n_out = n_foreign = n_empty = 0
    with open(OUT, "w", encoding="utf-8") as w:
        for idx, line in enumerate(open(SRC, encoding="utf-8-sig")):
            if idx == 0:
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 4: continue
            raw, corr, ann = c[0].split(), c[2].split(), c[3].split()
            if not raw or not corr: continue
            if any(("foreign" in x or "neologism" in x) for x in ann):
                n_foreign += 1; continue
            n_in += 1
            src_toks = [t.replace("|", "") for t in raw if not is_junk(t)]
            tgt_toks = [t.replace("|", " ") for t in corr if not is_junk(t)]
            source = " ".join(src_toks).strip()
            target = " ".join(tgt_toks).strip()
            if not source or not target:
                n_empty += 1; continue
            w.write(json.dumps({"source": source, "target": target}, ensure_ascii=False) + "\n")
            n_out += 1
    print(f"dropped (foreign/neologism): {n_foreign} | dropped (empty): {n_empty} | written: {n_out}")

if __name__ == "__main__":
    main()
