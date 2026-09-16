import argparse, json, re, unicodedata
from pathlib import Path
from collections import Counter
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import evaluate, print_report, _align_ops

ROOT = Path(__file__).resolve().parents[2]

NORMALIZE_GOLDS = {"tweets"}

def load_jsonl_pairs(path):
    src, tgt = [], []
    for line in open(path, encoding="utf-8"):
        d = json.loads(line)
        if d.get("source") and d.get("target"):
            src.append(d["source"]); tgt.append(d["target"])
    return src, tgt

def load_boun():
    text = open(ROOT/"data/raw/gold/boun/boun_source_test.txt", encoding="utf-8").read()
    targets_all = [l for l in open(ROOT/"data/raw/gold/boun/boun_target_test.txt", encoding="utf-8").read().split("\n")]
    blocks = [b for b in text.split("\n\n") if b.strip()]
    src, tgt = [], []
    for i, block in enumerate(blocks):
        lines = block.split("\n")
        s_line = next((l[2:] for l in lines if l.startswith("S ")), None)
        has_a = any(l.startswith("A ") for l in lines)
        if s_line and has_a and i < len(targets_all):
            t = targets_all[i].strip()
            if t and s_line.strip() != t:
                src.append(s_line.strip()); tgt.append(t)
    return src, tgt

GOLD_LOADERS = {
    "tweets": lambda: load_jsonl_pairs(ROOT/"data/raw/gold/turkish_tweets/tweets_clean.jsonl"),
    "boun":   load_boun,
}

EMOTICON = re.compile(r"^[:;=8xX]['`\-^]?[)(\]\[dDpPoO/\\|3><]+$|^<3+$|^\^_?\^$|^[)(]+$")
def _tr_lower(s): return s.replace("İ","i").replace("I","ı").lower()
def normalize(s):
    s = _tr_lower(s)
    out = []
    for t in s.split():
        if t.startswith("@") or t.startswith("#"): continue
        if EMOTICON.match(t): continue
        t = "".join(ch for ch in t if not (unicodedata.category(ch) in ("So","Sk","Cs") or ord(ch) >= 0x1F000))
        t = re.sub(r"[^\wçğıöşü']", " ", t)
        for p in t.split():
            p = p.strip("'")
            if p: out.append(p)
    return " ".join(out)

_ASC = {"ş":"s","ç":"c","ğ":"g","ı":"i","ö":"o","ü":"u"}
def _deascii(s): return "".join(_ASC.get(c, c) for c in s.lower())
_DEDA = {"de","da","te","ta","ki","mi","mı","mu","mü"}
_VOWELS = set("aeıioöuüâîû")
def _collapse(s): return re.sub(r"(.)\1+", r"\1", s)
def _novowel(s): return "".join(c for c in s if c not in _VOWELS)
def _clean(s): return re.sub(r"[^\wçğıöşü]", "", s.lower())
G_NAME = {"G1":"typo/letter","G2":"vowel","G3":"casing","G4":"repetition",
          "G5":"deascii","G6":"dialect","G7":"de/da/ki/mi","G8":"spacing"}

def _is_deda_piece(p):
    p = _clean(p)
    return p in _DEDA or p.startswith(("mı","mi","mu","mü"))

def _g_bucket(src, tgt):
    if src is None or tgt is None:
        return "G7" if _is_deda_piece(tgt or src) else "G8"
    if src == tgt: return None
    cs, ct = _clean(src), _clean(tgt)
    for d in _DEDA:
        if cs == ct + d or ct == cs + d: return "G7"
    if src.lower() == tgt.lower(): return "G3"
    if _deascii(src) == _deascii(tgt): return "G5"
    if _collapse(src.lower()) == _collapse(tgt.lower()): return "G4"
    if _novowel(cs) == _novowel(ct) and _novowel(cs): return "G2"
    return "G1"

def per_g_fixrate(sources, targets, preds, force_g=None):
    needed, fixed = Counter(), Counter()
    for s, t, p in zip(sources, targets, preds):
        st, tt, pt = s.split(), t.split(), p.split()
        ok = set(ti for tag, ti, pi in _align_ops(tt, pt) if tag == "equal")
        for tag, si, ti in _align_ops(st, tt):
            if tag == "equal": continue
            g = force_g or _g_bucket(st[si] if si is not None else None,
                                     tt[ti] if ti is not None else None)
            if g is None: continue
            needed[g] += 1
            if ti is not None and ti in ok: fixed[g] += 1
    return needed, fixed

def model_predict(model_dir, sources, beam=4):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    PREFIX = "düzelt: "
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=("byt5" in model_dir.lower()))
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()
    maxlen = 384 if "byt5" in model_dir.lower() else 128
    preds = []
    B = 32
    for i in range(0, len(sources), B):
        batch = [PREFIX + s for s in sources[i:i+B]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=maxlen).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, num_beams=beam, max_length=maxlen)
        preds += tok.batch_decode(out, skip_special_tokens=True)
    return preds

def evaluate_gold(model_dir, beam=4, sets=None, compare_copy=True):
    sets = sets or list(GOLD_LOADERS)
    results = {}
    for name in sets:
        src, tgt = GOLD_LOADERS[name]()
        preds = model_predict(model_dir, src, beam)
        if name in NORMALIZE_GOLDS:
            src_n = [normalize(s) for s in src]
            tgt_n = [normalize(t) for t in tgt]
            prd_n = [normalize(p) for p in preds]
        else:
            src_n, tgt_n, prd_n = src, tgt, preds
        record = {"n": len(src), "normalize": name in NORMALIZE_GOLDS,
                 "model": evaluate(src_n, tgt_n, prd_n)}
        if compare_copy:
            record["copy"] = evaluate(src_n, tgt_n, src_n)
        results[name] = record
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="models/<name>-tr-normalizer")
    ap.add_argument("--copy", action="store_true", help="no model: pred=source (baseline + harness test)")
    ap.add_argument("--gold", default="all", help="tweets | boun | all")
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--json", help="also write results to this JSON file (rapor_uret.py reads it)")
    args = ap.parse_args()
    if not args.model and not args.copy:
        ap.error("provide --model or --copy")

    sets = list(GOLD_LOADERS) if args.gold == "all" else [args.gold]
    all_results = {}
    for name in sets:
        src, tgt = GOLD_LOADERS[name]()
        preds = list(src) if args.copy else model_predict(args.model, src, args.beam)

        if name in NORMALIZE_GOLDS:
            src_n = [normalize(s) for s in src]
            tgt_n = [normalize(t) for t in tgt]
            prd_n = [normalize(p) for p in preds]
        else:
            src_n, tgt_n, prd_n = src, tgt, preds
        m = evaluate(src_n, tgt_n, prd_n)
        label = "COPY-baseline" if args.copy else Path(args.model).name
        print_report(f"{name}  [{label}]" + ("  (normalize)" if name in NORMALIZE_GOLDS else "  (raw)"), m)

        all_results[name] = {"n": len(src), "normalize": name in NORMALIZE_GOLDS,
                     ("copy" if args.copy else "model"): m}
        if args.json and not args.copy:
            all_results[name]["copy"] = evaluate(src_n, tgt_n, src_n)

        if name == "boun":
            needed, fixed = per_g_fixrate(src_n, tgt_n, prd_n, force_g="G7")
            source = "G7 (de/da benchmark)"
        else:
            needed = None
        if needed is not None:
            print(f"  ---- PER-G  [source: {source}] ----")
            print(f"     {'G':4s} {'type':12s} {'needed':>9s} {'fixed':>10s} {'success%':>8s}")
            for g in ["G1","G2","G3","G4","G5","G6","G7","G8"]:
                n, fx = needed.get(g, 0), fixed.get(g, 0)
                if n == 0: continue
                print(f"     {g:4s} {G_NAME[g]:12s} {n:>9d} {fx:>10d} {100*fx/n:>6.0f}%")

    if args.json:
        import time
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"model": (Path(args.model).name if args.model else "copy-baseline"),
                       "beam": args.beam, "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                       "sets": all_results}, f, ensure_ascii=False, indent=2)
        print(f"\nGold results written to JSON: {args.json}")

if __name__ == "__main__":
    main()
