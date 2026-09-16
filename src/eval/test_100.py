import os, sys, json, random
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

sys.path.insert(0, os.path.dirname(__file__))
import metrics

ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL  = os.path.join(ROOT, "models/byt5-small-tr-normalizer")
TEST   = os.path.join(ROOT, "data/final/test_100.jsonl")
VAL    = os.path.join(ROOT, "data/final/val.jsonl")
CLEAN  = os.path.join(ROOT, "data/clean_source/tatoeba_eval.txt")
PREFIX = "düzelt: "
MAXLEN = 384
SEED   = 42


def build_test_set():
    if os.path.exists(TEST):
        return
    val = [json.loads(l) for l in open(VAL, encoding="utf-8")]
    noisy = [{"source": r["source"], "target": r["target"]}
             for r in val if r["source"].strip() != r["target"].strip()]
    random.Random(SEED).shuffle(noisy)
    noisy = noisy[:50]
    clean_raw = [l for l in open(CLEAN, encoding="utf-8").read().split("\n") if l.strip()]
    random.Random(SEED).shuffle(clean_raw)
    clean = [{"source": s, "target": s} for s in clean_raw[:50]]
    with open(TEST, "w", encoding="utf-8") as f:
        for r in noisy + clean:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("test_100.jsonl created (50 noisy + 50 clean, seed=42).")


def main():
    build_test_set()
    rows = [json.loads(l) for l in open(TEST, encoding="utf-8")]

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev).eval()

    preds = []
    B = 16
    for i in range(0, len(rows), B):
        batch = [PREFIX + r["source"] for r in rows[i:i+B]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(dev)
        with torch.no_grad():
            out = model.generate(**enc, num_beams=1, max_length=MAXLEN)
        preds += [p.strip() for p in tok.batch_decode(out, skip_special_tokens=True)]

    quality_target, quality_pred, preserve_source, preserve_pred = [], [], [], []
    for r, p in zip(rows, preds):
        s, t = r["source"].strip(), r["target"].strip()
        if s == t:
            preserve_source.append(s); preserve_pred.append(p)
        else:
            quality_target.append(t); quality_pred.append(p)

    m = metrics.selection_scores(quality_target, quality_pred, preserve_source, preserve_pred, a=0.6, b=0.4, w=0.7)
    print("=" * 52)
    print(f"  100-SENTENCE TEST  (noisy={len(quality_target)}, clean={len(preserve_source)})   device={dev}")
    print("=" * 52)
    print(f"  CER                : {m['CER']:.5f}")
    print(f"  WER                : {m['WER']:.5f}")
    print(f"  Q_good  (quality)  : {m['Q_good']:.5f}")
    print(f"  unchanged (preserve): {m['unchanged_rate']:.5f}")
    print(f"  selection_score    : {m['selection_score']:.5f}   <-- number to compare across runs")
    print("=" * 52)


if __name__ == "__main__":
    main()
