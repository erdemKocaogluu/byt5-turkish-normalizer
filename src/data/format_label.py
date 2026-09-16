import sys, os, json, glob, csv
import pandas as pd
from collections import Counter

csv.field_size_limit(10**9)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "data", "labeled")

SIX = {"ascii", "lowercase", "spacing", "punctuation_drop", "abbreviation", "character_repeat"}


def goktugd_textnorm_label(noise_type: str) -> str:
    return noise_type if noise_type in SIX else "common"


def _emit(sources, targets, label_or_fn, labels_col=None, limit=None, n=0):
    if labels_col is None:
        for s, t in zip(sources, targets):
            yield str(s), str(t), label_or_fn
            n += 1
            if limit and n >= limit:
                return
    else:
        for s, t, lab in zip(sources, targets, labels_col):
            yield str(s), str(t), label_or_fn(str(lab))
            n += 1
            if limit and n >= limit:
                return


def read_oscar(limit=None):
    got = 0
    for split in ["oscar_gec_train.csv", "oscar_gec_val.csv"]:
        path = os.path.join(ROOT, "data/raw", split)
        for chunk in pd.read_csv(path, chunksize=100000):
            for rec in _emit(chunk["sentence"], chunk["correction"], "unknown",
                             limit=None if not limit else limit - got):
                yield rec
                got += 1
                if limit and got >= limit:
                    return


def read_gpt_gec(limit=None):
    got = 0
    for split in ["train.csv", "val.csv"]:
        path = os.path.join(ROOT, "data/raw/external/gpt_gec", split)
        df = pd.read_csv(path)
        for rec in _emit(df["sentence"], df["correction"], "unknown",
                         limit=None if not limit else limit - got):
            yield rec
            got += 1
            if limit and got >= limit:
                return


def read_goktugd_textnorm(limit=None):
    got = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "data/raw/external/goktugd_textnorm/data/train-*.parquet"))):
        df = pd.read_parquet(f, columns=["noisy_text", "normalized_text", "noise_type"])
        for rec in _emit(df["noisy_text"], df["normalized_text"], goktugd_textnorm_label,
                         labels_col=df["noise_type"], limit=None if not limit else limit - got):
            yield rec
            got += 1
            if limit and got >= limit:
                return


def read_goktugd_diacritics(limit=None):
    got = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "data/raw/external/goktugd_diacritics/data/train-*.parquet"))):
        df = pd.read_parquet(f, columns=["ascii_text", "restored_text"])
        for rec in _emit(df["ascii_text"], df["restored_text"], "ascii",
                         limit=None if not limit else limit - got):
            yield rec
            got += 1
            if limit and got >= limit:
                return


def read_noisedwikitr(limit=None):
    got = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "data/raw/external/noisywikitr/nwt_*"))):
        for chunk in pd.read_csv(f, usecols=["cleaned", "corrupted"], chunksize=100000):
            for rec in _emit(chunk["corrupted"], chunk["cleaned"], "common",
                             limit=None if not limit else limit - got):
                yield rec
                got += 1
                if limit and got >= limit:
                    return


def read_radicho(limit=None):
    got = 0
    path = os.path.join(ROOT, "data/raw/external/radicho/train.csv")
    for chunk in pd.read_csv(path, chunksize=100000):
        tr = chunk[chunk["Language"] == "tr"]
        for rec in _emit(tr["Text"], tr["Expected"], "unknown",
                         limit=None if not limit else limit - got):
            yield rec
            got += 1
            if limit and got >= limit:
                return


def read_aytan(limit=None):
    path = os.path.join(ROOT, "data/raw/external/aytan/turkish_spell_validation.txt")
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if "[ORG]" not in line:
                continue
            noisy, clean = line.split("[ORG]", 1)
            yield noisy, clean, "unknown"
            if limit and (limit := limit - 1) == 0:
                return


DATASETS = {
    "oscar":               read_oscar,
    "gpt_gec":             read_gpt_gec,
    "goktugd_textnorm":    read_goktugd_textnorm,
    "goktugd_diacritics":  read_goktugd_diacritics,
    "noisedwikitr":        read_noisedwikitr,
    "radicho":             read_radicho,
    "aytan":               read_aytan,
}


def run_test(n=100):
    print(f"=== TEST MODE: {n} examples from each dataset (nothing written to disk) ===\n")
    for name, reader in DATASETS.items():
        recs = list(reader(limit=n))
        labels = Counter(r[2] for r in recs)
        print(f"### {name}  ({len(recs)} examples)  label distribution: {dict(labels)}")
        for src, tgt, lab in recs[:3]:
            print(f"   [{lab}]")
            print(f"     source: {src[:75]}")
            print(f"     target: {tgt[:75]}")
        print()


def run_full():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== FULL MODE: all datasets -> {OUT_DIR}/<name>.jsonl ===\n")
    grand = Counter()
    for name, reader in DATASETS.items():
        out = os.path.join(OUT_DIR, f"{name}.jsonl")
        labels = Counter()
        n = 0
        with open(out, "w", encoding="utf-8") as w:
            for src, tgt, lab in reader():
                w.write(json.dumps({"source": src, "target": tgt, "label": lab}, ensure_ascii=False) + "\n")
                labels[lab] += 1
                n += 1
        grand.update(labels)
        print(f"### {name}: {n:,} rows -> {out}")
        print(f"     label distribution: {dict(labels)}")
    print(f"\n=== TOTAL label distribution: {dict(grand)} ===")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "test"
    if mode == "test":
        run_test(int(sys.argv[2]) if len(sys.argv) > 2 else 100)
    elif mode == "full":
        run_full()
    else:
        print("usage: python src/data/format_label.py [test|full]")
