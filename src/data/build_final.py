import json, glob, os, random

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LABELED = os.path.join(ROOT, "data/labeled")
FINAL = os.path.join(ROOT, "data/final")
VAL_TARGET = 15000
SEED = 42

EXCLUDE = {"goktugd_diacritics"}

def main():
    os.makedirs(FINAL, exist_ok=True)
    files = [f for f in sorted(glob.glob(os.path.join(LABELED, "*.jsonl")))
             if os.path.basename(f).replace(".jsonl", "") not in EXCLUDE]
    if EXCLUDE:
        print(f"EXCLUDED datasets: {', '.join(sorted(EXCLUDE))}\n")
    total = 0
    counts = {}
    for f in files:
        n = sum(1 for _ in open(f, encoding="utf-8"))
        counts[f] = n; total += n
    val_ratio = VAL_TARGET / max(total, 1)
    print(f"total rows: {total:,}  |  target val ratio: {val_ratio:.4%}\n")

    rnd = random.Random(SEED)
    tf = open(os.path.join(FINAL, "train.jsonl"), "w", encoding="utf-8")
    vf = open(os.path.join(FINAL, "val.jsonl"), "w", encoding="utf-8")
    train_n = val_n = 0
    dataset_counts = {}

    for f in files:
        name = os.path.basename(f).replace(".jsonl", "")
        kt = kv = 0
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if not d.get("source") or not d.get("target"):
                    continue
                record = {
                    "source": d["source"],
                    "target": d["target"],
                    "label": d.get("label", "unknown"),
                    "dataset": name,
                }
                out = json.dumps(record, ensure_ascii=False) + "\n"
                if rnd.random() < val_ratio:
                    vf.write(out); kv += 1; val_n += 1
                else:
                    tf.write(out); kt += 1; train_n += 1
        dataset_counts[name] = (kt, kv)
        print(f"  {name:22s}: train {kt:>9,}  val {kv:>5,}")

    tf.close(); vf.close()
    print(f"\nTOTAL -> train: {train_n:,}  |  val: {val_n:,}")
    print(f"Files: {FINAL}/train.jsonl , {FINAL}/val.jsonl")
    print("Note: every row has a 'dataset' field -> filter by it if needed.")

if __name__ == "__main__":
    main()
