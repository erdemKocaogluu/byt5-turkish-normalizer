import json, sys, time
from pathlib import Path
sys.path.insert(0, "src/eval")
from metrics import evaluate, print_report

def load(split):
    rows=[json.loads(l) for l in Path(f"data/gold/turkishtweets_{split}.jsonl").open(encoding="utf-8")]
    return rows

def main():
    print("Loading Zemberek (may download data on first run, this can take a while)...")
    from zemberek import TurkishMorphology, TurkishSentenceNormalizer
    morphology = TurkishMorphology.create_with_defaults()
    normalizer = TurkishSentenceNormalizer(morphology)
    print("Zemberek ready.\n")

    for split in ["dev", "test"]:
        rows = load(split)
        src = [r["source"] for r in rows]
        tgt = [r["target"] for r in rows]
        t0=time.time()
        pred=[]
        for i,s in enumerate(src):
            try:
                pred.append(normalizer.normalize(s))
            except Exception:
                pred.append(s)
            if (i+1)%200==0:
                print(f"  {split}: {i+1}/{len(src)} processed...")
        dt=time.time()-t0
        m = evaluate(src, tgt, pred)
        print_report(f"ZEMBEREK baseline — {split}", m)
        print(f"  Time: {dt:.1f}s  ({1000*dt/len(src):.1f} ms/sentence)")
        print("  --- example (noisy -> Zemberek -> gold) ---")
        for r,p in list(zip(rows,pred))[:4]:
            print(f"    noisy : {r['source']}")
            print(f"    output: {p}")
            print(f"    gold  : {r['target']}\n")
        Path("reports").mkdir(exist_ok=True)
        with open(f"reports/zemberek_{split}_result.json","w",encoding="utf-8") as f:
            json.dump(m,f,ensure_ascii=False,indent=2)

if __name__ == "__main__":
    main()
