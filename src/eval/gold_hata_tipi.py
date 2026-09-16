import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_model as EM

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "models/byt5-small-tr-normalizer")
BEAM = 1
G_ORDER = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]


def per_g(name, force_g=None):
    src, tgt = EM.GOLD_LOADERS[name]()
    preds = EM.model_predict(MODEL_DIR, src, BEAM)
    if name in EM.NORMALIZE_GOLDS:
        src = [EM.normalize(s) for s in src]
        tgt = [EM.normalize(t) for t in tgt]
        preds = [EM.normalize(p) for p in preds]
    needed, fixed = EM.per_g_fixrate(src, tgt, preds, force_g=force_g)
    return needed, fixed


def main():
    print("Loading model + gold set (boun)...")
    n_boun, f_boun = per_g("boun", force_g="G7")
    print("  boun done")

    def table(title, needed, fixed):
        lines = [f"### {title}\n",
                 "| G | type | needed | fixed | success% | miss% |",
                 "|---|---|---|---|---|---|"]
        for g in G_ORDER:
            ne = needed.get(g, 0)
            if ne == 0:
                continue
            fx = fixed.get(g, 0)
            success = 100 * fx / ne
            lines.append(f"| {g} | {EM.G_NAME[g]} | {ne} | {fx} | %{success:.0f} | %{100-success:.0f} |")
        return "\n".join(lines) + "\n"

    md = os.path.join(ROOT, "reports", "gold_error_type.md")
    os.makedirs(os.path.dirname(md), exist_ok=True)
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Gold — Model Success/Miss Rate by Error Type (G)\n\n")
        f.write(f"Model: byt5-small-tr-normalizer · beam={BEAM}\n\n")
        f.write("> **needed** = number of errors of that type that required correction · **fixed** = number the model corrected correctly · "
                "**success%** = fixed/needed · **miss%** = the model's miss rate for that type (100 - success)\n\n")
        f.write("## boun (de/da benchmark -> G7)\n\n")
        f.write(table("boun (de/da benchmark -> G7)", n_boun, f_boun))

    groups = [g for g in G_ORDER if n_boun.get(g, 0) > 0]
    success = [100 * f_boun[g] / n_boun[g] for g in groups]
    miss = [100 - b for b in success]
    labels = [f"{g}\n{EM.G_NAME[g]}" for g in groups]
    import numpy as np
    x = np.arange(len(groups)); w = 0.4
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - w/2, success, w, label="success%", color="#059669")
    ax.bar(x + w/2, miss, w, label="miss% (missed)", color="#dc2626")
    for i, (b, h) in enumerate(zip(success, miss)):
        ax.text(x[i]-w/2, b, f"%{b:.0f}", ha="center", va="bottom", fontsize=9)
        ax.text(x[i]+w/2, h, f"%{h:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Gold — Model Success / Miss Rate by Error Type (boun)", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("%"); ax.set_ylim(0, 105); ax.legend()
    ax.grid(axis="y", color="#e5e7eb")
    png = os.path.join(ROOT, "reports", "gold_error_type.png")
    fig.tight_layout(); fig.savefig(png, bbox_inches="tight"); plt.close(fig)

    print(f"\nDONE.\n  {md}\n  {png}")


if __name__ == "__main__":
    main()
