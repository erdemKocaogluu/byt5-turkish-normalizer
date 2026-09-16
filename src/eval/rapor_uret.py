import argparse, os, sys, json, csv, html

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLOR = {"sel": "#2563eb", "quality": "#059669", "preservation": "#d97706",
        "train": "#dc2626", "eval": "#7c3aed", "cer": "#0891b2", "wer": "#be123c",
        "lr": "#6b7280", "grid": "#e5e7eb", "good": "#f59e0b"}
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.color": COLOR["grid"],
                     "axes.edgecolor": "#9ca3af", "figure.dpi": 140})


def fnum(x):
    try:
        if x is None or str(x).strip() == "":
            return None
        return float(x)
    except (ValueError, TypeError):
        return None


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def column(rows, name):
    xs, ys = [], []
    for s in rows:
        a, v = fnum(s.get("step")), fnum(s.get(name))
        if a is not None and v is not None:
            xs.append(a); ys.append(v)
    return xs, ys


def epoch_summary(scores):
    import math
    bucket = {}
    for s in scores:
        ep = fnum(s.get("epoch"))
        if ep is None:
            continue
        b = max(1, math.ceil(ep))
        if b not in bucket or ep > bucket[b]["ep"]:
            bucket[b] = {"ep": ep, "tl": fnum(s.get("train_loss")), "vl": fnum(s.get("eval_loss"))}
    epochs = sorted(bucket)
    tl = [bucket[e]["tl"] for e in epochs]
    vl = [bucket[e]["vl"] for e in epochs]
    valid = [(e, bucket[e]["vl"]) for e in epochs if bucket[e]["vl"] is not None]
    best_epoch = min(valid, key=lambda x: x[1])[0] if valid else None
    return epochs, tl, vl, best_epoch


def _save(fig, out, name):
    path = os.path.join(out, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}")


def plot_learning_curve(scores, out):
    xs, ys = column(scores, "combined_selection_score")
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, "-o", color=COLOR["sel"], lw=2, ms=5, label="selection_score")
    best_i = max(range(len(ys)), key=lambda i: ys[i])
    ax.plot(xs[best_i], ys[best_i], "*", color=COLOR["good"], ms=22, zorder=5,
            label=f"best: step {int(xs[best_i])} = {ys[best_i]:.4f}")
    ax.annotate(f"{ys[best_i]:.4f}", (xs[best_i], ys[best_i]),
                textcoords="offset points", xytext=(0, 14), ha="center", fontweight="bold")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.08)
    ax.set_title("Learning Curve — Combined Selection Score (0.7·quality + 0.3·preservation)", fontweight="bold")
    ax.set_xlabel("Training step (checkpoint)"); ax.set_ylabel("selection_score (higher = better)")
    ax.legend()
    _save(fig, out, "01_learning_curve.png")


def plot_quality_preservation(scores, out):
    xk, yk = column(scores, "quality_Q_good")
    xr, yr = column(scores, "preservation_unchanged_rate")
    if not xk and not xr:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    if xk:
        ax.plot(xk, yk, "-o", color=COLOR["quality"], lw=2, ms=5, label="QUALITY (Q_good) — fixing noisy text")
    if xr:
        ax.plot(xr, yr, "-s", color=COLOR["preservation"], lw=2, ms=5, label="PRESERVATION (unchanged) — leaving clean text alone")
    ax.set_title("Quality vs. Preservation — The Core Trade-off", fontweight="bold")
    ax.set_xlabel("Training step (checkpoint)"); ax.set_ylabel("rate (higher = better)")
    ax.set_ylim(0, 1.02); ax.legend()
    _save(fig, out, "02_quality_vs_preservation.png")


def plot_loss(scores, out):
    xt, yt = column(scores, "train_loss")
    xe, ye = column(scores, "eval_loss")
    if not xt and not xe:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    if xt:
        ax.plot(xt, yt, "-o", color=COLOR["train"], lw=2, ms=4, label="train_loss")
    if xe:
        ax.plot(xe, ye, "-s", color=COLOR["eval"], lw=2, ms=4, label="eval_loss")
    ax.set_title("Loss — Learning or Memorizing?", fontweight="bold")
    ax.set_xlabel("Training step"); ax.set_ylabel("loss (lower = better)")
    ax.legend()
    _save(fig, out, "03_loss.png")


def plot_epoch_loss(scores, out):
    epochs, tl, vl, best_epoch = epoch_summary(scores)
    if not epochs:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    if any(v is not None for v in tl):
        ax.plot(epochs, tl, "-o", color=COLOR["train"], lw=2, ms=6, label="train_loss")
    if any(v is not None for v in vl):
        ax.plot(epochs, vl, "-s", color=COLOR["eval"], lw=2, ms=6, label="val_loss")
    if best_epoch is not None:
        vl_best = vl[epochs.index(best_epoch)]
        ax.axvline(best_epoch, color=COLOR["good"], ls="--", lw=1.5)
        ax.plot(best_epoch, vl_best, "*", color=COLOR["good"], ms=22, zorder=5,
                label=f"BEST epoch: {best_epoch} (val_loss={vl_best:.4f})")
    ax.set_title("Train vs. Validation Loss by Epoch — Best Epoch", fontweight="bold")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss (lower = better)")
    ax.set_xticks(epochs); ax.legend()
    _save(fig, out, "09_epoch_loss.png")


def plot_cer_wer(scores, out):
    xc, yc = column(scores, "CER")
    xw, yw = column(scores, "WER")
    if not xc and not xw:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    if xc:
        ax.plot(xc, yc, "-o", color=COLOR["cer"], lw=2, ms=4, label="CER (character error rate)")
    if xw:
        ax.plot(xw, yw, "-s", color=COLOR["wer"], lw=2, ms=4, label="WER (word error rate)")
    ax.set_title("Error Rates on Noisy Sentences — Lower Is Better", fontweight="bold")
    ax.set_xlabel("Training step"); ax.set_ylabel("error rate")
    ax.legend()
    _save(fig, out, "04_cer_wer.png")


def plot_lr(scores, out):
    xs, ys = column(scores, "learning_rate")
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, "-o", color=COLOR["lr"], lw=2, ms=4)
    ax.set_title("Learning Rate Schedule — Warmup + Decay", fontweight="bold")
    ax.set_xlabel("Training step"); ax.set_ylabel("learning_rate")
    _save(fig, out, "05_learning_rate.png")


def plot_gold(gold, out):
    if not gold or not gold.get("sets"):
        return
    sets = gold["sets"]
    names = list(sets.keys())
    import numpy as np
    x = np.arange(len(names))
    width = 0.36

    def value(set_name, which, metric):
        d = sets[set_name].get(which, {})
        return d.get(metric) if isinstance(d, dict) else None

    mw = [value(a, "model", "WER") for a in names]
    cw = [value(a, "copy", "WER") for a in names]
    if any(v is not None for v in mw):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width/2, [v or 0 for v in cw], width, label="copy (no correction)", color="#cbd5e1")
        ax.bar(x + width/2, [v or 0 for v in mw], width, label="MODEL", color=COLOR["sel"])
        for i, v in enumerate(mw):
            if v is not None:
                ax.text(x[i] + width/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title("Gold Evaluation — WER (lower = better): Model vs. Copy Baseline", fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("WER"); ax.legend()
        _save(fig, out, "07_gold_wer.png")

    me = [value(a, "model", "exact_match") for a in names]
    ce = [value(a, "copy", "exact_match") for a in names]
    if any(v is not None for v in me):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width/2, [v or 0 for v in ce], width, label="copy (no correction)", color="#cbd5e1")
        ax.bar(x + width/2, [v or 0 for v in me], width, label="MODEL", color=COLOR["quality"])
        for i, v in enumerate(me):
            if v is not None:
                ax.text(x[i] + width/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title("Gold Evaluation — Exact Sentence Match (higher = better)", fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("exact_match"); ax.legend()
        _save(fig, out, "08_gold_exact.png")


def md_table(headers, rows):
    top = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(h) for h in r) + " |" for r in rows)
    return "\n".join([top, sep, body])


def write_results_md(scores, datasets, examples, summary, gold, out):
    L = ["# Training Report — byT5-small Turkish Normalizer\n"]

    if summary:
        hp = summary.get("hyperparameters", {})
        data = summary.get("data", {})
        L.append("## Summary\n")
        L.append(md_table(["field", "value"], [
            ["Model", summary.get("model", "-")],
            ["Device", summary.get("device", "-")],
            ["Date", summary.get("date", "-")],
            ["Training sentences", f"{data.get('train','-'):,}" if isinstance(data.get('train'), int) else data.get('train','-')],
            ["Eval noisy (quality)", data.get("eval_noisy_quality", "-")],
            ["Eval clean (preservation)", data.get("eval_clean_preservation", "-")],
            ["Batch", hp.get("batch", "-")],
            ["Epoch", hp.get("epoch", "-")],
            ["Learning rate", hp.get("learning_rate", "-")],
            ["Seed", hp.get("seed", "-")],
            ["Selection weight w (quality)", hp.get("w_quality", "-")],
            ["Total steps", summary.get("total_steps", "-")],
            ["Total duration (min)", summary.get("total_minutes", "-")],
        ]))
        L.append("")

    L.append("## Charts\n")
    for name, title in [("09_epoch_loss.png", "Train vs. val loss by epoch (best epoch)"),
                    ("01_learning_curve.png", "Learning curve (selection score)"),
                    ("02_quality_vs_preservation.png", "Quality vs. preservation trade-off"),
                    ("03_loss.png", "Loss (per step, train vs. eval)"),
                    ("04_cer_wer.png", "CER & WER"),
                    ("05_learning_rate.png", "Learning rate schedule"),
                    ("07_gold_wer.png", "Gold evaluation — WER (model vs. copy)"),
                    ("08_gold_exact.png", "Gold evaluation — exact match (model vs. copy)")]:
        if os.path.exists(os.path.join(out, name)):
            L.append(f"### {title}\n\n![{title}]({name})\n")

    if summary and summary.get("top_3"):
        L.append("## Top 3 Checkpoints\n")
        L.append(md_table(["rank", "step", "selection_score"],
                          [[c["rank"], c["step"], c["selection_score"]] for c in summary["top_3"]]))
        L.append("")

    if scores:
        if summary and summary.get("top_3"):
            top_steps = {str(c["step"]) for c in summary["top_3"]}
        else:
            ranked = sorted([s for s in scores if fnum(s.get("combined_selection_score")) is not None],
                            key=lambda s: fnum(s["combined_selection_score"]), reverse=True)[:3]
            top_steps = {s["step"] for s in ranked}
        L.append("## All Checkpoints\n")
        L.append("> ★ = top 3 checkpoints by combined selection score\n")
        cols = ["step", "epoch", "examples_seen", "train_loss", "eval_loss", "CER", "WER",
                "quality_Q_good", "preservation_unchanged_rate", "combined_selection_score"]
        headers = ["★", "step", "epoch", "examples", "train_loss", "eval_loss", "CER", "WER",
                "Q_good", "unchanged", "selection"]
        rows = []
        for s in scores:
            mark = "★" if str(s.get("step")) in top_steps else ""
            rows.append([mark] + [s.get(k, "") for k in cols])
        L.append(md_table(headers, rows))
        L.append("")

    if gold and gold.get("sets"):
        L.append("## Gold Evaluation (naturally occurring human text)\n")
        L.append(f"Beam: {gold.get('beam','-')}  ·  Date: {gold.get('date','-')}\n")
        headers = ["set", "n", "WER (model)", "WER (copy)", "exact match", "norm_F1", "over-correction"]
        rows = []
        for name, d in gold["sets"].items():
            mo = d.get("model", {}); co = d.get("copy", {})
            rows.append([name, d.get("n", "-"),
                          mo.get("WER", "-"), co.get("WER", "-"),
                          mo.get("exact_match", "-"), mo.get("norm_F1", "-"),
                          mo.get("overcorrection", "-")])
        L.append(md_table(headers, rows))
        L.append("\n> WER lower = better · exact match/F1 higher = better · over-correction lower = better (damaging clean text)\n")

    path = os.path.join(out, "RESULTS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("  ✓ RESULTS.md")


def write_dashboard_html(scores, datasets, examples, summary, gold, out):
    def series(name):
        xs, ys = column(scores, name)
        return xs, ys

    def trace(name, label, color):
        xs, ys = series(name)
        return {"x": xs, "y": ys, "name": label, "color": color}

    data = {
        "sel": trace("combined_selection_score", "selection_score", COLOR["sel"]),
        "quality": trace("quality_Q_good", "quality (Q_good)", COLOR["quality"]),
        "preservation": trace("preservation_unchanged_rate", "preservation (unchanged)", COLOR["preservation"]),
        "train": trace("train_loss", "train_loss", COLOR["train"]),
        "eval": trace("eval_loss", "eval_loss", COLOR["eval"]),
        "cer": trace("CER", "CER", COLOR["cer"]),
        "wer": trace("WER", "WER", COLOR["wer"]),
        "lr": trace("learning_rate", "learning_rate", COLOR["lr"]),
    }
    info = summary or {}
    epochs, e_tl, e_vl, e_best = epoch_summary(scores)
    data_json = json.dumps({"data": data,
                            "summary": info,
                            "gold": (gold or {}).get("sets", {}),
                            "epoch": {"x": epochs, "train": e_tl, "val": e_vl, "best": e_best}},
                           ensure_ascii=False)

    html_str = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training Dashboard — byT5 Turkish Normalizer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;background:#f8fafc;color:#0f172a}
 header{background:#1e293b;color:#fff;padding:22px 28px}
 header h1{margin:0;font-size:20px} header p{margin:6px 0 0;color:#cbd5e1;font-size:13px}
 .cards{display:flex;flex-wrap:wrap;gap:14px;padding:20px 28px}
 .card{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:14px 18px;min-width:150px;box-shadow:0 1px 2px rgba(0,0,0,.04)}
 .card .b{font-size:12px;color:#64748b} .card .d{font-size:22px;font-weight:700;margin-top:4px}
 .charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px;padding:0 28px 28px}
 .g{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:8px}
 table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}
 th,td{border:1px solid #e2e8f0;padding:6px 9px;text-align:left} th{background:#f1f5f9}
 h2{padding:0 28px;margin:8px 0} .section{padding:0 28px 24px}
</style></head><body>
<header><h1>Training Dashboard — byT5-small Turkish Normalizer</h1>
<p id="subtitle"></p></header>
<div class="cards" id="cards"></div>
<h2>Charts</h2>
<div class="charts">
 <div class="g" id="g_epoch"></div><div class="g" id="g_sel"></div>
 <div class="g" id="g_kk"></div><div class="g" id="g_loss"></div>
 <div class="g" id="g_cerwer"></div><div class="g" id="g_lr"></div>
 <div class="g" id="g_gold"></div>
</div>
<h2>Gold Evaluation (naturally occurring human text)</h2><div class="section"><div id="gold_table"></div></div>
<h2>All Checkpoints</h2><div class="section"><div id="cp_table"></div></div>
<script>
const D = __DATA__;
const cfg = {responsive:true, displModeBar:false};
const L = (t,yb)=>({title:{text:t,font:{size:14}}, margin:{t:40,r:15,b:45,l:55},
   xaxis:{title:'training step'}, yaxis:{title:yb}, height:330, legend:{orientation:'h',y:-0.25}});
function ln(o){return {x:o.x,y:o.y,name:o.name,mode:'lines+markers',line:{color:o.color,width:2},marker:{size:5}};}

const info=D.summary||{}, hp=info.hyperparameters||{}, data=info.data||{};
document.getElementById('subtitle').textContent =
  (info.model||'')+'  ·  '+(info.device||'')+'  ·  '+(info.date||'');
const best=(info.top_3&&info.top_3[0])||{};
const cards=[['Best selection_score',(best.selection_score??'-')],
  ['Best step',(best.step??'-')],['Total steps',(info.total_steps??'-')],
  ['Training sentences',(data.train??'-')],['Batch',(hp.batch??'-')],
  ['Epoch',(hp.epoch??'-')],['Duration (min)',(info.total_minutes??'-')]];
document.getElementById('cards').innerHTML =
  cards.map(k=>`<div class="card"><div class="b">${k[0]}</div><div class="d">${k[1]}</div></div>`).join('');

const ep=D.epoch||{};
if(ep.x&&ep.x.length){
 const tr=[{x:ep.x,y:ep.train,name:'train_loss',mode:'lines+markers',line:{color:'#dc2626',width:2}},
           {x:ep.x,y:ep.val,name:'val_loss',mode:'lines+markers',line:{color:'#7c3aed',width:2}}];
 const lay=Object.assign(L('Train vs. val loss by epoch','loss'),{xaxis:{title:'epoch',dtick:1}});
 if(ep.best!=null){lay.shapes=[{type:'line',x0:ep.best,x1:ep.best,yref:'paper',y0:0,y1:1,
   line:{color:'#f59e0b',width:2,dash:'dash'}}];
   lay.annotations=[{x:ep.best,yref:'paper',y:1,text:'best: '+ep.best,showarrow:false,font:{color:'#b45309'}}];}
 Plotly.newPlot('g_epoch',tr,lay,cfg);
}else{document.getElementById('g_epoch').style.display='none';}

Plotly.newPlot('g_sel',[ln(D.data.sel)],L('Learning curve — selection_score','score'),cfg);
Plotly.newPlot('g_kk',[ln(D.data.quality),ln(D.data.preservation)],L('Quality vs. Preservation','rate'),cfg);
Plotly.newPlot('g_loss',[ln(D.data.train),ln(D.data.eval)],L('Loss (train vs. eval)','loss'),cfg);
Plotly.newPlot('g_cerwer',[ln(D.data.cer),ln(D.data.wer)],L('CER & WER','error rate'),cfg);
Plotly.newPlot('g_lr',[ln(D.data.lr)],L('Learning rate schedule','lr'),cfg);

const gold=D.gold||{}, goldSets=Object.keys(gold);
if(goldSets.length){
 const mw=goldSets.map(a=>(gold[a].model||{}).WER), cw=goldSets.map(a=>(gold[a].copy||{}).WER);
 Plotly.newPlot('g_gold',[{x:goldSets,y:cw,type:'bar',name:'copy',marker:{color:'#cbd5e1'}},
   {x:goldSets,y:mw,type:'bar',name:'MODEL',marker:{color:'#2563eb'}}],
   Object.assign(L('Gold WER — model vs. copy','WER'),{barmode:'group',xaxis:{title:'gold set'}}),cfg);
 let gh='<table><tr><th>set</th><th>n</th><th>WER model</th><th>WER copy</th><th>exact match</th><th>norm_F1</th><th>over-correction</th></tr>';
 goldSets.forEach(a=>{const mo=gold[a].model||{},co=gold[a].copy||{};
  gh+=`<tr><td>${a}</td><td>${gold[a].n||''}</td><td>${mo.WER??''}</td><td>${co.WER??''}</td><td>${mo.exact_match??''}</td><td>${mo.norm_F1??''}</td><td>${mo.overcorrection??''}</td></tr>`;});
 document.getElementById('gold_table').innerHTML=gh+'</table>';
}else{
 document.getElementById('g_gold').style.display='none';
 document.getElementById('gold_table').innerHTML='<p style="color:#64748b">No gold evaluation results yet.</p>';
}

const sx=D.data.sel.x;
let ht='<table><tr><th>step</th><th>selection</th><th>Q_good</th><th>unchanged</th><th>CER</th><th>WER</th><th>train_loss</th><th>eval_loss</th></tr>';
sx.forEach((a,i)=>{const g=k=>{const s=D.data[k]; const j=s.x.indexOf(a); return j<0?'':(s.y[j]);};
 ht+=`<tr><td>${a}</td><td>${g('sel')}</td><td>${g('quality')}</td><td>${g('preservation')}</td><td>${g('cer')}</td><td>${g('wer')}</td><td>${g('train')}</td><td>${g('eval')}</td></tr>`;});
document.getElementById('cp_table').innerHTML=ht+'</table>';
</script></body></html>"""
    html_str = html_str.replace("__DATA__", data_json)
    path = os.path.join(out, "dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_str)
    print("  ✓ dashboard.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=os.path.join(ROOT, "models/byt5-small-tr-normalizer"))
    ap.add_argument("--out", default=os.path.join(ROOT, "reports/training"))
    a = ap.parse_args()

    scores = read_csv(os.path.join(a.model_dir, "score_report.csv"))
    datasets = read_csv(os.path.join(a.model_dir, "dataset_report.csv"))
    examples = []
    ep = os.path.join(a.model_dir, "examples.jsonl")
    if os.path.exists(ep):
        examples = [json.loads(l) for l in open(ep, encoding="utf-8") if l.strip()]
    summary = None
    sp = os.path.join(a.model_dir, "summary.json")
    if os.path.exists(sp):
        summary = json.load(open(sp, encoding="utf-8"))
    elif os.path.exists(os.path.join(a.model_dir, "training_config.json")):
        summary = json.load(open(os.path.join(a.model_dir, "training_config.json"), encoding="utf-8"))
    gold = None
    gp = os.path.join(a.model_dir, "gold_results.json")
    if os.path.exists(gp):
        gold = json.load(open(gp, encoding="utf-8"))

    if not scores:
        print(f"WARNING: {a.model_dir}/score_report.csv is missing or empty. Run training first.")
        sys.exit(1)

    os.makedirs(a.out, exist_ok=True)
    print(f"Generating report -> {a.out}  ({len(scores)} checkpoints)")
    plot_learning_curve(scores, a.out)
    plot_quality_preservation(scores, a.out)
    plot_loss(scores, a.out)
    plot_cer_wer(scores, a.out)
    plot_lr(scores, a.out)
    plot_epoch_loss(scores, a.out)
    plot_gold(gold, a.out)
    write_results_md(scores, datasets, examples, summary, gold, a.out)
    write_dashboard_html(scores, datasets, examples, summary, gold, a.out)
    print("DONE.")


if __name__ == "__main__":
    main()
