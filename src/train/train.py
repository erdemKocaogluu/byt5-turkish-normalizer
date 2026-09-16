import argparse, os, sys, random, time, json, shutil, functools
import numpy as np
import torch
from datasets import load_dataset, Dataset

_orig_torch_load = torch.load
def _safe_torch_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)
torch.load = _safe_torch_load
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval"))
import metrics


MODEL_NAME = "google/byt5-small"

MAX_LENGTH = 384
PREFIX = "düzelt: "
OUTPUT_DIR = "models/byt5-small-tr-normalizer"
CER_W, WER_W = 0.6, 0.4
CLEAN_EVAL = "data/clean_source/tatoeba_eval.txt"
N_EXAMPLES = 5


class CheckpointTracker(TrainerCallback):
    HEADER = ("step,epoch,examples_seen,elapsed_min,train_loss,eval_loss,learning_rate,"
              "CER,WER,quality_Q_good,preservation_unchanged_rate,combined_selection_score\n")
    DATASET_HEADER = "step,dataset,n,CER,WER\n"

    def __init__(self, csv_path, dataset_csv_path, examples_path, ckpt_dir, k=3, resume=False):
        self.csv_path = csv_path
        self.dataset_csv_path = dataset_csv_path
        self.examples_path = examples_path
        self.ckpt_dir = ckpt_dir
        self.k = k
        self.top = []
        self.start_time = None
        self.pending_examples = None
        self.pending_dataset_breakdown = None
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if resume and os.path.exists(csv_path):
            os.makedirs(ckpt_dir, exist_ok=True)
            self._load_resume_state()
            print(f"resume: keeping existing reports, restored top-{self.k}: "
                  f"{[c['step'] for c in self.top]}")
        else:
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            os.makedirs(ckpt_dir, exist_ok=True)
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(self.HEADER)
            with open(dataset_csv_path, "w", encoding="utf-8") as f:
                f.write(self.DATASET_HEADER)
            open(examples_path, "w", encoding="utf-8").close()

    def _load_resume_state(self):
        import csv as _csv
        candidates = []
        try:
            for r in _csv.DictReader(open(self.csv_path, encoding="utf-8")):
                score, step = r.get("combined_selection_score"), r.get("step")
                if not score or not step:
                    continue
                path = os.path.join(self.ckpt_dir, f"step-{step}")
                if os.path.isdir(path):
                    try:
                        candidates.append({"score": float(score), "step": int(step), "path": path})
                    except ValueError:
                        continue
        except Exception:
            return
        candidates.sort(key=lambda x: x["score"], reverse=True)
        self.top = candidates[:self.k]

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def _last_train_log(self, state):
        for record in reversed(state.log_history):
            if "loss" in record:
                return record.get("loss", ""), record.get("learning_rate", "")
        return "", ""

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if not metrics:
            return
        step = state.global_step
        epoch = round(state.epoch, 4) if state.epoch is not None else ""
        examples_seen = step * getattr(args, "per_device_train_batch_size", 0)
        elapsed_min = round((time.time() - self.start_time) / 60, 2) if self.start_time else ""
        train_loss, lr = self._last_train_log(state)
        eval_loss = metrics.get("eval_loss", "")
        score = metrics.get("eval_selection_score")
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(f"{step},{epoch},{examples_seen},{elapsed_min},{train_loss},{eval_loss},{lr},"
                    f"{metrics.get('eval_CER','')},{metrics.get('eval_WER','')},"
                    f"{metrics.get('eval_Q_good','')},{metrics.get('eval_unchanged_rate','')},{score}\n")
        if self.pending_dataset_breakdown:
            with open(self.dataset_csv_path, "a", encoding="utf-8") as f:
                for name, n, c, w in self.pending_dataset_breakdown:
                    f.write(f"{step},{name},{n},{c},{w}\n")
        if self.pending_examples:
            with open(self.examples_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "examples": self.pending_examples}, ensure_ascii=False) + "\n")
        if model is None or score is None:
            return
        if len(self.top) < self.k or score > self.top[-1]["score"]:
            path = os.path.join(self.ckpt_dir, f"step-{step}")
            model.save_pretrained(path)
            self.top.append({"score": score, "step": step, "path": path})
            self.top.sort(key=lambda x: x["score"], reverse=True)
            for dropped in self.top[self.k:]:
                shutil.rmtree(dropped["path"], ignore_errors=True)
            self.top = self.top[:self.k]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke_test", action="store_true", help="quick smoke test: 100 sentences, ~20 steps")
    p.add_argument("--subset", type=int, default=0, help="cap training to the first N sentences (0=all)")
    p.add_argument("--epoch", type=float, default=1.0, help="number of passes over the training data")
    p.add_argument("--max_steps", type=int, default=0, help="if >0, train this many steps instead of using epoch")
    p.add_argument("--batch", type=int, default=16, help="batch size")
    p.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    p.add_argument("--warmup", type=int, default=500, help="learning-rate warmup steps")
    p.add_argument("--logging_steps", type=int, default=50, help="how often to log the training loss")
    p.add_argument("--seed", type=int, default=42, help="random seed for reproducibility")
    p.add_argument("--resume", action="store_true", help="resume from the last full checkpoint in OUTPUT_DIR; keeps existing reports")
    p.add_argument("--w", type=float, default=0.7, help="quality weight in the selection score (preservation weight = 1-w)")
    p.add_argument("--eval_steps", type=int, default=2000, help="evaluate (and save) every N steps")
    p.add_argument("--eval_noisy", type=int, default=3000, help="number of noisy validation sentences for the quality metric")
    p.add_argument("--clean_eval", type=int, default=3000, help="number of clean sentences for the preservation metric")
    p.add_argument("--num_proc", type=int, default=8, help="CPU processes for tokenization")
    p.add_argument("--base_model", default=MODEL_NAME,
                   help="base model: an HF model id, or a local folder path on an offline machine")
    p.add_argument("--skip_gold", action="store_true", help="skip the post-training gold evaluation")
    p.add_argument("--gold_beam", type=int, default=1, help="beam size for the gold evaluation")
    return p.parse_args()


def select_device():
    if torch.cuda.is_available():
        print("Device: CUDA (GPU)"); return "cuda"
    if torch.backends.mps.is_available():
        print("Device: MPS (Mac GPU)"); return "mps"
    print("Device: CPU (this will be slow)"); return "cpu"


def main():
    args = parse_args()
    device = select_device()

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_bf16 = (device == "cuda" and torch.cuda.is_bf16_supported())
    use_fp16 = (device == "cuda" and not use_bf16)
    print(f"Model: {MODEL_NAME} | max_length={MAX_LENGTH} | batch={args.batch} | "
          f"bf16={use_bf16} fp16={use_fp16} | output={OUTPUT_DIR}")

    print("Loading data...")
    train_ds = load_dataset("json", data_files={"train": "data/final/train.jsonl"})["train"]
    if args.smoke_test:
        train_ds = train_ds.select(range(100))
    elif args.subset > 0:
        train_ds = train_ds.select(range(min(args.subset, len(train_ds))))

    k_noisy = 8 if args.smoke_test else args.eval_noisy
    k_clean = 8 if args.smoke_test else args.clean_eval

    val_raw = load_dataset("json", data_files={"val": "data/final/val.jsonl"})["val"]
    noisy_rows = [{"source": r["source"], "target": r["target"], "dataset": r.get("dataset", "?")}
                  for r in val_raw if r["source"].strip() != r["target"].strip()]
    random.Random(args.seed).shuffle(noisy_rows)
    noisy_rows = noisy_rows[:k_noisy]
    clean_sents = [l for l in open(CLEAN_EVAL, encoding="utf-8").read().split("\n") if l.strip()][:k_clean]

    clean_rows = [{"source": s, "target": s} for s in clean_sents]
    dataset_of = {r["source"].strip(): r["dataset"] for r in noisy_rows}
    eval_rows = [{"source": r["source"], "target": r["target"]} for r in noisy_rows] + clean_rows
    eval_ds = Dataset.from_list(eval_rows)
    print(f"  train: {len(train_ds)} | eval: {len(noisy_rows)} noisy (quality) + {len(clean_rows)} clean (preservation)")

    max_steps = args.max_steps if args.max_steps > 0 else (20 if args.smoke_test else -1)
    eval_steps = 10 if args.smoke_test else args.eval_steps
    gen_max = 64 if args.smoke_test else MAX_LENGTH

    print(f"Loading model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    def preprocess(examples):
        input_texts = [PREFIX + s for s in examples["source"]]
        batch_enc = tokenizer(input_texts, max_length=MAX_LENGTH, truncation=True)
        target_enc = tokenizer(text_target=examples["target"], max_length=MAX_LENGTH, truncation=True)
        batch_enc["labels"] = target_enc["input_ids"]
        return batch_enc

    tok_proc = 1 if args.smoke_test else args.num_proc
    print(f"Tokenizing... ({tok_proc} process(es))")
    train_tok = train_ds.map(preprocess, batched=True,
                             remove_columns=train_ds.column_names,
                             num_proc=(None if tok_proc <= 1 else tok_proc))
    eval_tok = eval_ds.map(preprocess, batched=True,
                            remove_columns=eval_ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    from transformers.trainer_utils import get_last_checkpoint
    last_checkpoint = None
    if args.resume:
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
        if last_checkpoint:
            print(f"resume: found full checkpoint -> {last_checkpoint}")
        else:
            print("resume requested but no full checkpoint found -> starting from scratch")
    resuming = args.resume and last_checkpoint is not None

    tracker = CheckpointTracker(os.path.join(OUTPUT_DIR, "score_report.csv"),
                                os.path.join(OUTPUT_DIR, "dataset_report.csv"),
                                os.path.join(OUTPUT_DIR, "examples.jsonl"),
                                os.path.join(OUTPUT_DIR, "top3_checkpoints"),
                                resume=resuming)

    def compute_metrics(eval_pred):
        preds = eval_pred.predictions
        labels = eval_pred.label_ids
        inputs = eval_pred.inputs
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        pred_txt = [t.strip() for t in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        tgt_txt = [t.strip() for t in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        src_raw = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        src_txt = [(s[len(PREFIX):] if s.startswith(PREFIX) else s).strip() for s in src_raw]

        quality_target, quality_pred, preservation_source, preservation_pred = [], [], [], []
        for s, t, p in zip(src_txt, tgt_txt, pred_txt):
            if s == t:
                preservation_source.append(s); preservation_pred.append(p)
            else:
                quality_target.append(t); quality_pred.append(p)

        m = metrics.selection_scores(quality_target, quality_pred, preservation_source, preservation_pred,
                                   a=CER_W, b=WER_W, w=args.w)

        examples, n_noisy, n_clean = [], 0, 0
        for s, t, p in zip(src_txt, tgt_txt, pred_txt):
            if s != t and n_noisy < N_EXAMPLES:
                examples.append({"type": "noisy", "input": s, "output": p, "expected": t}); n_noisy += 1
            elif s == t and n_clean < N_EXAMPLES:
                examples.append({"type": "clean", "input": s, "output": p, "expected": t}); n_clean += 1
        tracker.pending_examples = examples

        groups = {}
        for s, t, p in zip(src_txt, tgt_txt, pred_txt):
            if s == t:
                continue
            k = dataset_of.get(s, "?")
            groups.setdefault(k, ([], []))
            groups[k][0].append(t); groups[k][1].append(p)
        dataset_breakdown = []
        for k, (tt, pp) in groups.items():
            mm = metrics.selection_scores(tt, pp, [], [], a=CER_W, b=WER_W, w=args.w)
            dataset_breakdown.append((k, len(tt), round(mm["CER"], 4), round(mm["WER"], 4)))
        tracker.pending_dataset_breakdown = dataset_breakdown

        return {k: round(float(v), 4) for k, v in m.items()}

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epoch,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        group_by_length=True,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        eval_strategy="steps", eval_steps=eval_steps,
        save_strategy="steps", save_steps=eval_steps, save_total_limit=1,
        predict_with_generate=True,
        generation_max_length=gen_max,
        generation_num_beams=1,
        include_inputs_for_metrics=True,
        bf16=use_bf16, fp16=use_fp16,
        dataloader_num_workers=(4 if device == "cuda" else 0),
        report_to="none",
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=collator,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[tracker],
    )
    with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_NAME, "max_length": MAX_LENGTH, "device": device,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "hyperparameters": {
                "epoch": args.epoch, "max_steps": args.max_steps, "batch": args.batch,
                "learning_rate": args.lr, "warmup": args.warmup, "weight_decay": 0.01,
                "seed": args.seed, "w_quality": args.w, "CER_W": CER_W, "WER_W": WER_W,
                "eval_steps": args.eval_steps,
            },
            "data": {"train": len(train_ds), "eval_noisy_quality": len(noisy_rows),
                     "eval_clean_preservation": len(clean_rows)},
        }, f, ensure_ascii=False, indent=2)

    print("TRAINING STARTED..." if not last_checkpoint else f"RESUMING TRAINING ({last_checkpoint})...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    if tracker.top:
        print("\nTOP 3 CHECKPOINTS (by combined selection score, kept on disk):")
        for i, c in enumerate(tracker.top, 1):
            print(f"  {i}. step {c['step']:>7d}   selection_score={c['score']:.5f}   ({c['path']})")
        best = tracker.top[0]
        best_model = AutoModelForSeq2SeqLM.from_pretrained(best["path"])
        best_model.save_pretrained(OUTPUT_DIR)
        print(f"-> Best (step {best['step']}) loaded from disk and saved to: {OUTPUT_DIR}")
    else:
        trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    summary = {
        "model": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "device": device,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": {
            "epoch": args.epoch, "max_steps": args.max_steps, "batch": args.batch,
            "learning_rate": args.lr, "warmup": args.warmup, "weight_decay": 0.01,
            "seed": args.seed, "w_quality": args.w, "CER_W": CER_W, "WER_W": WER_W,
            "eval_steps": args.eval_steps,
        },
        "data": {
            "train": len(train_ds),
            "eval_noisy_quality": len(noisy_rows),
            "eval_clean_preservation": len(clean_rows),
        },
        "total_steps": trainer.state.global_step,
        "total_minutes": round((time.time() - tracker.start_time) / 60, 2) if tracker.start_time else None,
        "top_3": [{"rank": i, "step": c["step"], "selection_score": round(c["score"], 5)}
                     for i, c in enumerate(tracker.top, 1)],
    }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Model saved: {OUTPUT_DIR}")

    if not args.skip_gold and not args.smoke_test:
        try:
            print("\nRunning gold evaluation (tweets/boun)...")
            import eval_model
            gold = eval_model.evaluate_gold(OUTPUT_DIR, beam=args.gold_beam)
            with open(os.path.join(OUTPUT_DIR, "gold_results.json"), "w", encoding="utf-8") as f:
                json.dump({"model": os.path.basename(OUTPUT_DIR), "beam": args.gold_beam,
                           "date": time.strftime("%Y-%m-%d %H:%M:%S"), "sets": gold},
                          f, ensure_ascii=False, indent=2)
            print("-> gold_results.json written.")
        except Exception as e:
            print(f"WARNING: gold evaluation skipped (error): {e}")

    print(f"DONE. For a report, run: python src/eval/rapor_uret.py")
    print(f"Report files: score_report.csv, dataset_report.csv, examples.jsonl, summary.json, training_config.json, gold_results.json")


if __name__ == "__main__":
    main()
