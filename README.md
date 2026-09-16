# Turkish Text Normalizer

A fine-tuned [byT5-small](https://huggingface.co/google/byt5-small) model that rewrites informal, misspelled, or dialectal Turkish sentences into standard written Turkish — dropped diacritics, missing letters, colloquial contractions, ASCII-only typing, and regional spelling.

**Model on Hugging Face:** [erdemKocaogluu/byt5-small-tr-normalizer](https://huggingface.co/erdemKocaogluu/byt5-small-tr-normalizer)

```
sicil belgmi alcam            -> Sicil belgimi alacağım
araç lsansı nasl alacaım      -> araç lisansı nasıl alacağım
muhtara gitcem ne götüreyim   -> Muhtara gideceğim ne götüreyim
```

## How it works

The model uses byT5's byte-level tokenizer — there is no fixed vocabulary or BPE merging, every UTF-8 byte is its own token. That makes the model reason directly over characters, which is a natural fit for a task that is mostly about fixing individual letters and word boundaries rather than rewriting meaning.

Input is prefixed with `"düzelt: "` before being passed to the model, matching the format used during training.

## Training data

Training pairs come from six public datasets that already provide noisy/clean Turkish sentence pairs (grammar correction, typo validation, text normalization). Four sources are permissively licensed (MIT, Apache-2.0, or CC0-1.0); two (Wikipedia-derived) are CC BY-SA 3.0 — see the [model card](https://huggingface.co/erdemKocaogluu/byt5-small-tr-normalizer#data-sources) for the full list with links, licenses, and attribution.

The repository also includes a custom rule-based noiser (`src/noise/`) that can generate additional synthetic training pairs — typo, spacing, repetition, vowel-harmony slips, `de/da` conjunction errors, and deasciification — but it was not used for the checkpoint published here; this model was trained entirely on the six datasets above.

## Training procedure

Checkpoints were saved every 2,000 steps and evaluated on a held-out validation split. Each checkpoint is scored with:

```
Q_good         = 1 - (0.6 * CER + 0.4 * WER)      # how well it fixes noisy sentences
unchanged_rate = pred == source on clean sentences  # how well it leaves correct text alone
selection_score = 0.7 * Q_good + 0.3 * unchanged_rate
```

The top 3 checkpoints by `selection_score` are kept on disk; the best one is promoted to the final model and its weights are hash-verified before being published. `--devam` resumes training from the last checkpoint if it's interrupted.

## Results

![Learning curve — selection score](reports/training/01_learning_curve.png)

### Held-out gold sets (naturally occurring, human-written Turkish — never seen during training)

| Test set | n | WER (model) | WER (no correction) | F1 | Over-correction |
|---|---|---|---|---|---|
| tweets | 1,742 | 0.161 | 0.336 | 0.667 | 0.6% |
| boun | 507 | 0.093 | 0.129 | 0.590 | 4.3% |

- *tweets* — Köksal, A. T., Bozal, Ö., Yürekli, E., & Gezici, G. (2020). [#Turki$hTweets: A Benchmark Dataset for Turkish Text Correction](https://aclanthology.org/2020.findings-emnlp.374/). *Findings of ACL: EMNLP 2020*, pages 4190–4198.
- *boun* — Kara, A., Marouf Sofian, F., Bond, A., & Şahin, G. (2023). [GECTurk: Grammatical Error Correction and Detection Dataset for Turkish](https://aclanthology.org/2023.findings-ijcnlp.26/). *Findings of ACL: IJCNLP-AACL 2023*, pages 278–290.

![Gold evaluation — WER, model vs. do-nothing baseline](reports/training/07_gold_wer.png)

More charts (loss curves, quality-vs-preservation trade-off, learning rate schedule) and full per-checkpoint tables are in [`reports/training/`](reports/training/) and the interactive dashboard [`reports/training/dashboard.html`](reports/training/dashboard.html) — also viewable live at **[erdemkocaogluu.github.io/byt5-turkish-normalizer](https://erdemkocaogluu.github.io/byt5-turkish-normalizer/reports/training/dashboard.html)**.

## Limitations

- Generalizes very well to mechanical, deterministic corrections (like deasciification), where the rule mapping is unambiguous.
- Comparatively weaker on organic, human-style noise that doesn't closely match the synthetic noise patterns it was trained on, and on long, multi-clause informal sentences.
- Corrects spelling, spacing, and word-internal errors; does not reorder words or rewrite sentence structure.

## Repository structure

```
src/
  data/    dataset download, cleaning, and label formatting
  noise/   the synthetic noiser and its lexical resources
  eval/    metrics, gold-set evaluation, and report generation
  train/   the training script
web/
  app.py   a single-file Flask demo (type a sentence, see the correction)
reports/
  training/  training charts, tables, and the interactive dashboard
```

## Running the local demo

```bash
pip install -r requirements.txt
python web/app.py
```

Then open `http://127.0.0.1:7860`. The demo loads the model from Hugging Face (`erdemKocaogluu/byt5-small-tr-normalizer`) — no local weights needed.

## Reproducing training

```bash
python src/data/format_label.py full     # build data/labeled/*.jsonl from raw sources
python src/data/build_final.py           # merge into data/final/train.jsonl + val.jsonl
python src/train/train.py                # fine-tune byT5-small
python src/eval/rapor_uret.py            # regenerate the charts in reports/training/
```

## License

Code is released under the [MIT License](LICENSE). The model itself is released separately on Hugging Face under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial) — see the [model card](https://huggingface.co/erdemKocaogluu/byt5-small-tr-normalizer) for details.
