import difflib
from jiwer import wer, cer
import sacrebleu


def _align_ops(a_tokens, b_tokens):
    sm = difflib.SequenceMatcher(a=a_tokens, b=b_tokens, autojunk=False)
    ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                ops.append(("equal", i1 + k, j1 + k))
        elif tag == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                ai = i1 + k if i1 + k < i2 else None
                bj = j1 + k if j1 + k < j2 else None
                ops.append(("replace", ai, bj))
        elif tag == "delete":
            for k in range(i1, i2):
                ops.append(("delete", k, None))
        elif tag == "insert":
            for k in range(j1, j2):
                ops.append(("insert", None, k))
    return ops


def token_level_counts(source, target, pred):
    s, t, p = source.split(), target.split(), pred.split()

    tgt_of = {}
    st_ops = _align_ops(s, t)
    for tag, si, tj in st_ops:
        if si is None:
            continue
        tgt_of[si] = t[tj] if tj is not None else None

    pred_of = {}
    sp_ops = _align_ops(s, p)
    for tag, si, pj in sp_ops:
        if si is None:
            continue
        pred_of[si] = p[pj] if pj is not None else None

    needed = correct = changed = correct_changed = 0
    already_ok = overcorrected = 0

    for si in range(len(s)):
        want = tgt_of.get(si, s[si])
        got  = pred_of.get(si, s[si])
        src_tok = s[si]

        needs_change = (want != src_tok)
        model_changed = (got != src_tok)

        if needs_change:
            needed += 1
            if got == want:
                correct += 1
        else:
            already_ok += 1
            if model_changed:
                overcorrected += 1

        if model_changed:
            changed += 1
            if got == want:
                correct_changed += 1

    return dict(needed=needed, correct=correct, changed=changed,
                correct_changed=correct_changed, already_ok=already_ok,
                overcorrected=overcorrected)


def evaluate(sources, targets, preds):
    assert len(sources) == len(targets) == len(preds)
    n = len(sources)

    corpus_wer = wer(targets, preds)
    corpus_cer = cer(targets, preds)
    chrf = sacrebleu.corpus_chrf(preds, [targets]).score
    exact = sum(1 for t, p in zip(targets, preds) if t.strip() == p.strip()) / n

    tot = dict(needed=0, correct=0, changed=0, correct_changed=0,
               already_ok=0, overcorrected=0)
    for s, t, p in zip(sources, targets, preds):
        c = token_level_counts(s, t, p)
        for k in tot:
            tot[k] += c[k]

    recall = tot["correct"] / tot["needed"] if tot["needed"] else 0.0
    precision = tot["correct_changed"] / tot["changed"] if tot["changed"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    overcorr = tot["overcorrected"] / tot["already_ok"] if tot["already_ok"] else 0.0

    return {
        "n": n,
        "WER": round(corpus_wer, 4),
        "CER": round(corpus_cer, 4),
        "chrF": round(chrf, 2),
        "exact_match": round(exact, 4),
        "norm_precision": round(precision, 4),
        "norm_recall": round(recall, 4),
        "norm_F1": round(f1, 4),
        "overcorrection": round(overcorr, 4),
        "_tokens_needed": tot["needed"],
        "_tokens_already_ok": tot["already_ok"],
    }


def selection_scores(quality_target, quality_pred, preservation_source, preservation_pred,
                   a=0.6, b=0.4, w=0.7):
    if quality_target:
        q_cer = cer(quality_target, quality_pred)
        q_wer = wer(quality_target, quality_pred)
    else:
        q_cer = q_wer = 0.0
    q_good = 1.0 - (a * q_cer + b * q_wer)

    if preservation_source:
        unchanged = sum(1 for s, p in zip(preservation_source, preservation_pred)
                        if s.strip() == p.strip()) / len(preservation_source)
    else:
        unchanged = 0.0

    score = w * q_good + (1.0 - w) * unchanged
    return {"CER": q_cer, "WER": q_wer, "Q_good": q_good,
            "unchanged_rate": unchanged, "selection_score": score}


def print_report(name, m):
    print(f"\n{'='*56}\n  {name}   (n={m['n']})\n{'='*56}")
    print(f"  WER              : {m['WER']:.4f}   (lower is better)")
    print(f"  CER              : {m['CER']:.4f}   (lower is better)")
    print(f"  chrF             : {m['chrF']:.2f}    (higher is better)")
    print(f"  Exact match      : {m['exact_match']:.4f}")
    print(f"  ---- normalization (tokens that needed to change only) ----")
    print(f"  Precision        : {m['norm_precision']:.4f}")
    print(f"  Recall           : {m['norm_recall']:.4f}")
    print(f"  F1               : {m['norm_F1']:.4f}   <- main metric")
    print(f"  ---- safety ----")
    print(f"  Over-correction  : {m['overcorrection']:.4f}   (lower is better; damaging already-correct tokens)")
    print(f"  (tokens needing correction: {m['_tokens_needed']}, already correct: {m['_tokens_already_ok']})")


if __name__ == "__main__":
    s = "benım arkadasım gelcem"
    t = "benim arkadaşım geleceğim"
    print("copy   :", evaluate([s],[t],[s]))
    print("perfect:", evaluate([s],[t],[t]))
    print("overcor:", evaluate(["ben geldim"],["ben geldim"],["ben gittim"]))
