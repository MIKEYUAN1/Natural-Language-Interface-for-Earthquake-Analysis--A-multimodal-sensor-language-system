# ==========================================
# GENERATION EVALUATION: SGS + COMPLETENESS + ROUGE-L + BERTScore
# ==========================================
import re
import math
import pandas as pd
from collections import Counter

def extract_fields(text: str):
    text = str(text).strip()
    fields = {
        "station": None,
        "magnitude": None,
        "distance_km": None,
        "p_arrival": None,
        "s_arrival": None,
        "dominant_component": None,
    }

    m = re.search(r"station\s+([A-Za-z0-9]+)", text, re.IGNORECASE)
    if m:
        fields["station"] = m.group(1).upper()

    m = re.search(r"\bM\s*([0-9]+(?:\.[0-9]+)?)\b", text, re.IGNORECASE)
    if m:
        fields["magnitude"] = float(m.group(1))
    else:
        m = re.search(r"magnitude\s+(?:is\s+)?([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
        if m:
            fields["magnitude"] = float(m.group(1))

    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*km\b", text, re.IGNORECASE)
    if m:
        fields["distance_km"] = float(m.group(1))

    m = re.search(r"\bP(?:-wave)?(?:\s+arrival)?(?:\s+is)?(?:\s+annotated)?(?:\s+at)?\s*(?:sample\s*)?([0-9]+)\b", text, re.IGNORECASE)
    if m:
        fields["p_arrival"] = int(m.group(1))
    else:
        m = re.search(r"\bP\s*at\s*([0-9]+)\b", text, re.IGNORECASE)
        if m:
            fields["p_arrival"] = int(m.group(1))

    m = re.search(r"\bS(?:-wave)?(?:\s+arrival)?(?:\s+is)?(?:\s+annotated)?(?:\s+at)?\s*(?:sample\s*)?([0-9]+)\b", text, re.IGNORECASE)
    if m:
        fields["s_arrival"] = int(m.group(1))
    else:
        m = re.search(r"\bS\s*at\s*([0-9]+)\b", text, re.IGNORECASE)
        if m:
            fields["s_arrival"] = int(m.group(1))

    m = re.search(r"dominant\s+([ENZ])\s+component", text, re.IGNORECASE)
    if m:
        fields["dominant_component"] = m.group(1).upper()
    else:
        m = re.search(r"dominant\s+(?:component|channel)\s+(?:is\s+)?([ENZ])\b", text, re.IGNORECASE)
        if m:
            fields["dominant_component"] = m.group(1).upper()

    return fields

def completeness_score(text: str):
    fields = extract_fields(text)
    present = sum(v is not None for v in fields.values())
    return present / len(fields), fields

def clipped_score(err, tol):
    return max(0.0, 1.0 - float(err) / float(tol))

def seismic_grounding_score(gt_text: str, gen_text: str):
    gt = extract_fields(gt_text)
    pred = extract_fields(gen_text)

    station_exact = float(
        gt["station"] is not None and pred["station"] is not None and gt["station"] == pred["station"]
    )
    dominant_exact = float(
        gt["dominant_component"] is not None and pred["dominant_component"] is not None
        and gt["dominant_component"] == pred["dominant_component"]
    )

    mag_err = abs(pred["magnitude"] - gt["magnitude"]) if (
        pred["magnitude"] is not None and gt["magnitude"] is not None
    ) else None
    dist_err = abs(pred["distance_km"] - gt["distance_km"]) if (
        pred["distance_km"] is not None and gt["distance_km"] is not None
    ) else None
    p_err = abs(pred["p_arrival"] - gt["p_arrival"]) if (
        pred["p_arrival"] is not None and gt["p_arrival"] is not None
    ) else None
    s_err = abs(pred["s_arrival"] - gt["s_arrival"]) if (
        pred["s_arrival"] is not None and gt["s_arrival"] is not None
    ) else None

    mag_score = clipped_score(mag_err, 0.3) if mag_err is not None else 0.0
    dist_score = clipped_score(dist_err, 20.0) if dist_err is not None else 0.0
    p_score = clipped_score(p_err, 80.0) if p_err is not None else 0.0
    s_score = clipped_score(s_err, 120.0) if s_err is not None else 0.0

    sgs = (station_exact + dominant_exact + mag_score + dist_score + p_score + s_score) / 6.0

    detail = {
        "station_exact": station_exact,
        "dominant_exact": dominant_exact,
        "mag_err": mag_err,
        "dist_err_km": dist_err,
        "p_err_samples": p_err,
        "s_err_samples": s_err,
        "mag_score": mag_score,
        "dist_score": dist_score,
        "p_score": p_score,
        "s_score": s_score,
    }
    return sgs, detail

def tokenize_for_rouge(text: str):
    return re.findall(r"[A-Za-z0-9\.]+", str(text).lower())

def lcs_length(xs, ys):
    if not xs or not ys:
        return 0
    prev = [0] * (len(ys) + 1)
    for x in xs:
        curr = [0]
        for j, y in enumerate(ys, start=1):
            if x == y:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]

def rouge_l_f1(reference: str, candidate: str):
    ref_toks = tokenize_for_rouge(reference)
    cand_toks = tokenize_for_rouge(candidate)
    if len(ref_toks) == 0 or len(cand_toks) == 0:
        return 0.0
    lcs = lcs_length(ref_toks, cand_toks)
    prec = lcs / len(cand_toks)
    rec = lcs / len(ref_toks)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def compute_bertscore_batch(references, candidates, device=None):
    try:
        from bert_score import score as bertscore_score
        P, R, F1 = bertscore_score(
            candidates,
            references,
            lang="en",
            device=device,
            rescale_with_baseline=True,
            verbose=True,
        )
        return [float(x) for x in F1]
    except Exception as e:
        print("BERTScore unavailable, skipping. Reason:", repr(e))
        return [None] * len(candidates)

def evaluate_generation_metrics(model, cfg, tokenizer, n_samples=50, instruction=None, save_csv=True):
    eval_dataset = PreFilteredSTEADDataset(cfg, tokenizer=tokenizer, stage=1, split="eval")
    n = min(n_samples, len(eval_dataset))

    refs = []
    gens = []
    rows = []

    for i in range(n):
        sample = eval_dataset[i]
        trace_name = sample["trace_name"]
        gt_text = eval_dataset.reports[trace_name]["report"]
        gen_text = generate_report(model, cfg, tokenizer, sample["waveform"], instruction=instruction)

        comp, gen_fields = completeness_score(gen_text)
        gt_comp, gt_fields = completeness_score(gt_text)
        sgs, detail = seismic_grounding_score(gt_text, gen_text)
        rouge_l = rouge_l_f1(gt_text, gen_text)

        refs.append(gt_text)
        gens.append(gen_text)

        row = {
            "idx": i,
            "trace_name": trace_name,
            "ground_truth": gt_text,
            "generated": gen_text,
            "sgs": sgs,
            "completeness": comp,
            "gt_completeness": gt_comp,
            "rougeL_f1": rouge_l,
            **detail,
            "gt_station": gt_fields["station"],
            "pred_station": gen_fields["station"],
            "gt_magnitude": gt_fields["magnitude"],
            "pred_magnitude": gen_fields["magnitude"],
            "gt_distance_km": gt_fields["distance_km"],
            "pred_distance_km": gen_fields["distance_km"],
            "gt_p_arrival": gt_fields["p_arrival"],
            "pred_p_arrival": gen_fields["p_arrival"],
            "gt_s_arrival": gt_fields["s_arrival"],
            "pred_s_arrival": gen_fields["s_arrival"],
            "gt_dominant_component": gt_fields["dominant_component"],
            "pred_dominant_component": gen_fields["dominant_component"],
        }
        rows.append(row)

    bertscores = compute_bertscore_batch(
        refs,
        gens,
        device=cfg.device if str(cfg.device).startswith("cuda") else "cpu",
    )
    for row, bs in zip(rows, bertscores):
        row["bertscore_f1"] = bs

    df = pd.DataFrame(rows)

    summary = {
        "num_samples": len(df),
        "mean_SGS": float(df["sgs"].mean()),
        "mean_completeness": float(df["completeness"].mean()),
        "mean_ROUGE_L_F1": float(df["rougeL_f1"].mean()),
        "mean_BERTScore_F1": float(df["bertscore_f1"].dropna().mean()) if df["bertscore_f1"].notna().any() else None,
        "station_exact_rate": float(df["station_exact"].mean()),
        "dominant_exact_rate": float(df["dominant_exact"].mean()),
        "mean_mag_err": float(df["mag_err"].dropna().mean()) if df["mag_err"].notna().any() else None,
        "mean_dist_err_km": float(df["dist_err_km"].dropna().mean()) if df["dist_err_km"].notna().any() else None,
        "mean_p_err_samples": float(df["p_err_samples"].dropna().mean()) if df["p_err_samples"].notna().any() else None,
        "mean_s_err_samples": float(df["s_err_samples"].dropna().mean()) if df["s_err_samples"].notna().any() else None,
    }

    print("=" * 80)
    print("GENERATION EVALUATION SUMMARY")
    print("=" * 80)
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nSample rows:")
    display_cols = [
        "trace_name", "sgs", "completeness", "rougeL_f1", "bertscore_f1",
        "station_exact", "dominant_exact", "mag_err", "dist_err_km", "p_err_samples", "s_err_samples"
    ]
    display(df[display_cols].head(10))

    if save_csv:
        save_dir = f"{CODE_DIR}/stage1"
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "generation_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nSaved per-sample metrics to: {csv_path}")

    eval_dataset.close()
    return df, summary

# Run on a small eval subset first; increase n_samples after the smoke test looks correct.
metrics_df, metrics_summary = evaluate_generation_metrics(
    model,
    cfg,
    tokenizer,
    n_samples=20,
    instruction="Describe this seismic event in one sentence.",
    save_csv=True,
)
