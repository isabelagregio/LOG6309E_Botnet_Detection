"""Analyze CTU-13 Random Forest uncertainty on held-out scenarios 11 and 12.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("results")
SCENARIOS = (11, 12)
REVIEW_BUDGETS = (50, 100, 200)


def load_predictions(path):
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}")
    df = pd.read_csv(path)
    needed = {"true_label", "ml_prediction", "ml_probability"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{path}: empty predictions file")
    for name in ("true_label", "ml_prediction", "ml_probability"):
        df[name] = pd.to_numeric(df[name], errors="raise")
    if df[["true_label", "ml_prediction", "ml_probability"]].isna().any().any():
        raise ValueError(f"{path}: null values in required columns")
    if not df["true_label"].isin([0, 1]).all() or not df["ml_prediction"].isin([0, 1]).all():
        raise ValueError(f"{path}: binary labels and predictions must be 0 or 1")
    if not df["ml_probability"].between(0, 1).all():
        raise ValueError(f"{path}: probabilities must be between 0 and 1")
    # Recompute the ranking from probabilities, independent of any saved uncertainty convention.
    df["distance_to_half"] = (df["ml_probability"] - 0.5).abs()
    df["false_negative"] = (df["true_label"] == 1) & (df["ml_prediction"] == 0)
    df["false_positive"] = (df["true_label"] == 0) & (df["ml_prediction"] == 1)
    df["error"] = df["false_negative"] | df["false_positive"]
    return df


def probability_statistics(df):
    groups = [
        ("Normal (true label 0)", df["true_label"] == 0),
        ("Botnet (true label 1)", df["true_label"] == 1),
        ("False negatives", df["false_negative"]),
        ("False positives", df["false_positive"]),
        ("True positives", (df["true_label"] == 1) & (df["ml_prediction"] == 1)),
        ("True negatives", (df["true_label"] == 0) & (df["ml_prediction"] == 0)),
    ]
    rows = []
    for name, mask in groups:
        values = df.loc[mask, "ml_probability"]
        quant = values.quantile([0, .1, .25, .5, .75, .9, 1]) if len(values) else None
        rows.append({
            "group": name, "n": len(values),
            **{key: (float(quant.loc[p]) if quant is not None else np.nan)
               for key, p in [("min", 0), ("p10", .1), ("p25", .25),
                              ("median", .5), ("p75", .75), ("p90", .9), ("max", 1)]},
        })
    return pd.DataFrame(rows)


def analyze_scenario(scenario):
    path = RESULTS_DIR / f"scenario_{scenario}" / "predictions.csv"
    output_dir = path.parent / "uncertainty_analysis"
    df = load_predictions(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(df)
    total_fn = int(df["false_negative"].sum())
    total_fp = int(df["false_positive"].sum())
    total_errors = int(df["error"].sum())
    total_botnet = int((df["true_label"] == 1).sum())
    stats = probability_statistics(df)
    stats.to_csv(output_dir / "probability_statistics.csv", index=False)

    # mergesort preserves file order for exact ties, ensuring reproducibility.
    ranked = df.sort_values("distance_to_half", kind="mergesort")
    rows = []
    for requested_k in REVIEW_BUDGETS:
        k = min(requested_k, n)
        subset = ranked.head(k)
        selected_fn = int(subset["false_negative"].sum())
        selected_fp = int(subset["false_positive"].sum())
        selected_errors = selected_fn + selected_fp
        expected_fn = k * total_fn / n
        expected_fp = k * total_fp / n
        expected_errors = k * total_errors / n
        rows.append({
            "scenario": scenario,
            "requested_k": requested_k,
            "k": k,
            "reviewed_percent": 100 * k / n,
            "false_negatives_found": selected_fn,
            "false_positives_found": selected_fp,
            "all_errors_found": selected_errors,
            "fn_coverage_percent": 100 * selected_fn / total_fn if total_fn else 0.0,
            "fp_coverage_percent": 100 * selected_fp / total_fp if total_fp else 0.0,
            "error_precision_percent": 100 * selected_errors / k,
            "random_expected_fn": expected_fn,
            "random_expected_fp": expected_fp,
            "random_expected_all_errors": expected_errors,
            "fn_enrichment_vs_random": selected_fn / expected_fn if expected_fn else np.nan,
            "all_error_enrichment_vs_random": selected_errors / expected_errors if expected_errors else np.nan,
        })
    analysis = pd.DataFrame(rows)
    analysis.to_csv(output_dir / "analysis_summary.csv", index=False)

    # Observed probabilities only; no estimated or synthetic classifications.
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 41)
    for label, name in [(0, "Normal"), (1, "Botnet")]:
        values = df.loc[df["true_label"] == label, "ml_probability"]
        if len(values):
            ax.hist(values, bins=bins, alpha=.55, label=f"{name} (n={len(values):,})")
    ax.axvline(.5, ls="--", lw=1.3, color="black", label="Classification threshold = 0.5")
    ax.set(xlabel="P(Botnet)", ylabel="Number of flows",
           title=f"Scenario {scenario} — predicted probability by true class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "probability_distribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(analysis))
    ax.bar(x - .18, analysis["false_negatives_found"], width=.36, label="Uncertainty top-K: observed FN")
    ax.bar(x + .18, analysis["random_expected_fn"], width=.36, label="Random: expected FN")
    ax.set_xticks(x, [f"K={k}" for k in analysis["k"]])
    ax.set(ylabel="False negatives among selected flows",
           title=f"Scenario {scenario} — missed botnets found by selection")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "false_negative_coverage.png", dpi=160)
    plt.close(fig)

    lines = [
        f"CTU-13 — Scenario {scenario}: Random Forest uncertainty analysis",
        "=" * 67,
        f"Source: {path.resolve()}",
        f"Test flows: {n:,}",
        f"True Botnet flows: {total_botnet:,}",
        f"False negatives (missed botnets): {total_fn:,}",
        f"False positives (false alarms): {total_fp:,}",
        f"Total errors: {total_errors:,}",
        "",
        "Selection: smallest |P(Botnet) - 0.5|, stable ordering for ties.",
        "Random comparison: MATHEMATICAL EXPECTATION, not observed random reviews.",
        "No LLM is called; no LLM corrections are estimated.",
        "",
        "PROBABILITIES BY TRUE CLASS / PREDICTION OUTCOME",
        stats.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        "",
        "TOP-K ERROR COVERAGE",
    ]
    for r in rows:
        lines.append(
            f"K={r['k']}: FN={r['false_negatives_found']} "
            f"({r['fn_coverage_percent']:.2f}% of all FN), "
            f"FP={r['false_positives_found']}, total errors={r['all_errors_found']}; "
            f"random expected FN={r['random_expected_fn']:.2f}, "
            f"expected total errors={r['random_expected_all_errors']:.2f}; "
            f"FN enrichment={r['fn_enrichment_vs_random']:.2f}x, "
            f"all-error enrichment={r['all_error_enrichment_vs_random']:.2f}x"
        )
    lines += [
        "", "CONFIDENT FALSE NEGATIVES",
        f"FN with P(Botnet) < 0.10: {int((df['false_negative'] & (df['ml_probability'] < .1)).sum())}",
        f"FN with P(Botnet) < 0.25: {int((df['false_negative'] & (df['ml_probability'] < .25)).sum())}",
        "", "Interpretation:",
        "- Top-K selection captures model errors, not confirmed LLM-correctable flows.",
        "- Compare random EXPECTATIONS to observed top-K counts, not to actual LLM results.",
        "- Never change model, thresholds, or review policy using held-out test labels.",
    ]
    report = "\n".join(lines) + "\n"
    (output_dir / "analysis_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Files saved to: {output_dir.resolve()}\n")
    return analysis


def main():
    summaries = [analyze_scenario(scenario) for scenario in SCENARIOS]
    combined = pd.concat(summaries, ignore_index=True)
    combined.to_csv(RESULTS_DIR / "uncertainty_scenarios_comparison.csv", index=False)
    print("Both scenarios completed.")
    print(f"Combined comparison: {(RESULTS_DIR / 'uncertainty_scenarios_comparison.csv').resolve()}")


if __name__ == "__main__":
    main()
