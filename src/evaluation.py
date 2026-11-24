"""Final evaluation of selective LLM review on CTU-13 held-out scenarios.

Run from repository root:
    python -m src.evaluation --responses results/gemma_responses.jsonl

The selected flow lists, not the order of LLM responses, define the strategies.
No model inference, training, tuning, or synthetic observations occur here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

SCENARIOS = (11, 12)
STRATEGIES = ("uncertainty", "random")
BUDGETS = (50, 100, 200)


def read_responses(paths: list[Path]) -> dict[tuple[int, str], dict]:
    """Accept completed calls (including invalid labels), but not provider errors.

    When a flow was logged more than once, only identical completed predictions
    for the same configuration are accepted. Reused cached replies do not
    create another inference or overwrite the original token counts.
    """
    responses: dict[tuple[int, str], dict] = {}
    configurations = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                scenario = int(row["scenario"])
                flow_id = str(row["flow_id"])
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid response record") from exc
            if scenario not in SCENARIOS:
                continue  # Validation pilot or other experimental scenario
            if not flow_id:
                raise ValueError(f"{path}:{line_number}: missing flow_id")
            settings = json.dumps(row.get("generation_settings"), sort_keys=True)
            configuration = (row.get("model"), row.get("backend"), row.get("prompt_id"), settings)
            configurations.add(configuration)
            key = (scenario, flow_id)
            error = row.get("error")
            prediction = row.get("prediction")
            if error not in (None, "invalid_response"):
                continue  # Failed API call may have been retried in a later batch
            if error is None and prediction not in (0, 1):
                raise ValueError(f"{path}:{line_number}: successful response without binary prediction")
            if error == "invalid_response" and prediction is not None:
                raise ValueError(f"{path}:{line_number}: invalid response must be unclassified")
            for field in ("input_tokens", "output_tokens"):
                value = row.get(field)
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError(f"{path}:{line_number}: invalid {field}")
            prev = responses.get(key)
            if prev is not None:
                if (prev.get("prediction") != prediction or prev.get("error") != error or
                    prev.get("input_tokens") != row.get("input_tokens") or
                    prev.get("output_tokens") != row.get("output_tokens")):
                    raise ValueError(f"Conflicting completed responses for {key}")
                # Keep the original non-cached entry when available.
                if prev.get("cache_hit") and not row.get("cache_hit"):
                    responses[key] = row
            else:
                responses[key] = row
    if len(configurations) > 1:
        raise ValueError("Different LLM model/prompt/generation configurations found in test responses")
    return responses


def classification_metrics(truth: np.ndarray, pred: np.ndarray, probabilities=None) -> dict:
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(truth, pred, labels=[0, 1]).ravel())
    return {
        "n": len(truth), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else float("nan"),
        "roc_auc_rf": float(roc_auc_score(truth, probabilities)) if probabilities is not None else float("nan"),
    }


def evaluate(results_dir: Path, response_paths: list[Path], output_dir: Path) -> pd.DataFrame:
    responses = read_responses(response_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    details: list[dict] = []
    all_required_keys: set[tuple[int, str]] = set()
    selections: dict[tuple[int, str, int], list[str]] = {}
    predictions: dict[int, pd.DataFrame] = {}

    for scenario in SCENARIOS:
        source = results_dir / f"scenario_{scenario}" / "predictions.csv"
        data = pd.read_csv(source, dtype={"flow_id": str})
        needed = {"flow_id", "scenario", "true_label", "ml_prediction", "ml_probability"}
        if not needed.issubset(data.columns):
            raise ValueError(f"Missing columns in {source}: {sorted(needed-set(data.columns))}")
        if data.flow_id.isna().any() or data.flow_id.duplicated().any():
            raise ValueError(f"Duplicate/empty flow IDs in {source}")
        if set(data.scenario.unique()) != {scenario}:
            raise ValueError(f"Wrong scenario in {source}")
        for col in ("true_label", "ml_prediction"):
            if not set(data[col].unique()).issubset({0, 1}):
                raise ValueError(f"Invalid binary values in {source}: {col}")
        data = data.set_index("flow_id", drop=False)
        predictions[scenario] = data
        for strategy in STRATEGIES:
            previous: set[str] = set()
            for k in BUDGETS:
                review_path = results_dir / f"scenario_{scenario}" / f"review_{strategy}_k{k}.csv"
                chosen = pd.read_csv(review_path, dtype={"flow_id": str})
                if len(chosen) != k or chosen.flow_id.isna().any() or chosen.flow_id.duplicated().any():
                    raise ValueError(f"Expected exactly {k} distinct flows in {review_path}")
                if "scenario" not in chosen or set(chosen.scenario.unique()) != {scenario}:
                    raise ValueError(f"Wrong scenario in {review_path}")
                ids = chosen.flow_id.tolist()
                if not set(ids).issubset(data.index):
                    raise ValueError(f"Unknown flow IDs in {review_path}")
                if not previous.issubset(ids):
                    raise ValueError(f"Budgets not nested in {review_path}")
                previous = set(ids)
                selections[(scenario, strategy, k)] = ids
                all_required_keys.update((scenario, flow_id) for flow_id in ids)

    missing = sorted(all_required_keys - responses.keys())
    if missing:
        raise ValueError(
            f"Incomplete LLM evaluation: missing completed responses for {len(missing)} selected flows. "
            f"First missing IDs: {missing[:8]}. Finish inference before evaluating."
        )

    # Disallow comparing runs that silently lack usage metadata: the proposal
    # requires actual token accounting, including invalid completed outputs.
    for key in all_required_keys:
        row = responses[key]
        if row.get("input_tokens") is None or row.get("output_tokens") is None:
            raise ValueError(f"Missing actual token counts for {key}; cannot report token efficiency")

    for scenario in SCENARIOS:
        data = predictions[scenario]
        truth = data.true_label.to_numpy(dtype=int)
        baseline = data.ml_prediction.to_numpy(dtype=int)
        base = classification_metrics(truth, baseline, data.ml_probability.to_numpy(dtype=float))
        summaries.append({
            "scenario": scenario, "strategy": "ml_only", "k": 0, **base,
            "delta_f1": 0., "delta_recall": 0., "delta_fpr": 0.,
            "corrected_errors": 0, "introduced_errors": 0, "net_errors_corrected": 0,
            "valid_reviews": 0, "invalid_reviews": 0, "reviewed_flows": 0,
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "net_errors_per_1000_tokens": float("nan"), "f1_gain_pp_per_1000_tokens": float("nan"),
        })
        for strategy in STRATEGIES:
            for k in BUDGETS:
                ids = selections[(scenario, strategy, k)]
                updated = data.ml_prediction.copy()
                corrected = introduced = valid = invalid = 0
                input_tokens = output_tokens = 0
                for flow_id in ids:
                    row = responses[(scenario, flow_id)]
                    ml_label = int(data.at[flow_id, "ml_prediction"])
                    true_label = int(data.at[flow_id, "true_label"])
                    llm_label = row.get("prediction")
                    input_tokens += row["input_tokens"]
                    output_tokens += row["output_tokens"]
                    if llm_label is None:
                        invalid += 1
                    else:
                        valid += 1
                        updated.at[flow_id] = int(llm_label)
                        if ml_label != true_label and llm_label == true_label:
                            corrected += 1
                        if ml_label == true_label and llm_label != true_label:
                            introduced += 1
                    details.append({
                        "scenario": scenario, "strategy": strategy, "k": k,
                        "flow_id": flow_id, "true_label": true_label,
                        "ml_prediction": ml_label, "llm_prediction": llm_label,
                        "final_prediction": int(updated.at[flow_id]),
                        "error": row.get("error"),
                        "input_tokens": row["input_tokens"],
                        "output_tokens": row["output_tokens"],
                        "cache_hit": bool(row.get("cache_hit", False)),
                    })
                final = classification_metrics(truth, updated.to_numpy(dtype=int))
                total_tokens = input_tokens + output_tokens
                net = corrected - introduced
                summaries.append({
                    "scenario": scenario, "strategy": strategy, "k": k, **final,
                    "delta_f1": final["f1"] - base["f1"],
                    "delta_recall": final["recall"] - base["recall"],
                    "delta_fpr": final["false_positive_rate"] - base["false_positive_rate"],
                    "corrected_errors": corrected, "introduced_errors": introduced,
                    "net_errors_corrected": net, "reviewed_flows": k,
                    "valid_reviews": valid, "invalid_reviews": invalid,
                    "input_tokens": input_tokens, "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "net_errors_per_1000_tokens": (net / total_tokens * 1000) if total_tokens else float("nan"),
                    "f1_gain_pp_per_1000_tokens": (100 * (final["f1"] - base["f1"]) / total_tokens * 1000) if total_tokens else float("nan"),
                })

    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "evaluation_summary.csv", index=False)
    pd.DataFrame(details).to_csv(output_dir / "reviewed_flows_evaluation.csv", index=False)

    # Actual API usage is different from the logical per-strategy usage above:
    # each distinct, non-cached response is billed only once, even if used in
    # several budgets or both selection strategies.
    physical = {}
    for key in all_required_keys:
        row = responses[key]
        if row.get("cache_hit") is False:
            identifier = row.get("cache_key") or (key[0], key[1])
            physical[identifier] = row
    actual_usage = {
        "unique_selected_flows": len(all_required_keys),
        "recorded_api_calls_for_selected_flows": len(physical),
        "actual_input_tokens_recorded": sum(r["input_tokens"] for r in physical.values()),
        "actual_output_tokens_recorded": sum(r["output_tokens"] for r in physical.values()),
        "note": "Logical token usage is summed separately per strategy/K. Actual usage excludes cached copies; calls outside the selected sets are not counted.",
    }
    (output_dir / "api_usage.json").write_text(json.dumps(actual_usage, indent=2), encoding="utf-8")

    lines = ["# Final Evaluation — RF vs. Selective LLM Review", "",
             "The same held-out flows are evaluated for ML-only, uncertainty review, and random review.",
             "Only selected flows with valid LLM labels replace the RF prediction; invalid replies retain the RF prediction.",
             "All values below are based on completed real LLM responses, not estimates.", ""]
    for scenario in SCENARIOS:
        lines += [f"## Scenario {scenario}", "",
                  "| Strategy | K | Precision | Recall | F1 | FPR | Corrected | Introduced | Invalid | Tokens | Net corrections / 1k tokens |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, row in summary[summary.scenario == scenario].iterrows():
            efficiency = f"{row.net_errors_per_1000_tokens:.4f}" if pd.notna(row.net_errors_per_1000_tokens) else "—"
            lines.append(f"| {row.strategy} | {row.k} | {row.precision:.4f} | {row.recall:.4f} | {row.f1:.4f} | {row.false_positive_rate:.4f} | {row.corrected_errors} | {row.introduced_errors} | {row.invalid_reviews} | {row.total_tokens} | {efficiency} |")
        lines += [""]
    lines += ["## Token accounting", "",
              "Per-strategy/K tokens count the original inference cost of each selected flow, even when the response was reused from the cache. These are comparable hypothetical strategy costs, **not additive across strategies**.",
              "`api_usage.json` separately reports recorded non-cached calls among selected flows; it does not include unrelated validation/pilot calls.",
              "", "## Interpretation boundaries", "",
              "- RF probabilities are uncalibrated.",
              "- Background and ambiguously labeled traffic are excluded.",
              "- The test scenarios were selected after exploratory work on scenario 10; the evaluation should be described accordingly.",
              "- Only actual LLM calls establish correction ability; error concentration alone does not.", ""]
    (output_dir / "evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")

    for scenario in SCENARIOS:
        subset = summary[(summary.scenario == scenario) & (summary.k > 0)]
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        baseline_f1 = float(summary[(summary.scenario == scenario) & (summary.k == 0)].iloc[0].f1)
        ax.axhline(baseline_f1, linestyle="--", label="RF only")
        for strategy in STRATEGIES:
            s = subset[subset.strategy == strategy].sort_values("k")
            ax.plot(s.k, s.f1, marker="o", label=f"RF + LLM ({strategy})")
        ax.set(xlabel="Reviewed flows (K)", ylabel="Final F1-score", title=f"Scenario {scenario}: F1 vs. review budget")
        ax.set_xticks(BUDGETS)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"scenario_{scenario}_f1_by_budget.png", dpi=160)
        plt.close(fig)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate final CTU-13 selective LLM results without inference.")
    parser.add_argument("--responses", nargs="+", type=Path, required=True,
                        help="Gemma response JSONL files (one or more complete runs)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_evaluation"))
    args = parser.parse_args()
    summary = evaluate(args.results_dir, args.responses, args.output_dir)
    print(summary[["scenario", "strategy", "k", "f1", "recall", "false_positive_rate", "corrected_errors", "introduced_errors", "total_tokens"]].to_string(index=False))
    print(f"\nFinal evaluation saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
