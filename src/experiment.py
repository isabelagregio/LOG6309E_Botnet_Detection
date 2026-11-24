from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.ctu_data import CATEGORICAL, FEATURES, NUMERIC, find_scenarios, read_scenario

# Fixed experiment protocol from the proposal.
# Fixed experiment configuration
DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("results")

# Final experimental protocol
TRAIN_SCENARIOS = [1, 2, 4, 5, 6, 7, 8]
VALIDATION_SCENARIOS = [3, 9]

# Independent held-out scenarios
TEST_SCENARIOS = [11, 12]

REVIEW_BUDGETS = [50, 100, 200]
SEED = 42


def make_model() -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("categorical", categorical, CATEGORICAL),
    ])
    classifier = RandomForestClassifier(
        n_estimators=100, max_depth=15, min_samples_leaf=2,
        random_state=SEED, n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def evaluate(scored: pd.DataFrame) -> dict:
    truth = scored["true_label"].to_numpy(dtype=int)
    pred = scored["ml_prediction"].to_numpy(dtype=int)
    probability = scored["ml_probability"].to_numpy(dtype=float)
    tn, fp, fn, tp = map(int, confusion_matrix(truth, pred, labels=[0, 1]).ravel())
    return {
        "n": len(scored),
        "normal": int((truth == 0).sum()),
        "botnet": int((truth == 1).sum()),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": float(precision_score(truth, pred, zero_division=0)),
        "recall": float(recall_score(truth, pred, zero_division=0)),
        "f1": float(f1_score(truth, pred, zero_division=0)),
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "roc_auc": float(roc_auc_score(truth, probability)) if len(np.unique(truth)) == 2 else None,
    }


def score(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    classes = list(model.named_steps["classifier"].classes_)
    botnet_col = classes.index(1)
    probabilities = model.predict_proba(frame[FEATURES])[:, botnet_col]
    out = frame.copy()
    out["ml_probability"] = probabilities
    out["ml_prediction"] = (probabilities >= 0.5).astype("int8")
    out["uncertainty"] = 1 - 2 * np.abs(probabilities - 0.5)
    out["ml_correct"] = (out["true_label"] == out["ml_prediction"]).astype("int8")
    return out


def select_for_review(scored: pd.DataFrame, out_dir: Path) -> list[dict]:
    """Same observed flows, ranked either by uncertainty or seeded random order.

    One full order per method makes budgets nested; K=50 is a subset of K=100.
    LLM input files NEVER contain ground-truth labels or ML correctness.
    """
    uncertain = scored.sort_values(["uncertainty", "flow_id"], ascending=[False, True], kind="stable")
    random = scored.sample(frac=1, random_state=SEED)
    summary = []
    handoff_cols = ["flow_id", "scenario", *FEATURES, "ml_probability", "ml_prediction", "uncertainty"]
    for k in REVIEW_BUDGETS:
        if k > len(scored):
            raise ValueError(f"Review budget {k} exceeds test set ({len(scored)} rows)")
        for strategy, ordered in (("uncertainty", uncertain), ("random", random)):
            chosen = ordered.head(k)
            # Safe file for the teammate's LLM: only features and ML output.
            chosen[handoff_cols].to_csv(out_dir / f"review_{strategy}_k{k}.csv", index=False)
            missed = int(((chosen.true_label == 1) & (chosen.ml_prediction == 0)).sum())
            false_alarms = int(((chosen.true_label == 0) & (chosen.ml_prediction == 1)).sum())
            summary.append({
                "k": k, "strategy": strategy,
                "selected_false_negatives": missed,
                "selected_false_positives": false_alarms,
                "selected_ml_errors": missed + false_alarms,
                "selected_botnets": int(chosen.true_label.sum()),
                "mean_uncertainty": float(chosen.uncertainty.mean()),
            })
    pd.DataFrame(summary).to_csv(out_dir / "selection_analysis.csv", index=False)
    return summary


def save_chart(scored: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, title in ((0, "Normal"), (1, "Botnet")):
        values = scored.loc[scored.true_label == label, "ml_probability"]
        if not values.empty:
            ax.hist(values, bins=30, range=(0, 1), alpha=0.55, label=title)
    ax.set(xlabel="ML botnet probability", ylabel="Number of flows", title="Held-out scenario: predicted probabilities")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "test_probability_distribution.png", dpi=160)
    plt.close(fig)



def main() -> None:
    # Validate scenario split
    requested = (
        TRAIN_SCENARIOS
        + VALIDATION_SCENARIOS
        + TEST_SCENARIOS
    )

    if len(set(requested)) != len(requested):
        raise ValueError(
            "Train, validation and test scenarios must be disjoint"
        )

    available = find_scenarios(DATA_DIR)

    missing = sorted(set(requested) - available.keys())

    if missing:
        raise FileNotFoundError(
            f"Missing CTU-13 scenarios: {missing}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Scenario split:", {
        "train": TRAIN_SCENARIOS,
        "validation": VALIDATION_SCENARIOS,
        "test": TEST_SCENARIOS,
    }, flush=True)

    counts = []

    def load(ids):
        frames = []

        for scenario in ids:
            frame, info = read_scenario(
                available[scenario],
                scenario,
            )

            counts.append(info)
            print("Loaded:", info, flush=True)

            if frame.empty:
                raise ValueError(
                    f"Scenario {scenario} has no labeled flows"
                )

            frames.append(frame)

        return pd.concat(frames, ignore_index=True)

    # --------------------------------------------------
    # 1. Training
    # --------------------------------------------------

    train = load(TRAIN_SCENARIOS)

    if train["true_label"].nunique() != 2:
        raise ValueError(
            "Training data must contain both classes"
        )

    print("Training Random Forest...", flush=True)

    model = make_model()

    model.fit(
        train[FEATURES],
        train["true_label"],
    )

    # Save the trained model
    joblib.dump(
        model,
        RESULTS_DIR / "random_forest.joblib",
    )

    # --------------------------------------------------
    # 2. Validation
    # --------------------------------------------------

    validation = load(VALIDATION_SCENARIOS)

    validation_scored = score(model, validation)
    validation_metrics = evaluate(validation_scored)

    print(
        "Validation metrics:",
        json.dumps(validation_metrics, indent=2),
        flush=True,
    )

    validation_scored.to_csv(
        RESULTS_DIR / "validation_predictions.csv",
        index=False,
    )

    # --------------------------------------------------
    # 3. Independent test scenarios
    # --------------------------------------------------

    all_test_results = {}

    for scenario in TEST_SCENARIOS:

        print(f"\nEvaluating test scenario {scenario}...")

        # Separate output directory for each scenario
        scenario_dir = RESULTS_DIR / f"scenario_{scenario}"
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # Load only this test scenario
        test = load([scenario])

        # Apply the already trained model
        test_scored = score(model, test)

        # Evaluate classification performance
        test_metrics = evaluate(test_scored)

        print(
            f"Test metrics - Scenario {scenario}:",
            json.dumps(test_metrics, indent=2),
            flush=True,
        )

        # Save predictions with ground-truth labels
        test_scored.to_csv(
            scenario_dir / "predictions.csv",
            index=False,
        )

        # Generate top-K uncertain and random selections
        selection = select_for_review(
            test_scored,
            scenario_dir,
        )

        # Save probability distribution chart
        save_chart(
            test_scored,
            scenario_dir,
        )

        all_test_results[str(scenario)] = {
            "metrics": test_metrics,
            "selection": selection,
        }

    # --------------------------------------------------
    # 4. Save experiment summary
    # --------------------------------------------------

    pd.DataFrame(counts).to_csv(
        RESULTS_DIR / "dataset_counts.csv",
        index=False,
    )

    config = {
        "train_scenarios": TRAIN_SCENARIOS,
        "validation_scenarios": VALIDATION_SCENARIOS,
        "test_scenarios": TEST_SCENARIOS,
        "review_budgets": REVIEW_BUDGETS,
        "random_seed": SEED,
        "model": (
            "RandomForestClassifier("
            "n_estimators=100, "
            "max_depth=15, "
            "min_samples_leaf=2)"
        ),
        "threshold": 0.5,
        "features": FEATURES,
    }

    summary = {
        "config": config,
        "validation": validation_metrics,
        "tests": all_test_results,
        "methodological_notes": [
            "Training and test scenarios are disjoint.",
            "The model is trained only on training scenarios.",
            "The same trained model is evaluated on both test scenarios.",
            "No model tuning is performed using test labels.",
            "Background and ambiguous flows are excluded.",
            "LLM review files exclude ground-truth labels.",
            "Uncertainty is based on proximity to probability 0.5.",
        ],
    }

    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(
        f"\nFinished. Results saved under {RESULTS_DIR.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
