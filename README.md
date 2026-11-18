# CTU-13 — ML Baseline & Uncertainty Analysis

**Research project:** When Should an LLM Review Network Alerts? — LOG6309E

This repository implements **Phase 1** of the project: training a Random Forest on CTU-13 network traffic, evaluating its performance on unseen scenarios, and selecting uncertain flows for subsequent LLM review.

## Methodology

- **Dataset:** CTU-13 (Parquet format), using 9 network-flow features.
- **Labels:** Botnet (1) and Normal (0). Background and ambiguous flows are excluded.
- **Training:** Scenarios 1, 2, 4, 5, 6, 7, 8.
- **Validation:** Scenarios 3, 9.
- **Testing:** Scenario 11 (Rbot, known family) and Scenario 12 (NsisAy, unseen family), evaluated separately.
- **Model:** Random Forest (100 trees, max depth 15, minimum leaf size 2), with numeric imputation and categorical encoding.
- **Uncertainty:** Flows are ranked by proximity of the predicted botnet probability to 0.5.
- **Review budgets:** K = 50, 100, 200, compared against random selection with the same budget.

The model is trained on all explicitly labeled training flows. No test scenarios are used for model fitting. Phase 1 does not perform LLM inference.

## Setup & Execution

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset if it is not already available in `data/raw/`:

```bash
python download_data.py
```

Run the experiment:

```bash
python -m src.experiment
```

Run the uncertainty analysis:

```bash
python analyze_uncertainty.py
```

## Results

The experiment saves the trained model, evaluation metrics, predictions, and selected flows under `results/`.

| File | Description |
|---|---|
| `summary.json` | Model configuration and evaluation metrics |
| `random_forest.joblib` | Trained Random Forest pipeline |
| `scenario_11/predictions.csv` | Predictions and true labels for Scenario 11 |
| `scenario_12/predictions.csv` | Predictions and true labels for Scenario 12 |
| `scenario_*/review_uncertainty_k*.csv` | Top-K uncertain flows for LLM review |
| `scenario_*/review_random_k*.csv` | Randomly selected flows for comparison |
| `scenario_*/uncertainty_analysis/` | Error coverage, probability statistics, and plots |

### Baseline Performance

| Metric | Scenario 11 | Scenario 12 |
|---|---:|---:|
| Precision | 99.48% | 80.77% |
| Recall | 96.95% | 33.40% |
| F1-score | 98.20% | 47.26% |
| ROC-AUC | 0.9869 | 0.9195 |

These results illustrate different generalization conditions. The model performs well on a known botnet family but struggles to identify attacks from an unseen family.

## Next Steps — Phase 2: LLM Review

1. **Select an LLM:** Use one open-source model or a free inference API, keeping the same model and prompt across experiments.
2. **Implement LLM inference:** Read the `review_uncertainty_k*.csv` and `review_random_k*.csv` files for both test scenarios. Send only the nine network-flow features to the LLM, requesting a binary classification (Normal/Botnet).
3. **Track inference costs:** Save LLM predictions, input/output token counts, and response times. Cache repeated flows to avoid unnecessary calls.
4. **Evaluate the results:** Replace the Random Forest predictions only for reviewed flows, joining results by `flow_id` with the corresponding `predictions.csv`.
5. **Compare strategies:** Evaluate ML only, ML + uncertainty-based LLM review, and ML + random LLM review using identical test sets and review budgets.
6. **Report findings:** Compare F1, recall, false-positive rate, errors corrected/introduced, and improvement per 1,000 tokens.

**Important:** Never include ground-truth labels in LLM prompts. Keep the trained Random Forest and experimental configuration fixed during Phase 2.

## Dataset References

- [CTU-13 — Stratosphere Laboratory](https://www.stratosphereips.org/datasets-ctu13)
- [CTU-13 Parquet — Kaggle](https://www.kaggle.com/datasets/dhoogla/ctu13)
