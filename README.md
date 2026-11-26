# When Should an LLM Review Network Alerts?

LOG6309E Intelligent DevOps · Federico Capitani and Isabela Gregio

This project tests whether Random Forest uncertainty can identify missed botnet flows in CTU-13 and whether a second classification from Gemma improves detection for the tokens it consumes.

[Final presentation](docs/final_project_presentation.pdf) · [Results summary](docs/CTU13_Final_Results.pdf) · [Full evaluation](results/final_evaluation/evaluation_report.md)

## Final results

| Held-out capture | Random Forest recall | Missed botnets selected at K=200, uncertainty vs. random | Net corrections with uncertainty review at K=50 / 100 / 200 |
| --- | ---: | ---: | ---: |
| Scenario 11, Rbot family seen in training | 96.95% | 9 vs. 4 | +7 / +6 / +7 |
| Scenario 12, NsisAy family unseen in training | 33.40% | 63 vs. 26 | −19 / −24 / −35 |

**RQ1.** Uncertainty selected more missed botnets than the matched random sample in both captures at all three budgets. In scenario 12, the 63 misses it selected at K=200 represented 4.7% of the 1,332 misses in that capture.

**RQ2.** Zero-shot Gemma review improved net error counts in scenario 11 and worsened them in scenario 12 at all three uncertainty budgets. In scenario 11, K=50 and K=200 each removed seven net errors, using 7,578 and 30,045 tokens respectively. In scenario 12 at K=200, Gemma corrected 40 errors and introduced 75. Botnet recall fell from 33.40% to 30.10%.

The two captures were chosen after exploratory work on scenario 10. The results describe one LLM, one prompt, one random seed, and two held-out captures. The Random Forest scores are uncalibrated. The capture and family effects cannot be isolated with this design.

## Method

- **Data.** CTU-13 bidirectional flows labeled `From-Botnet` or `From-Normal`. Background and ambiguous flows are excluded. The nine features are duration, source and destination type of service, packet count, total and source byte counts, protocol, direction, and state.
- **Split.** Train on scenarios 1, 2, 4, 5, 6, 7, and 8. Validate on 3 and 9. Evaluate separately on 11 and 12.
- **Baseline.** A 100-tree Random Forest with maximum depth 15, minimum leaf size 2, and a classification threshold of 0.5. The pipeline imputes missing numeric and categorical values and one-hot encodes categorical features.
- **Selection.** Rank flows by how close their Random Forest score is to 0.5 or use a random ordering with seed 42. Each policy uses the first 50, 100, or 200 flows from its ordering. This gives nested budgets and a matched review count.
- **Gemma review.** We used Google's hosted `gemma-4-31b-it` in a zero-shot setup with the fixed [prompt](src/flow_review_draft.txt). Each request contains the instruction and one flow's nine features. Flow identity, scenario, ground-truth label, and Random Forest prediction stay outside the payload. Gemma's valid reply replaces the prediction for the selected flow. CTU-13 exposure during Google's pretraining is unknown.
- **Scoring.** Compare both sets of predictions against the true labels across the entire capture. Count corrections and newly introduced errors, and measure recall, false-positive rate, and F1. Net corrections per 1,000 tokens is `(corrected − introduced) × 1,000 / (input + output tokens)`.

## Reproduce the final evaluation offline

From the repository root, create a Python environment and install the dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The repository includes both test captures' scored flows, all review selections, and 779 completed reviews. Recompute the results without an API key or a dataset download.

```bash
python -m src.evaluation \
  --responses results/final_evaluation/completed_reviews.jsonl \
  --output-dir results/reproduced_evaluation
```

The evaluator checks coverage and configuration consistency before scoring and writes the aggregate CSV, the per-flow CSV, the token summary, and the two F1 plots. The published evaluation used all 779 selected flows, corresponding to 753 distinct completed request payloads.

| Path | Contents |
| --- | --- |
| [`results/scenario_11/`](results/scenario_11/) and [`results/scenario_12/`](results/scenario_12/) | Baseline predictions, ground-truth labels, exact review selections, and uncertainty analysis |
| [`selection_coverage.csv`](results/final_evaluation/selection_coverage.csv) | RQ1 counts and share of baseline false negatives captured per selection |
| [`evaluation_summary.csv`](results/final_evaluation/evaluation_summary.csv) | Final metrics for the baseline and every scenario, strategy, and budget |
| [`reviewed_flows_evaluation.csv`](results/final_evaluation/reviewed_flows_evaluation.csv) | Per-flow decisions and token counts for each comparison |
| [`error_transitions.csv`](results/final_evaluation/error_transitions.csv) | Corrected and introduced false negatives and false positives |
| [`completed_reviews.jsonl`](results/final_evaluation/completed_reviews.jsonl) | One completed response per evaluated flow, with prompt fingerprint, configuration, and usage |
| [`api_usage.json`](results/final_evaluation/api_usage.json) and [`run_audit.json`](results/final_evaluation/run_audit.json) | Usage across distinct completed payloads and original API run counts, including retries |

Token totals in the comparison table are specific to each policy and budget. The same flow can be included in several budgets. A reused response keeps its original token counts. Output usage includes any thinking tokens returned in the provider's total.

Run the offline request preparation, caching, and API runner tests with:

```bash
python -m unittest src.test_llm_review src.test_gemma_review -v
```

## Rerun training and inference

The baseline training and uncertainty analysis use the public CTU-13 Parquet files. Training writes model artifacts, validation predictions, and test selections under `results/`.

```bash
python download_data.py
python -m src.experiment
python analyze_uncertainty.py
```

Prepare one request per selected test flow. Choose a new output filename when preparing another run.

```bash
python -m src.llm_review \
  --prompt src/flow_review_draft.txt \
  --output results/llm_requests.jsonl \
  results/scenario_*/review_*_k*.csv
```

Preview the same model and generation settings used in the final evaluation. This preview is offline.

```bash
python -m src.gemma_review \
  --requests results/llm_requests.jsonl \
  --output results/gemma_responses.jsonl \
  --temperature 0 \
  --max-output-tokens 32 \
  --thinking-level minimal \
  --requests-per-minute 6
```

For a live rerun, add your own `GEMINI_API_KEY` to the ignored `.env` file or export it in your shell. Add `--execute --max-new-requests 1` to check the first call. Use a larger batch once you confirm your API project can support it. Resume unfinished batches with the same output file and configuration. Completed responses are reused, and HTTP 503 retries count toward the batch limit. For evaluation of a completed rerun, pass the new response file to `src.evaluation` and use a separate output directory.

## Sources

- [CTU-13 dataset, Stratosphere Laboratory](https://www.stratosphereips.org/datasets-ctu13)
- [CTU-13 Parquet files](https://www.kaggle.com/datasets/dhoogla/ctu13)
- [Gemma 4 31B model card](https://huggingface.co/google/gemma-4-31B-it)
