# Final Evaluation — RF vs. Selective LLM Review

The same held-out flows are evaluated for ML-only, uncertainty review, and random review.
Only selected flows with valid LLM labels replace the RF prediction; invalid replies retain the RF prediction.
All values below are based on completed real LLM responses, not estimates.

## Scenario 11

| Strategy | K | Precision | Recall | F1 | FPR | Corrected | Introduced | Invalid | Tokens | Net corrections / 1k tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ml_only | 0 | 0.9948 | 0.9695 | 0.9820 | 0.0027 | 0 | 0 | 0 | 0 | — |
| uncertainty | 50 | 0.9935 | 0.9796 | 0.9865 | 0.0033 | 9 | 2 | 0 | 7578 | 0.9237 |
| uncertainty | 100 | 0.9935 | 0.9784 | 0.9859 | 0.0033 | 9 | 3 | 0 | 15020 | 0.3995 |
| uncertainty | 200 | 0.9948 | 0.9784 | 0.9865 | 0.0027 | 10 | 3 | 0 | 30045 | 0.2330 |
| random | 50 | 0.9909 | 0.9707 | 0.9807 | 0.0046 | 1 | 3 | 0 | 7657 | -0.2612 |
| random | 100 | 0.9884 | 0.9720 | 0.9801 | 0.0060 | 2 | 5 | 0 | 15297 | -0.1961 |
| random | 200 | 0.9845 | 0.9720 | 0.9782 | 0.0080 | 2 | 8 | 0 | 30503 | -0.1967 |

## Scenario 12

| Strategy | K | Precision | Recall | F1 | FPR | Corrected | Introduced | Invalid | Tokens | Net corrections / 1k tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ml_only | 0 | 0.8077 | 0.3340 | 0.4726 | 0.0231 | 0 | 0 | 0 | 0 | — |
| uncertainty | 50 | 0.8070 | 0.3220 | 0.4603 | 0.0224 | 9 | 28 | 0 | 7712 | -2.4637 |
| uncertainty | 100 | 0.8137 | 0.3145 | 0.4537 | 0.0209 | 21 | 45 | 0 | 15389 | -1.5596 |
| uncertainty | 200 | 0.8247 | 0.3010 | 0.4410 | 0.0186 | 40 | 75 | 0 | 30646 | -1.1421 |
| random | 50 | 0.8007 | 0.3315 | 0.4689 | 0.0240 | 0 | 11 | 0 | 7749 | -1.4195 |
| random | 100 | 0.7973 | 0.3285 | 0.4653 | 0.0243 | 2 | 21 | 0 | 15467 | -1.2284 |
| random | 200 | 0.7895 | 0.3225 | 0.4579 | 0.0250 | 5 | 41 | 0 | 30943 | -1.1634 |

## Token accounting

Per-strategy/K tokens count the original inference cost of each selected flow, even when the response was reused from the cache. These are comparable hypothetical strategy costs, **not additive across strategies**.
`api_usage.json` separately reports recorded non-cached calls among selected flows; it does not include unrelated validation/pilot calls.

## Interpretation boundaries

- RF probabilities are uncalibrated.
- Background and ambiguously labeled traffic are excluded.
- The test scenarios were selected after exploratory work on scenario 10; the evaluation should be described accordingly.
- Only actual LLM calls establish correction ability; error concentration alone does not.
