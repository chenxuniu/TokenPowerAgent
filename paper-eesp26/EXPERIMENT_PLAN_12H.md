# TokenPowerSandbox: Final 12-Hour Experiment Plan

## Submission Baseline

The paper already has a complete sealed result: 3 anchor runs, 18 development
runs, and 24 blind holdout runs on one H100. The blind holdout reaches 6.23%
energy MAPE and 0.976 Spearman rank correlation. This result is the submission
fallback and must not be refit or overwritten.

## One Optional Confirmation Campaign

Run `qwen2.5-7b-h100-scope-confirmation-v3` with the unchanged v2 profile and
server. It contains nine previously unmeasured context/concurrency pairs and
three repeats per pair (27 runs):

- Input tokens: 384, 768, and 1536.
- Output tokens: 128.
- Concurrency: 1, 2, and 4.
- Model, image, serving controls, request regime, and 700-W cap unchanged.

The campaign is not another model-fitting split. Its pre-registered purpose is
to decide whether latency prediction should abstain below concurrency 4 while
retaining the already validated energy-screening claim. All points and repeats
must be reported regardless of outcome.

## Stop Rules

Stop new GPU work and keep the sealed v2 paper if any of these occurs:

1. The server configuration or pinned image differs from holdout-v2.
2. Frozen hashes cannot be verified before the first measurement.
3. More than two runs fail for infrastructure reasons after one clean retry.
4. The campaign is not complete by hour 6.

Do not start TP/PP, multi-node, H200/B200, a second model, or a new power-cap
sweep during this submission window. Each would require a new calibration,
baseline, and disjoint holdout to support a defensible claim.

## Time Budget

| Window | Work |
|---|---|
| 0:00--0:45 | Pull, verify the server contract, freeze predictions, and archive the freeze manifest. |
| 0:45--3:00 | Run 27 cyclically balanced measurements with checkpoint/resume enabled. |
| 3:00--4:00 | Seal raw artifacts and generate the validation report. |
| 4:00--5:00 | Evaluate the pre-registered all-workload energy endpoint and latency strata. |
| 5:00--7:00 | Add at most one compact secondary result to the paper if the campaign is valid. |
| 7:00--9:00 | Recompile, inspect every page, and rerun submission checks. |
| 9:00--12:00 | Preserve a three-hour buffer for failures, author review, and upload QA. |

## Interpretation

- If all-workload energy MAPE is at most 15%, report it as an independent
  confirmation of energy screening, not as a replacement for holdout-v2.
- If concurrency-4 TTFT MAPE is at most 20% and concurrency-1/2 TTFT MAPE is
  above 20%, adopt a future latency abstention rule below concurrency 4.
- If concurrency 4 also fails, broaden abstention and leave the current paper
  result unchanged.
- Never use this confirmation set to refit the model reported in the paper.
