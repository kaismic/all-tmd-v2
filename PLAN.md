# ALL-TMD v2 implementation plan

## Goal

Measure cross-regional calibration on one frozen Sydney LOPO evaluation set,
using fixed XGBoost hyperparameters selected without Sydney data.

## Implemented protocol

- Frozen 166-session Sydney snapshot; bus/car/train and the two-hour rule leave
  155 sessions before window-quality filtering.
- Seven participant-independent outer folds; a held-out participant never
  contributes calibration rows.
- Three seeds and 33 parents: three NOR-only baselines plus five nonzero Sydney
  fractions for Sydney-only and pooled conditions.
- Deterministic per-class whole-session prefixes, nested by fraction and paired
  between Sydney-containing conditions.
- 60-second windows, 30-second steps, the report-selected 11 features, collector
  sampling minima of 30/4/2 Hz, and a 500 ms maximum gap.
- Class-balanced weights without a domain multiplier or duration balancing.
- NOR-only selection with 45 Optuna XGBoost trials, five grouped folds, seed 42,
  and pooled out-of-fold macro F1.

## Implemented architecture

- Content-addressed ingestion, feature, split, and execution identities.
- Temporary builds, atomic promotion, `_SUCCESS`, and immutable JSON manifests
  for deterministic stages.
- Run-ID result directories and nested MLflow parent/fold runs; no training cache.
- Isolated AWS sweep stores plus an idempotent supported-API importer into one
  canonical local MLflow experiment.
- Report generation with paired hierarchical-bootstrap intervals and CSV, PNG,
  PDF, LaTeX, and JSON outputs.
- Exact dependency lock and matching local/EC2 CLI contracts.

## Operational sequence

1. Verify the frozen input and prepare ingestion/features.
2. Generate and review the immutable LOPO manifest.
3. Run NOR-only tuning and promote one immutable model lock.
4. Retain the exact contracts and run all 33 configurations.
5. Download/import isolated AWS stores where needed.
6. Generate and validate the report artifacts.
7. Insert real outputs into `reports/report-6-revised.tex` in a later change.

ONNX/deployment work and report insertion remain out of scope until real study
results have been validated.
