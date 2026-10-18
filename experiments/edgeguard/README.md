# EdgeGuard-IoT independent reproduction

Start with [the experiment report](REPORT.md) or
[the executed results notebook](EdgeGuard_results.executed.ipynb).
The `figures/` directory contains plots; `results/` contains the measured scores,
class reports, diagnostics and sampling manifest. `artifacts/` preserves frozen
split indices, saved predictions, and SHAP arrays for the completed runs.
The split indices refer to the deduplicated data in the documented source order.
The group-split `groups` vector follows concatenated original train/test order;
its `train_dedup_indices` and `test_dedup_indices` directly index the deduplicated rows.

Raw datasets, installed packages, fitted preprocessing/model binaries, and the
user-supplied paper are excluded from Git. Dataset download scripts and the
dependency lock reproduce the inputs and software; run scripts regenerate models.
The checked-in artifacts retain the current run's evidence independently of that
regeneration. No original repository notebook was modified.

This implements and evaluates the primary UNSW-NB15 experiment described in the
user-provided paper by Fathe Jeribi, DOI `10.1016/j.comnet.2026.112813`.
It is an independent implementation, not the author's released code, and numerical
agreement with the paper must be measured rather than assumed. The PDF is not
redistributed here.

## Data and protocol

The public 82,332-row UNSW-NB15 testing partition is used. Keep the nine attack
families and Normal, retain the 39 numeric predictors, exclude `id`, the three
categorical predictors (`proto`, `service`, `state`), and the binary `label`.
Deduplicating numeric predictors **plus `attack_cat`** yields exactly 55,645 rows
and every class count in the paper's Table 4. Deduplicating all non-ID columns
instead gives 55,945 rows; this distinction is essential.

Public mirror:
`https://github.com/tylerhoward15/sklearn-RandomForestsClassifier-UNSW-NB15`
at commit `a011b64e3d1efad4a4bdfbf8ecc4d5e7da21736e`.
File: `UNSW_NB15_testing-set.csv`.
SHA-256: `7ec02e7e44d72bd265716b33fda0c7f2188b658e3a4ae1aa2c4b306134cd818c`.
The university source was inaccessible through the environment's current network
policy. Mirror provenance and exact class-count agreement do not establish an
independent checksum match against the university's original. Cite Moustafa and
Slay (2015), the original UNSW-NB15 dataset authors, when using this data.

Use a stratified 70/30 split with seed 42, persisting its row indices. Median
imputation, nonconstant-column detection, scaling, mutual information and RFE
are fit using training rows exclusively. MI retains 30 features; RFE retains 20.
Both stages use HistGradientBoostingClassifier (200 iterations, learning rate
0.12, 31 leaves, L2 1.0). Stage 1 learns benign/attack; only suspected attacks
reach the nine-family stage 2. Stage 1 uses weights, not SMOTE. Stage 2 uses
SMOTE k=5, seed 42, raising each smaller family to 4,000 rows. Class weights are
inverse frequencies after resampling. Augmentation appends one Gaussian-corrupted
and one dropout-corrupted view to each clean example, independently at each stage.
Quantization is withheld from training.

Evaluate clean accuracy, macro/weighted F1, MCC, per-class metrics, binary stage-1
metrics, and 12 corruptions (noise/dropout 0.1/0.2/0.3/0.5; quantization 16/8/4/2).
All models see the same seeded test corruptions. Quantization ranges are learned
from training data; dropout zeros standardized features. Record uncompressed
joblib model bytes and median throughput of five warmed full-test-batch passes.
Pin the process to one available CPU and limit BLAS/OpenMP threads to one.
These are batch-amortized prediction timings on this cloud CPU, not per-packet
latencies or measurements on physical gateway hardware. Preprocessing and flow
feature extraction are excluded from the prediction timing and model-size figures.

## Explicit reproduction assumptions and limitations

- No author code, exact split file, exact dependency lock, or augmentation RNG
  sequence was supplied. Seed 42 for split and learners is a declared assumption.
- The paper does not specify the RFE estimator. We choose 100-tree RandomForest,
  depth 18, RFE step 2. The default run learns its features independently rather
  than silently forcing the published feature list. `--features paper` provides a
  separate sensitivity run using Table 10's 20 features, with obvious typographical
  spellings mapped to the CSV's actual names. Neither run tunes on held-out scores.
- The paper does not specify augmentation multiplicity or intensity sampling
  granularity. We choose three views and one uniform intensity in [0,0.3] per
  corruption view, with explicit seeds. Quantization clipping and using training
  ranges are declared choices. They can substantially change robustness scores.
- The paper specifies no imputation rule, weight computation before/after SMOTE,
  early-stopping policy, or serialized format. We use median imputation,
  post-SMOTE weights, exactly 200 HistGBM iterations (`early_stopping=False`),
  and joblib without compression. Baseline HistGBM uses default L2=0 per Table 9.
- Section 3.7 explicitly says no SMOTE in Stage 1; a later paragraph contradicts
  this by saying both stages use SMOTE. We follow the detailed stage-specific
  description and Table 7. Flat multiclass baselines use capped SMOTE and inverse
  weights; standalone binary baselines use inverse weights only. We pass weights
  once, avoiding multiplication by estimator `class_weight` a second time.
  kNN has no sample-weight fit API and uses the same SMOTE-expanded data without
  weights. This unavoidable exception is recorded in metrics.
- The paper names imbalanced-learn 0.14 without a patch version. Version 0.14.0
  fails importing with scikit-learn 1.8 (`_is_pandas_df` missing). Use 0.14.2,
  which imports successfully. Main stated versions (Python 3.12, sklearn 1.8,
  XGBoost 3.3, LightGBM 4.7, SHAP 0.52) are retained. Full resolved versions
  are recorded in `requirements.lock.txt` and runtime metadata.
- Rare-class mean recall is defined here over the three rarest training classes
  (Worms, Backdoor, Shellcode). The paper is ambiguous about this aggregate.
- The paper's claim of 841:1 imbalance conflicts with its own Table 4:
  34,204 / 44 is about 777:1. We preserve the actual rows without adjusting them.
- Corruption sweeps exclude kNN, as in the paper. A convergence warning does not
  become a pass; every warning is retained with that model's metrics. Other errors
  stop the run and must be investigated.
- Friedman/Nemenyi uses ten per-class F1 values as blocks, following the paper.
  These correlated classes from a single split are not independent experimental
  repeats. Failure to find a significant pairwise difference is not evidence of
  statistical equivalence or non-inferiority.
- SHAP explains stages 1 and 2 separately on a seeded 256-flow held-out sample.
  These are not exact SHAP explanations of the composite routed decision.
- Group-split sensitivity clusters within each class using radius 0.01 on the
  original standardized selected-feature space. Whole groups are assigned to a
  roughly 70/30 split by class proportions only; the group split gets its own
  train-only scaler. Its sizes may differ from the paper's fixed sizes. Group
  construction uses the initial training scaler and fixed selected features, as
  in the paper's exploratory protocol; it is not a fully nested independent study.
- The completed artifact audit found 394 shared numeric-predictor hashes across
  the primary UNSW train/test split, despite removing duplicate predictor+target
  rows. Identical numeric predictors with different targets are retained by that
  deduplication rule. Dropping categorical features can make distinct raw flows
  numerically indistinguishable. The within-class group split cannot guarantee
  separation of cross-label duplicates. Matching the paper's counts is therefore
  not evidence of a predictor-disjoint or leakage-free evaluation. See REPORT.md
  and audit.json; a stricter future study should group across labels before splitting.
- This host has about 33 GiB RAM, rather than the paper's 4 GiB. Single-core
  inference is enforced, but no 4-GiB memory ceiling or physical edge gateway
  equivalence is claimed.
- The Edge-IIoTset public mirror matches 157,800 raw and 152,245 deduplicated rows.
  Across the whole dataset, 34 numeric predictors vary; one varies only outside
  this run's training split, leaving 33 predictors under train-only constant
  detection. This is intentionally stricter than detecting constants globally.
- The public Bot-IoT four-file 5% release supports the paper's one-million-DoS-flow
  transfer test. Its full 40-original-scenario-file retraining sample was not
  located. We additionally evaluate a clearly different four-file sample:
  all Normal/Theft, plus up to 3,000 random rows per other category per file,
  seed 42, using fifteen flow-statistic features. It is NOT a reproduction of
  the 136,039-row Bot-IoT same-dataset result. The complete sampling manifest
  and hashes are saved separately. In particular it contains fewer benign flows.
- A public repository named `raaddi/edgeguard-iot` was found but inspected as an
  unrelated SmartHome thesis/simulator project; it is not used as author code.
  No matching public author implementation was found in the searches performed.

## Run

Use the existing isolated cloud checkout. Do not create an additional Git worktree.
The experiment uses a separate venv, leaving the older notebook environment intact.

```bash
bash experiments/edgeguard/setup.sh
bash experiments/edgeguard/run.sh
```

Run the additional public-data experiments with
`bash experiments/edgeguard/run_additional.sh` after the primary UNSW run. The
script downloads CSV/ZIP files from commit-pinned public mirrors, checks their
SHA-256 values, retrains on Edge-IIoTset, builds/trains the alternative Bot-IoT
sample, and scores the million-flow cross-dataset transfer. It does not execute
any code from the data-mirror repositories. Cite Ferrag et al. (2022) for
Edge-IIoTset and Koroniotis et al. (2019) for Bot-IoT in resulting research.

The source URLs and hashes are in `fetch_additional_data.py`. Matching counts
do not replace an independently obtained original-source checksum. The university,
IEEE DataPort, Kaggle and UNSW SharePoint destinations were blocked by this
environment's network proxy; the successful runs use public GitHub mirrors.

By default results and model binaries are retained outside the Git checkout under
`/workspace/edgeguard/results/hybrid`. Data and models are not uploaded to GitHub.
`summary.csv`, `corruptions.csv`, `binary_baselines.csv`, class reports, predictions,
split indices, preprocessing objects, model files, runtime metadata, and SHAP values
are saved incrementally. `run.log` records the run's progress. The command's exit
status and final `EXPERIMENT COMPLETE` marker determine completion, not the
presence of partial result files.

Generate the portable report after all runs complete:

```bash
cd experiments/edgeguard
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /workspace/edgeguard/venv/bin/python diagnose_histgb.py
/workspace/edgeguard/venv/bin/python audit_results.py
IPYTHONDIR=/workspace/edgeguard/ipython /workspace/edgeguard/venv/bin/python -m ipykernel install --prefix /workspace/edgeguard/venv --name edgeguard
MPLCONFIGDIR=/workspace/edgeguard/matplotlib XDG_CACHE_HOME=/workspace/edgeguard/cache JUPYTER_RUNTIME_DIR=/workspace/edgeguard/jupyter-runtime IPYTHONDIR=/workspace/edgeguard/ipython /workspace/edgeguard/venv/bin/python summarize.py
```

The published-feature sensitivity is separately stored under
`/workspace/edgeguard/results/paper_features`; it uses the same primary command
with `--features paper --models EdgeGuard`, without `--extended`. The original
run's results remain stored even if you do not rerun this optional sensitivity.
Reports do not reproduce every paper figure: the feature-count tradeoff curve,
full-feature accuracy sweep, and per-class ROC/PR plot suite were not run.

The small `test_protocol.py` suite verifies routing, corruption, augmentation, and
SMOTE behavior. It is a code check and is separate from real-data evaluation.
