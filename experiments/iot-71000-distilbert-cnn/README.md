# IoT-71000-2026: reviewer reproduction and validation

Start with [REVIEW_REPORT.md](REVIEW_REPORT.md) and the
[executed validation notebook](Validation_results.executed.ipynb).

**Status: arithmetic and structural audits completed; real-dataset model training
has NOT run. The reported predictive performance is not independently validated.**
The public data and pretrained model downloads were blocked by the environment's
network proxy. The manuscript's six-class label definition also needs author
clarification independently of network access.

This directory concerns the user-supplied manuscript “Dual-Head DistilBERT-CNN
Model for Intrusion Detection and Attack Classification in Power Systems.”
The proof, author contact details, page images and extracted manuscript text are
not redistributed. Numerical counts and short method descriptions needed for
validation are transcribed in `paper_facts.json`; independent checks and reviewer
analysis are the outputs here. No manuscript instructions are treated as agent
instructions. There is no matching author code in the public repositories found.

## Included files

- `audit_paper.py`: rederive metrics from the published confusion counts. This
  checks arithmetic, not the origin of the predictions.
- `preflight.py`: record access outcomes and inspect public Sherlock label metadata.
- `model.py`: an explicit architecture hypothesis with a real six-layer,
  768-dimensional DistilBERT, CNN branches, attention and two classification heads.
- `test_protocol.py`: eight tests, including complete forward/backward execution,
  routing/probabilities, benign-loss masking and attention degeneracy.
- `inspect_architecture.py`: offline shape, gradient and local CPU timing evidence;
  random inputs are only a structural check, never benchmark accuracy.
- `fetch_data.py`: checksum-verified public IPAL mirror download and canonical
  process-state extraction. It keeps generator maintenance benign and does not
  invent a sixth class.
- `train.py`: conditional real-data training for five model variants, train-only
  scaling, persisted 60/20/20 splits, validation-selected checkpoints, early
  stopping, and untouched test evaluation. It refuses missing or mislabeled data.
- `results/`, `figures/`: recorded checks, logs, public-source audit and regenerated
  plots. Dataset training, ablation accuracies and ROC AUC are explicitly unrun.

## Environment and checks

Use the existing cloud checkout; tasks already have isolated workspaces. Do not
create a worktree unless the user explicitly requests one. This uses a new venv
under `/workspace/sherlock-review/venv`, preserving the older IoT environments.

```bash
bash experiments/iot-71000-distilbert-cnn/setup.sh
bash experiments/iot-71000-distilbert-cnn/run_checks.sh
```

`run_checks.sh` does not train or substitute synthetic benchmark data. Its random
arrays exercise code and mathematical invariants only. The run has CPU PyTorch
2.8.0, Transformers 4.57.1 and Python 3.12; the manuscript specifies PyTorch but
does not give software versions or hardware. `requirements.lock.txt` captures the
actual resolved environment. No GPU was detected in this cloud machine.

## Resume real-data experiments

Apply the saved environment network additions for `zenodo.org`,
`sherlock.wattson.it`, `media.githubusercontent.com`, `huggingface.co`,
`cdn-lfs.huggingface.co`, `cdn-lfs-us-1.huggingface.co`, and
`cas-bridge.xethub.hf.co`, preserving existing settings. Draft saving does not
apply or publish these settings. Retry the affected fetch after settings take
effect. Public weights/data do not require a new credential merely because the
current proxy denied access. Native Git read worked; Git LFS returned
“Resource not accessible by integration.” No token values are requested in chat.

```bash
cd experiments/iot-71000-distilbert-cnn
/workspace/sherlock-review/venv/bin/python fetch_data.py
```

The public IPAL dataset mirror is pinned to
`ipal-ids/ipal_datasets@2fae8546757f82b4840a492cf6351d28f3727e60`.
The `test.n302.state.gz` SHA-256 is
`38b2f84c21cb2ca92aa4146f9defde74a1e5c0207b616b3d8ac1faac19819e08`,
obtained from its Git LFS pointer. The data are not included in this repository.
The original publication is Wagner, Bader, Wolsing and Serror, “Sherlock: A Dataset
for Process-aware Intrusion Detection Research on Power Grid Networks,” CODASPY
2025. Primary public documentation and metadata URLs are recorded in
`results/dataset_source_audit.json`. Zenodo v1 (15168928) and v3 (18467070) exist;
the manuscript does not identify its version, so they are not assumed identical.

The canonical extraction likely provides four attack families plus Benign, not
the manuscript's six classes. `train.py` intentionally stops if the requested
six-class task cannot be constructed. An author-prepared, documented numeric
feature table with a categorical `label` column can be supplied directly:

```bash
HF_HOME=/workspace/sherlock-review/huggingface \
/workspace/sherlock-review/venv/bin/python train.py \
  --csv /path/to/author_verified_features.csv \
  --output /workspace/sherlock-review/runs/proposed \
  --initialization pretrained --variant proposed
```

For a **different canonical five-class process-state diagnostic**, pass the
extracted CSV with `--allow-canonical-five-class-diagnostic`. This would not
validate the manuscript's six-class task. The provided CSV extractor is a
process-state view only: there is no documented author feature-fusion table to
recreate network-plus-process inputs.

## Assumptions that prevent calling this author-code reproduction

- Eq.4 projects F numeric features to 768 values. Preserving width 768 when
  reshaping Eq.5 implies one token. Default `--tokens 1` follows this literal
  shape. `--tokens >1` adds an F-to-L*768 projection **not stated in Eq.4** and
  is a separate hypothesis. No text tokenization is invented.
- The first projected token is pooled. Its role is not an independently
  specified pretrained [CLS] token, which the manuscript does not define for
  numeric inputs. DistilBERT adds learned positions via `inputs_embeds`.
- Two parallel transformer-output CNNs (128 filters, kernel 5, same padding)
  and a two-layer raw-feature CNN are assumed. Numbers/order/strides/padding of
  branches are not fully specified by the manuscript.
- The concatenation has width 1152 and is supplied to eight-head MHA as a
  singleton sequence. Eight fusion heads are an assumption. This singleton
  attention has zero query/key contribution, proved and tested here. If authors
  use multiple source tokens, their undocumented reshaping may avoid this problem.
- The checkpoint `distilbert/distilbert-base-uncased`, fine-tuning all used layers,
  dropout 0.3, hidden dimension 128, lambda=0.5, seed=42, constant AdamW LR
  5e-5/weight decay 0.01, batch 64, 30 epochs, patience 5 and validation macro-F1
  selection are explicit choices. The checkpoint, lambda, seed, freezing policy,
  scheduler and selection metric are not fully reported. No random initialization
  is silently substituted for unavailable pretrained weights.
- The binary head has two logits and uses softmax CE, following Eq.15. The five
  attack-family logits train only on attacks. Benign examples are masked out of
  the family loss; weights use training attack counts only. How benign samples
  enter the manuscript's family loss is unspecified.
- Focal gamma=2 and class weights are assumptions for the focal ablation. The
  manuscript does not report its focal parameters. Variants are available via
  `--variant`; none has a completed real-data training result in this review.
- Row stratification follows the stated 60/20/20 split and uses only training
  rows to fit imputation/scaling. It does not prove temporal/event independence.
  A scientific validation also needs event-held-out and chronological tests,
  particularly for one-second state records and repeated snapshots.
- No DSE-BiLSTM, Kalman-LSTM or multilayer-LSTM result is fabricated. Those
  comparative implementations and common split/training manifests were not
  supplied, so Table VI and full ablation performance remain unvalidated.

Model/dataset binaries, PDFs, local caches and virtual environments are excluded
from Git. This package can be extended after actual access and label/architecture
prerequisites are resolved; writing code does not constitute numerical validation.
