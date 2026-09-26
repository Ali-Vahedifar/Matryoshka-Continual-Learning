# Matryoshka Continual Learning (MCL)

Code for the MCL paper: the method, every continual-learning baseline it is
compared against, the GTEP evaluation protocol, the CL metrics and the cost
measurements. No checkpoints or results are included. Everything here is
source code you can run.

## What is included

| Group | Methods | Folder |
|---|---|---|
| Bounds | SGD (lower), Joint training (upper) | `SGD/`, `Joint/` + `joint_audited.py` |
| Regularisation | EWC, SI, LwF | `EWC/`, `SI/`, `LwF/` |
| Sparse / architecture | WSN (Task-IL only), PEC (Class-IL only), SpaceNet, NISPA | `WSN/`, `PEC/`, `SpaceNet/`, `NISPA/` |
| CL + unlearning (†) | UniCLUN | `UniCLUN/` |
| Shapley valuation | SNV, the SNV-A version used in the paper | `SNV/snv_adaptive.py` (on top of `SNV/snv_core.py`) |
| **Ours** | **MCL** | `MCL/mcl.py` |

WSN needs the task identity at test time, so it is evaluated only in Task-IL.
PEC is defined only for Class-IL. † marks methods designed jointly for
continual learning and machine unlearning.

## Layout

```
audited_gtep.py      GTEP protocol: splits, search spaces, one run, cost ledger hooks
snv_adaptive_run.py  runs SNV-A through the unchanged GTEP worker
campaign/
  run_campaign.py    full campaign: 30 configs x 3 seeds tuning, 3 clean D_E runs per winner
  build_report.py    metrics / cost / hyperparameter tables (CSV, XLSX, Markdown, LaTeX, HTML)
  snv_adaptive_compare.py
metrics.py           ACC, BWT, FWT, PS (+ P, S, AF)
audit_cost.py        cost ledger: GPU-hours, peak memory, parameters, GFLOPs, latency, energy
cost.py              lightweight cost tracker used by train.py
train.py             standalone single-method trainer (any dataset / scenario)
baselines.py         method registry
cl_base.py, models.py, datasets.py, training_policy.py, utils.py, inrun.py
<Method>/            one folder per method
docs/                PROTOCOL.md, METRICS.md, COSTS.md, BASELINES.md
scripts/smoke_test.sh
tests/
```

## Install

```bash
pip install -r requirements.txt       # Python 3.10, PyTorch 2.6 / CUDA 12.4 used for the paper
```

CIFAR-100 downloads into `./data` on first use. To use a different directory,
set `GTEP_DATA_ROOT=/path/to/data`.

## Quick check (runs on CPU, a few minutes)

```bash
bash scripts/smoke_test.sh            # every method, 2 tasks, 1 epoch, 64 samples
python -m pytest -q tests
```

On CPU the smoke script skips SNV-A, because its Shapley valuation takes
more than 30 minutes on ResNet-18 without a GPU. `tests/test_snv.py::TestSNVAdaptive`
covers SNV-A on CPU. Use `GTEP_DEVICE=cuda:0 bash scripts/smoke_test.sh` to include it.

