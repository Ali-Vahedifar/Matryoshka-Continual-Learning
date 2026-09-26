# MCL3 CIFAR-10 ablation campaign

This campaign evaluates the actual `MCL3 = MCL` implementation in `MCL/mcl.py`.
The experimental runner is `campaign/mcl_cifar10_ablation.py`; the production
implementation is unchanged. A hashed source snapshot is taken at preparation,
and all queued jobs use that snapshot, so later editing cannot change a campaign
halfway through.

## Protocol

- CIFAR-10, all ten classes. Official 10,000-image test partition is untouched.
- Fixed stratified split of the official training partition: 4,500 training and
  500 validation images per class, split seed 1234. The exact sample indices are
  saved for every run. This is a dedicated CIFAR-10 protocol, not the pooled
  CIFAR-20 GTEP evaluation-half protocol.
- Main sequence: five tasks of two classes. A separate sequence-length study
  uses two tasks of five classes; one-class tasks are excluded because local CE
  would be degenerate.
- Seeds 42, 43 and 44 control initialization, task order and augmentation. Every
  variant uses the same class order and initialization within a seed/scenario
  and task-count setting. Split membership is fixed across seeds. Training RNG
  is reset at each task; diagnostics preserve its state.
- Randomly initialized CIFAR ResNet-18, 512-dimensional features, no pretraining.
- Adam, learning rate 0.001, weight decay 0, no scheduler, batch size 64,
  maximum 100 epochs/task, patience 10. The best validation-loss state is restored.
  These settings are declared reference settings, not test-selected winners.
- Reference MCL3: widths 32/64/128/256/512; distillation lambda 1, temperature 2,
  beta 0; cosine scale 16 in Class-IL; density mixing alpha 0.25 with nearest
  class means; variance shrinkage 0.1; weight alignment enabled.
- The test split is used only for reporting. Post-hoc test sweeps describe
  sensitivity; their best point must not be presented as a validation-selected
  new method. Any subsequent configuration selection needs validation evidence.

## Scope: 35 training configurations, 198 runs

Both scenarios and all three seeds are used except the four classifier variants,
which apply only to Class-IL. Component studies are first in the queue.

| Family | Configurations beyond full MCL3 |
|---|---|
| Core components | No distillation; only 512 dimensions; only 512 without distillation; no weight alignment |
| Reference baselines | Sequential fine-tuning (`sgd`, using the same Adam optimizer); LwF |
| Distillation location | Only 512; only 32; endpoints 32 and 512; supervised losses remain at all five widths |
| Width count | 128/512; 32/128/512; 8/16/32/64/128/256/512 |
| Distillation weighting | beta = -0.5, 0.5, 1, 2 (reference is 0); density/alignment unchanged |
| Distillation strength | lambda = 0.1, 0.5, 2, 5 (reference 1; zero covered in core) |
| Temperature | 1, 4, 8 (reference 2) |
| Learning rate | 0.0003, 0.003 (reference 0.001) |
| Class-IL classifier | Cosine scale 8 or 32; linear classifier; CE over all seen classes instead of current task only |
| Old classifier updates | Freeze old heads |
| Training sample size | Nested 10%, 25%, 50% training subsets; unchanged validation/test data |
| Number of tasks | Two five-class tasks |

Sample-size experiments change both feature learning and density estimation.
They do not by themselves isolate prototype-estimation error. Likewise, changing
task count changes classes per task and number of training stages.

## Extra evaluations without duplicate training

Each task checkpoint produces the main production-readout accuracy trajectory
and parametric-only trajectories at 32/64/128/256/512 dimensions. The latter
exclude density blending consistently for every method. Widths not supervised
by a given variant, and widths of non-nested baselines, are truncation diagnostics.

For each Class-IL MCL variant, the same features and stored statistics also give
the full-width readout grid:

- alpha = 0, 0.1, 0.25, 0.5, 0.75, 1;
- nearest-class-mean and shared diagonal variance distance.

The two alpha=0 metric choices are redundant controls and are not independent
experiments. `no_density_readout` and `no_density_no_weight_alignment` appear as
explicit derived rows in the main table, from full MCL3 and the no-alignment run
respectively. These are inference interventions; their training cost has not
been reduced by disabling density-statistics collection.

Density blending is inactive for Task-IL. Positive head-weight rescaling cancels
in the Class-IL cosine readout, so weight-alignment training interventions test
effects on subsequent optimization, not an immediate cosine prediction change.
The default nearest-mean readout does not use variances; a variance-only ablation
would be meaningless there. The diagonal metric is the implementation's shared
diagonal metric, not a full reproduction of FeCAM.

At the last task, retrieval uses test images as queries and validation images as
the gallery (no self matches), with cosine similarity at each width. Task-IL
retrieval restricts the gallery to the query's task; Class-IL uses all classes.
Recall@k is the proportion of queries with at least one correct-class gallery
item in the first k results. Gallery float32 embedding storage is reported.
One search-pass duration is a diagnostic, not a repeated latency benchmark;
it excludes feature extraction and normalization. A smaller embedding does not
avoid running the full backbone in this implementation.

## Outputs

For each job, `runs/JOB/` contains configuration, split indices, progress, task
checkpoints, logs and the final `result.json`. Every raw accuracy matrix and
random-initialization baseline is preserved, along with validation histories,
epochs, training time, peak allocated memory and stored density-state bytes.
Checkpoint files omit a duplicate teacher because it equals the student at each
task boundary. There is no retained training-image replay buffer; density state
and teacher/model storage are still real memory costs.

`report/` updates after every completed run:

- `metrics_per_seed.csv`, `metrics_summary.csv`: ACC/BWT/FWT/PS/AvgAcc/HARMONIC/AF
  for main, width and readout views.
- `paired_differences.csv`, `paired_summary.csv`: same-seed differences from full
  MCL3. Sample-size and task-count differences are sensitivity comparisons, not
  estimates of a single mechanism.
- `retrieval_per_seed.csv`, `retrieval_summary.csv`.
- `costs_per_seed.csv`, `costs_summary.csv`.
- `ablation_table.tex`: only completed groups with all three seeds, mean ± sample
  SD. Raw PS is dimensionless; LaTeX displays 100 times PS and its SD.
- PDF/PNG plots of width versus ACC/BWT and density-readout sensitivity, generated
  once the corresponding three-seed groups exist.

Single-seed CSV summaries have a missing SD, never an invented zero. Smoke tests
are kept in a separate `smoke/` folder and never enter reports. Reference labels
describe implementations rather than guarantees of upper/lower performance.
Validation-accuracy logging corrects the generic loop's local-label mismatch for
MCL Class-IL; the validation loss used for stopping is unchanged.

## Scheduling and commands

The campaign runs one job at a time. By default it waits until both the existing
SNV-A campaign process and all GPU compute processes are gone. It does not stop
or modify other experiments. `--share-gpu` is an explicit override, unsuitable
for clean cost comparisons.

Example preparation (performed once):

```bash
python campaign/mcl_cifar10_ablation.py prepare \
  --root $ABLATION_ROOT \
  --data-root $DATA_ROOT
```

Run/resume using the prepared snapshot:

```bash
python $ABLATION_ROOT/source/campaign/mcl_cifar10_ablation.py \
  campaign --root $ABLATION_ROOT
```

Read live `status.json`, then the active run's `progress.json` for task-level
progress. A nonblocking file lock prevents duplicate campaign drivers. Completed
runs are skipped on resume; an interrupted incomplete job restarts from task one.
The driver stops at the first failed job and records the log path, avoiding a
long series of failures. Checkpoints are retained for later evaluation.

Verification command:

```bash
python -m unittest discover -s tests -p test_mcl_cifar10_ablation.py -v
```

Tests compare reference losses and gradients against unmodified MCL, selected
distillation against a manual KL calculation, main readout against production
prediction, global-CE scope, metric definitions, split isolation and RNG handling.
Small CPU smoke runs additionally traverse the full five-task training/evaluation
and checkpoint path before the GPU campaign is queued.
