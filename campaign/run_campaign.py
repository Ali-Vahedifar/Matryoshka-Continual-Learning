#!/usr/bin/env python3
"""Run the CIFAR-100 GTEP campaign: every method, both scenarios, tuning then
clean evaluation.

Protocol (identical for every method, see docs/PROTOCOL.md):

* CIFAR-100 is split into two disjoint halves of 50 classes (split seed 1234).
  Half 1 (``D_HT``) selects hyper-parameters, half 2 (``D_E``) produces the
  reported numbers.  Each half is 10 tasks of 5 classes.
* R = 30 configurations are drawn per method from its search space with sample
  seed 7, and each is run on seeds 42/43/44 of ``D_HT``.  The winner is the
  configuration with the highest mean HARMONIC score (harmonic mean of final
  ACC and average-over-tasks ACC on the D_HT validation split).
* The winner is then run three times on ``D_E``, one run at a time on an
  otherwise idle GPU, so the published cost numbers
  (GPU-hours, peak memory, GFLOPs, inference time) are measured exclusively.
  A D_E result whose ledger did not observe an exclusive GPU is rejected.
* SNV means SNV-A (``SNV/snv_adaptive.py``), driven through the unchanged
  audited worker by ``snv_adaptive_run.py``; the SNV-A switches travel inside
  the config and are recorded in every ``result.json``.

The driver is resumable: a finished run whose ``result.json`` matches its
identity (method, scenario, half, seed, config, 10 tasks, 200-epoch policy) is
reused, so re-running the command after an interruption continues the campaign.
A run that exits without a valid result, or writes nothing to its log for
--stall_hours, is killed and retried up to --retries times.

  python campaign/run_campaign.py --out runs/cifar100 --gpus 0 1 2 3
  python campaign/run_campaign.py --out runs/cifar100 --gpus 0 --methods mcl snv
  # CIFAR-20 (superclasses: halves of 10, 5 tasks x 2) on one GPU, 16 tuning runs
  # sharing it; D_E winner runs still take it alone.
  GTEP_DATASET=cifar20 GTEP_NUM_WORKERS=0 python campaign/run_campaign.py \
      --out runs/cifar20 --gpus 0 --pack 16
"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault('GTEP_PROTOCOL', 'legacy')
import audited_gtep as G                                            # noqa: E402

# Dispatch order: MCL first, SNV-A last (it is also gated behind the rest).
METHODS = ['mcl', 'sgd', 'joint', 'ewc', 'si', 'lwf', 'wsn', 'pec', 'spacenet',
           'nispa', 'uniclun', 'snv']
SEEDS = (42, 43, 44)
ROUNDS = 30
EPOCHS = 200
PATIENCE = 15
TASKS = G.TASKS
SAMPLE_SEED = 7

# SNV-A switches (SNV/snv_adaptive.py).  Fixed across the search, recorded in
# every config so a run file states the variant it measured.
SNV_VARIANT = dict(task_local=True, frozen_norm_eval=True, bn_recal=True,
                   routing='maxprob', adaptive=True, adaptive_rule='coverage',
                   adaptive_coverage=0.9)


def scenarios(method):
    """PEC is Class-IL only, WSN needs the task identity so it is Task-IL only."""
    return (['class_il'] if method == 'pec' else
            ['task_il'] if method == 'wsn' else ['class_il', 'task_il'])


# --snv_scenario_switches: the per-scenario switches of the CIFAR-100 winners.
# Class-IL routes on rotation+energy scores and trains the rotation head;
# routing has no effect in Task-IL, where the rotation loss stays off.
SNV_SCENARIO = {'class_il': dict(routing='rot_energy_z', rot_aux=1.0),
                'task_il': dict(routing='maxprob', rot_aux=0.0)}


def snv_variant(scenario, per_scenario):
    return {**SNV_VARIANT, **(SNV_SCENARIO[scenario] if per_scenario else {})}


def configs_for(method, scenario='class_il', per_scenario=False):
    rng = random.Random(SAMPLE_SEED)
    draws = [G.sample(G.SPACE[method], rng) for _ in range(ROUNDS)]
    return [{**d, **snv_variant(scenario, per_scenario)} if method == 'snv' else d for d in draws]


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def valid_result(path, method, scenario, half, seed, config, tasks=TASKS, epochs=EPOCHS):
    """Return the result only if it is this exact job, run under this protocol."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except ValueError:
        return None
    ok = (d.get('complete') and d.get('completed_tasks') == tasks
          and d.get('method') == method and d.get('scenario') == scenario
          and d.get('half') == half and d.get('seed') == seed and d.get('config') == config
          and d.get('training_policy', {}).get('max_epochs') == epochs)
    return d if ok else None


def cost_row(method, scenario, result_file, selected_config):
    d = json.loads(Path(result_file).read_text())
    cost = d['cost_summary']
    if not cost['gpu_exclusive_observed']:
        raise RuntimeError(f'Cost run was not observed GPU-exclusive: {result_file}')
    inference = [x for x in d['inference'] if x['batch_size'] == 64]
    mean = lambda key: (sum(x[key] for x in inference) / len(inference)) if inference else None
    return dict(method=method, scenario=scenario, seed=d['seed'],
                **{k: v for k, v in d['metrics'].items() if k != 'I'},
                train_minutes_per_task=cost['train_minutes_per_task_mean'],
                gpu_hours=cost['accounted_run_gpu_hours'],
                peak_gpu_MB=(cost['gpu_peak_allocated_bytes'] or 0) / 1e6 or None,
                model_MB=cost['final_state']['resident_parameter_bytes'] / 1e6,
                gpu_util_percent=cost['gpu_util_percent_sample_mean'],
                checkpoint_bytes=d['checkpoint_bytes'],
                infer_ms_per_sample=mean('inference_ms_per_sample'),
                GFLOPs_per_sample=mean('supported_operator_gflops_per_sample'),
                selected_config=json.dumps(selected_config, sort_keys=True),
                result=str(result_file))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, help='campaign directory (created, resumable)')
    p.add_argument('--gpus', type=int, nargs='+', required=True,
                   help='GPU indices in nvidia-smi order; runs are pinned by UUID')
    p.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    p.add_argument('--scenarios', nargs='+', default=['class_il', 'task_il'],
                   choices=['class_il', 'task_il'])
    p.add_argument('--rounds', type=int, default=ROUNDS)
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--patience', type=int, default=PATIENCE)
    p.add_argument('--tasks', type=int, default=TASKS)
    p.add_argument('--poll', type=float, default=5.0)
    p.add_argument('--pack', type=int, default=1,
                   help='concurrent tuning / FLOP-count runs per GPU; D_E winner runs always run alone')
    p.add_argument('--retries', type=int, default=2, help='retries per failed or hung run')
    p.add_argument('--stall_hours', type=float, default=3.0,
                   help='kill a run that writes nothing to its log for this long')
    p.add_argument('--snv_scenario_switches', action='store_true',
                   help='SNV-A: rot_energy_z routing + rot_aux=1 in Class-IL, maxprob + rot_aux=0 in Task-IL')
    args = p.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not G.GPUS:
        raise RuntimeError('no GPUs visible; set GTEP_GPU_UUIDS')
    for gpu in args.gpus:
        if gpu >= len(G.GPUS):
            raise RuntimeError(f'GPU index {gpu} out of range; {len(G.GPUS)} visible')

    # One campaign per directory.
    lock = (out / 'campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Runs outlive a dead driver (own sessions); starting again would launch the
    # same jobs a second time into the same directories.
    runs_dir = str(out / 'runs') + '/'
    orphans = [q.pid for q in psutil.process_iter(['cmdline'])
               if any(runs_dir in a for a in q.info['cmdline'] or [])]
    if orphans:
        raise RuntimeError(f'runs from an earlier driver are still alive (pids {orphans}); '
                           'kill their process groups first')

    fingerprint = G.source_fingerprint()
    protocol = dict(dataset=f'{G.DATASET_LABEL} disjoint halves', classes_per_half=G.HALF_CLASSES,
                    tasks=args.tasks, classes_per_task=G.CPT, data_loader_workers=G.NUM_WORKERS,
                    tuning_runs_per_gpu=args.pack, D_HT=1, D_E=2, split_seed=1234, seeds=list(SEEDS),
                    R=args.rounds, sample_seed=SAMPLE_SEED, max_total_epochs=args.epochs,
                    patience=args.patience, gtep_protocol=os.environ['GTEP_PROTOCOL'],
                    selection='mean over seeds of the harmonic mean of final ACC and AvgAcc on D_HT validation',
                    costs='tuning runs share GPUs and their timings are not reported; D_E winner runs '
                          'take the GPU exclusively and carry the published cost numbers; training '
                          'FLOPs come from one extra D_E seed-42 run per winner with FLOP counting on '
                          '(its timings are not reported)',
                    snv_variant={s: snv_variant(s, args.snv_scenario_switches) for s in ('class_il', 'task_il')},
                    methods=args.methods,
                    spaces={m: G.SPACE[m] for m in args.methods},
                    source_sha256=fingerprint)
    protocol = G.serial(protocol)
    if (out / 'protocol.json').exists():
        previous = json.loads((out / 'protocol.json').read_text())
        if previous['source_sha256'] != protocol['source_sha256']:
            raise RuntimeError('training source changed since this campaign started; '
                               'start a new --out directory instead of mixing code versions')
        # Documented, deliberate source fixes survive restarts.
        protocol['source_amendments'] = previous.get('source_amendments', [])
    atomic(out / 'protocol.json', protocol)

    def verify_source():
        if G.source_fingerprint() != fingerprint:
            raise RuntimeError('training source changed while the campaign was running')

    jobs, blocks, running, failed = [], [], [], []

    def usable(job):
        """A finished run of this exact job; a winner run must also have been GPU-exclusive."""
        d = valid_result(job['directory'] / 'result.json', job['method'], job['scenario'], job['half'],
                         job['seed'], job['config'], args.tasks, args.epochs)
        return bool(d) and (not job['exclusive'] or d['cost_summary']['gpu_exclusive_observed'])

    def add_job(method, scenario, half, seed, config, tag, flops=False):
        name = f'{method}_{scenario}_{tag}_s{seed}'
        job = dict(method=method, scenario=scenario, half=half, seed=seed, config=config,
                   directory=out / 'runs' / name, state='pending', priority=len(jobs),
                   flops=flops, exclusive=half == 2 and not flops, attempts=0)
        if usable(job):
            job['state'] = 'complete'
        jobs.append(job)
        return job

    def add_block(method, scenario):
        configs = configs_for(method, scenario, args.snv_scenario_switches)[:args.rounds]
        block = dict(method=method, scenario=scenario, configs=configs, winner_jobs=None,
                     flops_job=None, done=False, signature=None, path=out / 'blocks' / f'{method}_{scenario}.json')
        block['trials'] = [[add_job(method, scenario, 1, seed, config, f'ht_r{i}') for seed in SEEDS]
                           for i, config in enumerate(configs)]
        blocks.append(block)

    def refresh():
        for block in blocks:
            complete = [dict(trial_index=i, config=block['configs'][i],
                             score=sum(json.loads((j['directory'] / 'result.json').read_text())
                                       ['metrics']['HARMONIC'] for j in trial) / len(SEEDS),
                             results=[str(j['directory'] / 'result.json') for j in trial])
                        for i, trial in enumerate(block['trials'])
                        if all(j['state'] == 'complete' for j in trial)]
            best = max(complete, key=lambda t: t['score']) if complete else None
            if len(complete) == len(block['trials']) and block['winner_jobs'] is None:
                block['winner_jobs'] = [add_job(block['method'], block['scenario'], 2, seed,
                                                best['config'], 'clean_eval') for seed in SEEDS]
                # Training FLOPs need instrumentation inside the timed phases, so
                # they come from a separate run whose timings are not reported.
                block['flops_job'] = add_job(block['method'], block['scenario'], 2, SEEDS[0],
                                             best['config'], 'train_flops', flops=True)
            evaluation = [str(j['directory'] / 'result.json')
                          for j in block['winner_jobs'] or [] if j['state'] == 'complete']
            flops = block['flops_job']
            train_flops = (str(flops['directory'] / 'result.json')
                           if flops and flops['state'] == 'complete' else None)
            block['done'] = (len(complete) == len(block['trials']) and len(evaluation) == len(SEEDS))
            signature = (len(complete), len(evaluation), train_flops)
            if signature != block['signature']:
                atomic(block['path'], dict(method=block['method'], scenario=block['scenario'],
                                           config_origin=f'search space, sample seed {SAMPLE_SEED}',
                                           variant=(snv_variant(block['scenario'], args.snv_scenario_switches)
                                                    if block['method'] == 'snv' else None),
                                           tuning=complete, best=best, evaluation=evaluation,
                                           train_flops=train_flops))
                block['signature'] = signature

    def launch(job, gpu):
        """Start a run.  A D_E winner run takes the GPU's exclusive lock and needs
        an idle device; tuning and FLOP-count runs share it (timings unreported)."""
        verify_source()
        lease = None
        if job['exclusive']:
            lease = open(f'/tmp/gtep-{G.GPUS[gpu]}.lock', 'w')
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                return False
            if gpu_busy(gpu):
                lease.close()
                return False
        directory = Path(job['directory'])
        directory.mkdir(parents=True, exist_ok=True)
        worker = 'snv_adaptive_run.py' if job['method'] == 'snv' else 'audited_gtep.py'
        command = [sys.executable, '-u', str(ROOT / worker)]
        if job['method'] != 'snv':
            command += ['--one', '--method', job['method']]
        command += ['--scenario', job['scenario'], '--half', str(job['half']), '--seed', str(job['seed']),
                    '--config', json.dumps(job['config']), '--out', str(directory),
                    '--epochs', str(args.epochs), '--patience', str(args.patience),
                    '--tasks', str(args.tasks)]
        # --flops instruments the timed training phases, so it only ever goes to
        # the separate train_flops run, never to a run whose costs are reported.
        if job['flops']:
            command.append('--flops')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=G.GPUS[gpu], GTEP_PROTOCOL=os.environ['GTEP_PROTOCOL'],
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        atomic(directory / 'command.json', dict(command=command, gpu=G.GPUS[gpu],
               cost_policy=('exclusive GPU; clean training timing; separate inference FLOP pass'
                            if job['exclusive'] else 'training FLOP count; timings not reported'
                            if job['flops'] else 'tuning run; timings not reported')))
        log = (directory / 'train.log').open('a')
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        job.update(state='running', gpu=gpu)
        running.append(dict(job=job, process=process, lease=lease, log=log, gpu=gpu, started=time.time()))
        print('START', directory.name, 'GPU', gpu, flush=True)
        return True

    try:
        import pynvml
        pynvml.nvmlInit()
        handles = {gpu: pynvml.nvmlDeviceGetHandleByUUID(G.GPUS[gpu]) for gpu in args.gpus}

        def gpu_busy(gpu):
            return bool(pynvml.nvmlDeviceGetComputeRunningProcesses(handles[gpu]))
    except Exception:                                   # pynvml is optional
        print('pynvml unavailable: relying on the per-GPU lock alone', flush=True)

        def gpu_busy(gpu):
            return False

    for method in args.methods:
        for scenario in scenarios(method):
            if scenario in args.scenarios:
                add_block(method, scenario)
    # SNV runs after the baselines when both are in the same campaign, so the
    # baseline table is fixed before the proposed method is measured.
    gate_snv = any(b['method'] != 'snv' for b in blocks) and any(b['method'] == 'snv' for b in blocks)

    while True:
        for entry in list(running):
            job = entry['job']
            code = entry['process'].poll()
            if code is None:
                # Silence since this attempt started: a retry reuses an old, stale log.
                quiet = time.time() - max(entry['started'],
                                          (Path(job['directory']) / 'train.log').stat().st_mtime)
                if quiet < args.stall_hours * 3600:
                    continue
                # Hung (e.g. a deadlocked DataLoader): kill the run's whole session.
                try:
                    os.killpg(entry['process'].pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = entry['process'].wait()
                reason = f'killed after {quiet / 3600:.2f} h without log output'
            else:
                reason = f'exit code {code}'
            entry['log'].close()
            if entry['lease']:
                entry['lease'].close()
            running.remove(entry)
            if usable(job):
                job['state'] = 'complete'
            else:
                if code == 0:
                    reason += ' without a usable result (incomplete, or a winner run not GPU-exclusive)'
                job['attempts'] += 1
                job['state'] = 'pending' if job['attempts'] <= args.retries else 'failed'
                failed.append(dict(directory=str(job['directory']), reason=reason, returncode=code,
                                   attempt=job['attempts'], retrying=job['state'] == 'pending'))
                atomic(out / 'failed_jobs.json', failed)
                print('RETRY' if job['state'] == 'pending' else 'FAILED', Path(job['directory']).name,
                      reason, flush=True)
        refresh()
        baselines_done = all(b['done'] for b in blocks if b['method'] != 'snv')
        for gpu in args.gpus:
            while True:
                here = [e for e in running if e['gpu'] == gpu]
                ready = [j for j in jobs if j['state'] == 'pending'
                         and (j['method'] != 'snv' or baselines_done or not gate_snv)]
                if not ready or any(e['job']['exclusive'] for e in here):
                    break
                # Finished searches' winner measurements first, then tuning in order.
                job = min(ready, key=lambda j: (j['half'] != 2, j['priority']))
                # An exclusive run waits for the GPU to drain; nothing joins it.
                full = bool(here) if job['exclusive'] else len(here) >= args.pack
                if full or not launch(job, gpu):
                    break
        states = [j['state'] for j in jobs]
        atomic(out / 'status.json', dict(
            stage='snv' if baselines_done else 'baselines',
            active=[dict(gpu=e['gpu'], run=Path(e['job']['directory']).name) for e in running],
            complete=states.count('complete'), pending=states.count('pending'),
            running=states.count('running'), failed=states.count('failed'),
            retried=sum(f['retrying'] for f in failed),
            blocks_done=sum(b['done'] for b in blocks), blocks=len(blocks),
            updated_at=time.time()))
        # Also wait for the FLOP-count runs; a failed one does not block the report.
        if all(b['done'] for b in blocks) and not running and 'pending' not in states:
            break
        if not running and not any(s == 'pending' for s in states):
            raise RuntimeError(f'campaign stalled on failed runs; see {out / "failed_jobs.json"}')
        time.sleep(args.poll)

    rows = []
    for block in blocks:
        state = json.loads(Path(block['path']).read_text())
        for result in state['evaluation']:
            rows.append(cost_row(block['method'], block['scenario'], result, state['best']['config']))
    with (out / 'results_per_seed.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic(out / 'CAMPAIGN_READY.json', dict(completed_at=time.time(),
           blocks=[dict(method=b['method'], scenario=b['scenario'], state=str(b['path'])) for b in blocks],
           table=str(out / 'results_per_seed.csv')))
    print('campaign complete:', out / 'results_per_seed.csv', flush=True)


if __name__ == '__main__':
    main()
