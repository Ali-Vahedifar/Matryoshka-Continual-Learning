"""Reproducible CIFAR-10 MCL3 component/sensitivity campaign.

Separate runner and experimental subclass; the published MCL implementation is
not modified. See docs/MCL_CIFAR10_ABLATIONS.md for the protocol and commands.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import os
import random
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variants():
    rows = []
    def add(name, family, scenarios=('class_il', 'task_il'), **kw):
        rows.append(dict(name=name, family=family, scenarios=list(scenarios), overrides=kw))
    add('full', 'core')
    add('no_distillation', 'core', sdft_lambda=0.)
    add('single_width', 'core', granularities=[512])
    add('single_width_no_distillation', 'core', granularities=[512], sdft_lambda=0.)
    add('no_weight_alignment', 'core', weight_align=False)
    add('sgd', 'baseline', method='sgd')
    add('lwf', 'baseline', method='lwf')
    for name, widths in [('widest', [512]), ('smallest', [32]), ('endpoints', [32, 512])]:
        add('distill_' + name, 'distillation_location', distill_widths=widths)
    for name, widths in [('two', [128, 512]), ('three', [32, 128, 512]),
                         ('seven', [8, 16, 32, 64, 128, 256, 512])]:
        add('widths_' + name, 'width_count', granularities=widths)
    for value in [-.5, .5, 1., 2.]:
        add('beta_' + str(value), 'beta', sdft_beta=value)
    for value in [.1, .5, 2., 5.]:
        add('lambda_' + str(value), 'lambda', sdft_lambda=value)
    for value in [1., 4., 8.]:
        add('temperature_' + str(value), 'temperature', temperature=value)
    for value in [3e-4, 3e-3]:
        add('lr_' + str(value), 'learning_rate', lr=value)
    for value in [8., 32.]:
        add('cosine_scale_' + str(value), 'classifier', scenarios=('class_il',), cosine_scale=value)
    add('linear_head', 'classifier', scenarios=('class_il',), linear_cil=True)
    add('global_cross_entropy', 'classifier', scenarios=('class_il',), global_ce=True)
    add('freeze_old_heads', 'head_updates', freeze_old_heads=True)
    for fraction in [.1, .25, .5]:
        add('train_fraction_' + str(fraction), 'sample_size', train_fraction=fraction)
    add('two_tasks', 'task_count', tasks=2)
    return rows


DEFAULT = dict(tasks=5, train_fraction=1., lr=1e-3, epochs=100, patience=10,
               batch_size=64, sdft_lambda=1., temperature=2., sdft_beta=0.,
               density_alpha=.25, density_metric='ncm', density_shrinkage=.1,
               weight_align=True, cosine_scale=16., granularities=[32,64,128,256,512],
               distill_widths=None, linear_cil=False, global_ce=False,
               freeze_old_heads=False, method='mcl', workers=2, split_seed=1234)


def prepare(root, data_root, wait_pids):
    root = Path(root).resolve()
    if (root/'manifest.json').exists():
        raise FileExistsError('Campaign already exists; resume with campaign, do not overwrite it.')
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root/'source'
    sources = ['models.py', 'datasets.py', 'cl_base.py', 'training_policy.py',
               'metrics.py', 'MCL/mcl.py', 'SGD/sgd.py', 'LwF/lwf.py',
               'campaign/mcl_cifar10_ablation.py']
    hashes = {}
    for relative in sources:
        dst = snapshot/relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO/relative, dst)
        hashes[relative] = digest(dst)
    jobs = []
    for variant in variants():
        for scenario in variant['scenarios']:
            for seed in [42, 43, 44]:
                config = {**DEFAULT, **variant['overrides'], 'scenario': scenario, 'seed': seed}
                jobs.append(dict(id=f"{variant['name']}__{scenario}__s{seed}",
                                 variant=variant['name'], family=variant['family'], config=config))
    manifest = dict(dataset='cifar10', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        data_root=str(Path(data_root).resolve()), source_sha256=hashes,
        protocol='official test untouched; stratified 4500 train/500 validation per class; fixed split seed 1234',
        initialization='random CIFAR ResNet-18, 512 features; all heads initialized before training',
        seeds=[42,43,44], optimizer='Adam', scheduler='none', weight_decay=0.,
        reference='fixed declared settings, not selected on test results',
        width_metrics='parametric classifier at each width; main metrics use production MCL readout',
        readout_grid=dict(alpha=[0.,.1,.25,.5,.75,1.], metric=['ncm','diag'],
                          scope='full width, Class-IL only; post-hoc metrics, not training variants'),
        sample_size='nested subsets of fixed training split; validation/test unchanged',
        wait_pids=wait_pids, jobs=jobs)
    atomic(root/'manifest.json', manifest)
    atomic(root/'status.json', dict(state='prepared', total=len(jobs), complete=0))
    print(json.dumps(dict(root=str(root), training_jobs=len(jobs), variants=len(variants()))))


def experimental_class():
    import torch
    import torch.nn.functional as F
    from MCL.mcl import MCL, granularity_weights

    class AblationMCL(MCL):
        def __init__(self, *args, distill_widths=None, linear_cil=False, global_ce=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.distill_widths = tuple(distill_widths or self.granularities)
            if not set(self.distill_widths).issubset(self.granularities):
                raise ValueError('Distillation widths must be trained widths.')
            self.distill_weights = granularity_weights(self.distill_widths, self.sdft_beta)
            self.linear_cil, self.global_ce = linear_cil, global_ce

        def _doll(self, head, features, m):
            if self.linear_cil and self.scenario == 'class_il':
                return F.linear(features[:, :m], head.weight[:, :m], head.bias)
            return super()._doll(head, features, m)

        def _local_ce(self, out, y):
            return F.cross_entropy(out, y if self.global_ce else y-self._offset(self._task))

        def logits(self, x, task_id, for_training=True):
            if not self.global_ce:
                return super().logits(x, task_id, for_training)
            self._task, self._features = task_id, self.model.get_features(x)
            return self._heads_of(self.model, list(range(task_id+1)), self._features,
                                  self.granularities[-1])

        def validate(self, loader, task_id):
            # Same stopping loss as the source. Correct the displayed accuracy
            # for local-CE CIL, whose labels in the loader are global.
            self.model.eval()
            loss_sum = correct = total = 0
            with torch.no_grad():
                for x,y in loader:
                    x,y=x.to(self.device),y.to(self.device)
                    out=self.logits(x,task_id,False)
                    loss_sum += self.criterion(out,y).item()*y.numel()
                    target=y if self.global_ce else y-self._offset(task_id)
                    correct += int(out.argmax(1).eq(target).sum())
                    total += y.numel()
            return loss_sum/max(total,1),correct/max(total,1)

        def extra_loss(self, x, y, logits, task_id):
            # Unchanged source implementation for the reference and normal sweeps.
            if not self.global_ce and self.distill_widths == self.granularities:
                return super().extra_loss(x, y, logits, task_id)
            f = self._features
            target = y if self.global_ce else y-self._offset(task_id)
            tasks = list(range(task_id+1)) if self.global_ce else [task_id]
            ce = [F.cross_entropy(self._heads_of(self.model, tasks, f, m), target)
                  for m in self.granularities]
            loss = sum(ce)/len(ce)-F.cross_entropy(logits, target)
            if self.teacher is not None and task_id > 0 and self.sdft_lambda > 0:
                with torch.no_grad():
                    tf = self.teacher.get_features(x)
                tau = self.temperature
                for m in self.distill_widths:
                    with torch.no_grad():
                        teacher = self._heads_of(self.teacher, list(range(task_id)), tf, m)
                    student = self._heads_of(self.model, list(range(task_id)), f, m)
                    kl = F.kl_div(F.log_softmax(student/tau, dim=1), F.softmax(teacher/tau, dim=1),
                                  reduction='batchmean') * tau**2
                    loss = loss + self.sdft_lambda*self.distill_weights[m]*kl
            return loss
    return AblationMCL


def make_loaders(config, data_root, smoke=False):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from torchvision.datasets import CIFAR10
    from datasets import _TaskSubset, build_transforms
    tr = CIFAR10(data_root, train=True, download=False)
    te = CIFAR10(data_root, train=False, download=False)
    order = np.random.RandomState(config['seed']).permutation(10).tolist()
    cpt = 10//config['tasks']
    assert 10 % config['tasks'] == 0 and cpt >= 2
    splits = {}
    for c in range(10):
        ids = np.flatnonzero(np.asarray(tr.targets) == c)
        np.random.RandomState(config['split_seed']+c).shuffle(ids)
        train_ids, val_ids = ids[500:], ids[:500]
        train_ids = train_ids[:max(1, int(len(train_ids)*config['train_fraction']))]
        test_ids = np.flatnonzero(np.asarray(te.targets) == c)
        if smoke:
            train_ids, val_ids, test_ids = train_ids[:4], val_ids[:2], test_ids[:2]
        splits[c] = (train_ids, val_ids, test_ids)
    loaders, split_record = [], dict(class_order=order, per_class={})
    for c, parts in splits.items():
        split_record['per_class'][str(c)] = {k: v.tolist() for k, v in zip(['train', 'val', 'test'], parts)}
    for task in range(config['tasks']):
        classes = order[task*cpt:(task+1)*cpt]
        mapping = {c: (task*cpt+i if config['scenario']=='class_il' else i) for i,c in enumerate(classes)}
        group = []
        for part in range(3):
            indices = np.concatenate([splits[c][part] for c in classes])
            ds = _TaskSubset(tr if part < 2 else te, indices,
                             build_transforms('cifar10', part == 0), mapping)
            generator = torch.Generator().manual_seed(config['seed']*1000+task*10+part)
            group.append(DataLoader(ds, batch_size=config['batch_size'], shuffle=part==0,
                num_workers=0 if smoke else config['workers'], generator=generator,
                pin_memory=not smoke, persistent_workers=False))
        loaders.append(group)
    return loaders, split_record


@contextlib.contextmanager
def preserved_rng():
    import numpy as np
    import torch
    py, npstate, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(npstate)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def metric_dict(matrix, baseline):
    import numpy as np
    from metrics import ContinualLearningMetrics
    a = np.asarray(matrix, dtype=float)
    tracker = ContinualLearningMetrics(len(a))
    tracker.accuracy_matrix = a
    tracker.set_random_baseline(baseline)
    raw = tracker.get_all_metrics()
    raw['AvgAcc'] = float(np.mean([np.mean(a[t,:t+1]) for t in range(len(a))]))
    raw['HARMONIC'] = 2*raw['ACC']*raw['AvgAcc']/(raw['ACC']+raw['AvgAcc']) if raw['ACC']+raw['AvgAcc'] else 0.
    return {k: (float(v)*(1 if k in ('P','S','PS') else 100)) for k,v in raw.items()
            if k != 'I' and np.isfinite(v)}


def extract(method, loader):
    import torch
    features, labels = [], []
    method.model.eval()
    with torch.inference_mode():
        for x,y in loader:
            features.append(method.model.get_features(x.to(method.device)).cpu())
            labels.append(y)
    return torch.cat(features), torch.cat(labels)


def scores(method, features, tasks, width):
    import torch
    import torch.nn.functional as F
    if hasattr(method, '_heads_of'):
        return method._heads_of(method.model, tasks, features, width)
    return torch.cat([F.linear(features[:,:width], method.model.heads[str(t)].weight[:,:width],
                               method.model.heads[str(t)].bias) for t in tasks], dim=1)


def evaluate_stage(method, loaders, stage, config, readout_grid):
    """One feature pass per test task; all evaluation variants reuse its features.

    Width metrics are parametric-only. The main full-width metric matches the
    production readout. Future diagnostics always use all output classes.
    """
    import torch
    from MCL.mcl import _standardize
    widths = [32,64,128,256,512]
    out = {'main': [], **{'width_'+str(m): [] for m in widths}}
    if hasattr(method, 'density_stats') and config['scenario']=='class_il':
        out.update({f'readout_{metric}_{alpha:g}': [] for metric in ['ncm','diag']
                    for alpha in readout_grid})
    cache = []
    with torch.inference_mode():
        for task, triple in enumerate(loaders):
            f_cpu, y_cpu = extract(method, triple[2])
            cache.append((f_cpu,y_cpu))
            f, y = f_cpu.to(method.device), y_cpu.to(method.device)
            tasks = ([task] if config['scenario']=='task_il' else
                     list(range(config['tasks'] if stage < 0 or task > stage else stage+1)))
            parametric = {}
            for m in widths:
                z = scores(method, f, tasks, m)
                parametric[m] = z
                out['width_'+str(m)].append(float(z.argmax(1).eq(y).float().mean()))
            main = parametric[512]
            if hasattr(method, 'density_stats') and config['scenario']=='class_il':
                complete = bool(method.density_stats) and len(method.density_stats)==main.shape[1]
                density_scores = {}
                if complete:
                    original_metric = method.density_metric
                    for metric in ['ncm','diag']:
                        method.density_metric = metric
                        density_scores[metric] = method._density_scores(f, sorted(method.density_stats))
                    method.density_metric = original_metric
                for metric in ['ncm','diag']:
                    for alpha in readout_grid:
                        mixed = ((1-alpha)*_standardize(main)+alpha*_standardize(density_scores[metric])
                                 if complete and alpha > 0 else main)
                        out[f'readout_{metric}_{alpha:g}'].append(float(mixed.argmax(1).eq(y).float().mean()))
                if complete and method.density_alpha > 0:
                    main = (1-method.density_alpha)*_standardize(main)+method.density_alpha*_standardize(
                        density_scores[method.density_metric])
            out['main'].append(float(main.argmax(1).eq(y).float().mean()))
    return out, cache


def retrieval(method, loaders, cache, config):
    """Held-out test queries, fixed validation gallery; no test-test self matches."""
    import torch
    import torch.nn.functional as F
    val = [extract(method, x[1]) for x in loaders]
    cpt = 10//config['tasks']
    def global_labels(task, labels):
        return labels+task*cpt if config['scenario']=='task_il' else labels
    gy = torch.cat([global_labels(t,y) for t,(_,y) in enumerate(val)]).to(method.device)
    qy = torch.cat([global_labels(t,y) for t,(_,y) in enumerate(cache)]).to(method.device)
    gallery = torch.cat([f for f,_ in val]).to(method.device)
    queries = torch.cat([f for f,_ in cache]).to(method.device)
    output = {}
    for width in [32,64,128,256,512]:
        g = F.normalize(gallery[:,:width],dim=1).contiguous()
        q = F.normalize(queries[:,:width],dim=1).contiguous()
        hit1, hit5 = 0, 0
        if method.device.type=='cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for offset in range(0,len(q),256):
            labels = qy[offset:offset+256]
            sim = q[offset:offset+256]@g.T
            if config['scenario']=='task_il':
                sim.masked_fill_(labels[:,None]//cpt != gy[None,:]//cpt, -float('inf'))
            indices = sim.topk(min(5,len(g)),dim=1).indices
            matched = gy[indices].eq(labels[:,None])
            hit1 += int(matched[:,0].sum())
            hit5 += int(matched.any(dim=1).sum())
        if method.device.type=='cuda':
            torch.cuda.synchronize()
        output[str(width)] = dict(recall_at_1_pct=100*hit1/len(q), recall_at_5_pct=100*hit5/len(q),
            gallery_embedding_bytes=g.numel()*g.element_size(), queries=len(q), gallery=len(g),
            search_seconds=time.perf_counter()-start,
            timing_scope='one diagnostic pass, excludes extraction/normalization; not a latency benchmark')
    return output


def run_job(root, job_id, device='cuda:0', smoke=False):
    import numpy as np
    import torch
    from models import create_model
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    job = next(x for x in manifest['jobs'] if x['id']==job_id)
    config = dict(job['config'])
    if smoke:
        config.update(epochs=1, patience=1, workers=0, batch_size=8)
    out = root/('smoke' if smoke else 'runs')/job_id
    out.mkdir(parents=True, exist_ok=True)
    if (out/'result.json').exists():
        return
    torch.set_num_threads(2)
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = create_model('cifar10',10//config['tasks'],config['tasks'],config['scenario'])
    for task in range(config['tasks']):
        model.ensure_head(task)
    model.seen_upto = -1
    model_hash = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        model_hash.update(name.encode()); model_hash.update(parameter.numpy().tobytes())
    if config['method']=='mcl':
        keys = ['sdft_lambda','temperature','sdft_beta','density_alpha','density_metric',
                'density_shrinkage','weight_align','cosine_scale','granularities',
                'distill_widths','linear_cil','global_ce','freeze_old_heads']
        method = experimental_class()(model,torch.device(device),scenario=config['scenario'],
            lr=config['lr'], **{k:config[k] for k in keys})
    else:
        from SGD.sgd import SGDBaseline
        from LwF.lwf import LwF
        cls = SGDBaseline if config['method']=='sgd' else LwF
        method = cls(model,torch.device(device),scenario=config['scenario'],lr=config['lr'],
                     lwf_lambda=config['sdft_lambda'],temperature=config['temperature'])
    method.training_policy = dict(optimizer='adam',scheduler='none',weight_decay=0.)
    loaders, split_record = make_loaders(config,manifest['data_root'],smoke)
    atomic(out/'split.json', split_record)
    atomic(out/'config.json', dict(**job, resolved=config, smoke=smoke,
        initial_model_sha256=model_hash.hexdigest(), split_sha256=digest(out/'split.json')))
    alpha_grid = manifest['readout_grid']['alpha']
    started = time.time()
    with preserved_rng():
        baseline, _ = evaluate_stage(method,loaders,-1,config,alpha_grid)
    matrices = {k:[] for k in baseline}
    training_seconds, cache = [], None
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    for task,triple in enumerate(loaders):
        # Reset task RNG so extra diagnostics or different stopping epochs on an
        # earlier task do not change the next task's augmentation stream.
        task_seed = seed*1000+task
        random.seed(task_seed); np.random.seed(task_seed); torch.manual_seed(task_seed)
        if device.startswith('cuda'):
            torch.cuda.manual_seed_all(task_seed)
        start = time.perf_counter()
        history = method.train_task(task,triple[0],triple[1],config['epochs'],config['patience'],False)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        training_seconds.append(time.perf_counter()-start)
        with preserved_rng():
            rows,cache = evaluate_stage(method,loaders,task,config,alpha_grid)
        for key,row in rows.items():
            matrices[key].append(row)
        progress = dict(job_id=job_id,complete=False,completed_tasks=task+1,
            total_tasks=config['tasks'],latest_task_history=history,matrices=matrices,
            training_seconds=training_seconds,elapsed_seconds=time.time()-started)
        atomic(out/'progress.json',progress)
        # Each checkpoint keeps model and stored statistics, but not a redundant
        # teacher copy (at task boundaries the teacher is this exact model).
        state = dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                     seen_upto=model.seen_upto,config=config,task=task,
                     density_stats={k:tuple(v.cpu() for v in vv) for k,vv in
                                    getattr(method,'density_stats',{}).items()})
        torch.save(state,out/f'task_{task+1}.pt')
        print(json.dumps(dict(job_id=job_id,task=task+1,train_seconds=training_seconds[-1])),flush=True)
    with preserved_rng():
        retrieval_metrics = retrieval(method,loaders,cache,config)
    result = dict(job=job,config=config,complete=True,smoke=smoke,
        initial_model_sha256=model_hash.hexdigest(),split_sha256=digest(out/'split.json'),
        baseline=baseline,matrices=matrices,
        metrics={k:metric_dict(a,baseline[k]) for k,a in matrices.items()},
        retrieval=retrieval_metrics,history=method.history,training_seconds=training_seconds,
        elapsed_seconds=time.time()-started,device=str(device),
        gpu_name=torch.cuda.get_device_name() if device.startswith('cuda') else None,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None,
        density_state_bytes=sum(v.numel()*v.element_size() for pair in
                                getattr(method,'density_stats',{}).values() for v in pair),
        source_sha256=manifest['source_sha256'])
    atomic(out/'result.json',result)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def summarize(root):
    root=Path(root)
    per_seed, retrieval_rows, cost_rows, results = [], [], [], []
    for path in sorted((root/'runs').glob('*/result.json')):
        r=json.loads(path.read_text())
        if not r.get('complete') or r.get('smoke'):
            continue
        results.append(r)
        job=r['job']; c=r['config']
        base=dict(variant=job['variant'],family=job['family'],scenario=c['scenario'],seed=c['seed'])
        for view, metrics in r['metrics'].items():
            per_seed.append(dict(**base,view=view,**metrics))
        # These are exact inference interventions on the same learned weights.
        # They need no duplicate training jobs and must be labelled accordingly.
        if c['scenario']=='class_il' and job['variant'] in ('full','no_weight_alignment'):
            name=('no_density_readout' if job['variant']=='full' else 'no_density_no_weight_alignment')
            per_seed.append(dict(**{**base,'variant':name,'family':'readout_intervention'},view='main',
                **r['metrics']['readout_ncm_0']))
        for width, metrics in r['retrieval'].items():
            retrieval_rows.append(dict(**base,width=int(width),**metrics))
        cost_rows.append(dict(**base,training_seconds=sum(r['training_seconds']),
            elapsed_seconds=r['elapsed_seconds'],peak_allocated_bytes=r['peak_allocated_bytes'],
            density_state_bytes=r['density_state_bytes'],epochs=sum(h['epochs_run'] for h in r['history']),
            tasks_reaching_epoch_cap=sum(h['stop_reason']=='epoch_cap' for h in r['history'])))
    groups=defaultdict(list)
    for row in per_seed:
        groups[row['variant'],row['family'],row['scenario'],row['view']].append(row)
    summaries=[]
    for key,rows in groups.items():
        summary=dict(zip(['variant','family','scenario','view'],key),n=len(rows),
                     complete_three_seed=sorted(r['seed'] for r in rows)==[42,43,44])
        for metric in ['ACC','BWT','FWT','PS','AvgAcc','HARMONIC','AF']:
            v=[r[metric] for r in rows]
            summary[metric+'_mean']=statistics.mean(v)
            summary[metric+'_sd']=statistics.stdev(v) if len(v)>1 else None
        summaries.append(summary)
    paired=[]
    ref={(r['scenario'],r['seed'],r['view']):r for r in per_seed if r['variant']=='full'}
    for row in per_seed:
        reference=ref.get((row['scenario'],row['seed'],row['view']))
        if reference and row['variant']!='full':
            paired.append({**{k:row[k] for k in ['variant','family','scenario','seed','view']},
                           **{metric+'_delta':row[metric]-reference[metric]
                              for metric in ['ACC','BWT','FWT','PS']}})
    report=root/'report';report.mkdir(exist_ok=True)
    write_csv(report/'metrics_per_seed.csv',per_seed)
    write_csv(report/'metrics_summary.csv',summaries)
    write_csv(report/'retrieval_per_seed.csv',retrieval_rows)
    write_csv(report/'paired_differences.csv',paired)
    write_csv(report/'costs_per_seed.csv',cost_rows)
    for filename, rows, group_keys, metrics in [
        ('retrieval_summary.csv',retrieval_rows,['variant','family','scenario','width'],
         ['recall_at_1_pct','recall_at_5_pct','gallery_embedding_bytes','search_seconds']),
        ('paired_summary.csv',paired,['variant','family','scenario','view'],
         ['ACC_delta','BWT_delta','FWT_delta','PS_delta']),
        ('costs_summary.csv',cost_rows,['variant','family','scenario'],
         ['training_seconds','elapsed_seconds','peak_allocated_bytes','density_state_bytes','epochs','tasks_reaching_epoch_cap']),
    ]:
        grouped=defaultdict(list)
        for row in rows:
            grouped[tuple(row[k] for k in group_keys)].append(row)
        aggregate=[]
        for key,group in grouped.items():
            record=dict(zip(group_keys,key),n=len(group),
                        complete_three_seed=sorted(r['seed'] for r in group)==[42,43,44])
            for metric in metrics:
                values=[r[metric] for r in group if r[metric] is not None]
                record[metric+'_mean']=statistics.mean(values) if values else None
                record[metric+'_sd']=statistics.stdev(values) if len(values)>1 else None
            aggregate.append(record)
        write_csv(report/filename,aggregate)
    lines=[r'% Only completed three-seed groups. PS is displayed multiplied by 100.',
           r'\begin{tabular}{llrrrr}',r'\toprule',r'Variant & Scenario & ACC & BWT & FWT & PS \\',r'\midrule']
    for row in summaries:
        if row['view']!='main' or not row['complete_three_seed']:
            continue
        cells=[]
        for metric in ['ACC','BWT','FWT','PS']:
            factor=100 if metric=='PS' else 1
            cells.append(f"${factor*row[metric+'_mean']:.2f}_{{\\pm {factor*row[metric+'_sd']:.2f}}}$")
        lines.append(row['variant'].replace('_',r'\_')+' & '+row['scenario'].replace('_',r'\_')+' & '+' & '.join(cells)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (report/'ablation_table.tex').write_text('\n'.join(lines)+'\n')
    atomic(report/'completion.json',dict(completed_runs=len(results),complete_three_seed_views=
            sum(r['complete_three_seed'] for r in summaries)))
    complete=[r for r in summaries if r['complete_three_seed']]
    if complete:
        plot_report(report,complete)
    return len(results)


def plot_report(report, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'full':'#176b87','no_distillation':'#b45309','single_width':'#7c3aed',
            'single_width_no_distillation':'#6b7280','sgd':'#b91c1c','lwf':'#15803d'}
    for scenario in ['class_il','task_il']:
        fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
        plotted=False
        for variant,color in colors.items():
            rows=sorted((r for r in summaries if r['scenario']==scenario and r['variant']==variant
                         and r['view'].startswith('width_')),key=lambda r:int(r['view'].split('_')[1]))
            if not rows:
                continue
            plotted=True
            for axis,metric in zip(axes,['ACC','BWT']):
                axis.errorbar([int(r['view'].split('_')[1]) for r in rows],
                              [r[metric+'_mean'] for r in rows],yerr=[r[metric+'_sd'] for r in rows],
                              label=variant.replace('_',' '),color=color,marker='o',capsize=2)
                axis.set(xscale='log',xlabel='Embedding dimensions',ylabel=metric+(' (%)' if metric=='ACC' else ' (pp)'))
                axis.set_xticks([32,64,128,256,512],['32','64','128','256','512'])
                axis.grid(alpha=.2)
        if plotted:
            axes[0].legend(fontsize=7)
            fig.suptitle(scenario.replace('_','-')+': parametric readout, mean ± sample SD (3 seeds)',fontsize=10)
            for ext in ['pdf','png']:
                fig.savefig(report/f'width_accuracy_forgetting_{scenario}.{ext}',dpi=180)
        plt.close(fig)
    rows=[r for r in summaries if r['variant']=='full' and r['scenario']=='class_il'
          and r['view'].startswith('readout_')]
    if rows:
        fig,axis=plt.subplots(figsize=(5,3.4),layout='constrained')
        for metric in ['ncm','diag']:
            values=sorted([r for r in rows if r['view'].startswith('readout_'+metric+'_')],
                          key=lambda r:float(r['view'].split('_')[-1]))
            axis.errorbar([float(r['view'].split('_')[-1]) for r in values],
                          [r['ACC_mean'] for r in values],yerr=[r['ACC_sd'] for r in values],
                          label=metric,marker='o',capsize=2)
        axis.set(xlabel='Density mixing weight α',ylabel='Class-IL ACC (%)',
                 title='Full MCL3: post-hoc readout sensitivity')
        axis.legend();axis.grid(alpha=.2)
        for ext in ['pdf','png']:
            fig.savefig(report/f'density_readout_sweep.{ext}',dpi=180)
        plt.close(fig)


def live_pid(pid):
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:
        return False


def gpu_pids():
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return [int(x.strip()) for x in out.splitlines() if x.strip().isdigit()]


def campaign(root, share_gpu=False):
    import fcntl
    root=Path(root).resolve()
    lock=(root/'campaign.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((root/'manifest.json').read_text())
    for rel,expected in manifest['source_sha256'].items():
        if digest(root/'source'/rel)!=expected:
            raise RuntimeError('Immutable source snapshot changed: '+rel)
    failed=[]
    for job in manifest['jobs']:
        out=root/'runs'/job['id']
        if (out/'result.json').exists():
            continue
        while True:
            blockers=[pid for pid in manifest['wait_pids'] if live_pid(pid)]
            active=gpu_pids()
            if share_gpu or (not blockers and not active):
                break
            atomic(root/'status.json',dict(state='waiting_for_gpu',next_job=job['id'],
                blocking_campaign_pids=blockers,gpu_pids=active,total=len(manifest['jobs']),
                complete=summarize(root),updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
            time.sleep(30)
        out.mkdir(parents=True,exist_ok=True)
        atomic(root/'status.json',dict(state='running',job=job['id'],total=len(manifest['jobs']),
                                      complete=summarize(root),share_gpu=share_gpu))
        env={**os.environ,'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONUNBUFFERED':'1',
             'CUDA_VISIBLE_DEVICES':'0'}
        command=[sys.executable,str(root/'source/campaign/mcl_cifar10_ablation.py'),
                 'run','--root',str(root),'--job',job['id']]
        with (out/'run.log').open('a') as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
            returncode=process.wait()
        if returncode or not (out/'result.json').exists():
            failed.append(dict(job=job['id'],returncode=returncode,log=str(out/'run.log')))
            atomic(root/'failures.json',failed)
            # A broken harness must not burn through hundreds of jobs.
            atomic(root/'status.json',dict(state='failed',**failed[-1],complete=summarize(root)))
            raise RuntimeError('Job failed; inspect '+str(out/'run.log'))
        summarize(root)
    atomic(root/'status.json',dict(state='complete',complete=summarize(root),total=len(manifest['jobs'])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','campaign','summarize','status'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data-root',default=os.environ.get('GTEP_DATA_ROOT','./data'))
    parser.add_argument('--wait-pids',type=int,nargs='*',default=[])
    parser.add_argument('--job')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--share-gpu',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':
        prepare(args.root,args.data_root,args.wait_pids)
    elif args.action=='run':
        run_job(args.root,args.job,args.device,args.smoke)
    elif args.action=='campaign':
        campaign(args.root,args.share_gpu)
    elif args.action=='summarize':
        print(summarize(args.root))
    else:
        print((args.root/'status.json').read_text())


if __name__=='__main__':
    main()
