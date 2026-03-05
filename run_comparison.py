"""
UTR-LM comparison experiment runner.

Runs three controlled conditions on every benchmark so you can directly
compare the architectural choices.

Backbone is identical for all three: model_dim=128, num_layers=6, reduced_dim=16.
Only bpp_backend and aux_struct differ, keeping the comparison strictly controlled.

  A) seq_only   bpp=zero  aux_struct=False   Sequence-only baseline
                                             (no structure anywhere)

  B) utrlm      bpp=zero  aux_struct=True    Sequence input + SS/MFE targets
                                             ← mirrors what UTR-LM does
                                             lambda_ss=0.05, lambda_mfe=0.001
                                             epochs>=120, patience>=15

  C) plucker    bpp=mfe   aux_struct=False   Structure-as-graph-input (your model)
                                             ← no auxiliary supervision

WHY NOT  bpp=mfe + aux_struct=True?
  With structure already flowing in as graph edges the SS prediction task
  becomes much easier than the MLM-style SS task in UTR-LM (where the
  encoder never sees structure).  That condition conflates two differences
  (input modality AND training signal) making the comparison unclean.
  It is not run here. If you want it for an ablation you can add it manually.

Usage:
    python run_comparison.py                   # fresh run
    python run_comparison.py --resume_dir outputs/comparison_20250301_220000
    python run_comparison.py --tasks mrl te    # subset of benchmarks
    python run_comparison.py --conditions A B  # subset of conditions

Results are written to  outputs/comparison_<timestamp>/
Each experiment also writes a _resume.pt checkpoint so interrupted runs
can be continued with --resume_dir.
"""

import argparse
import os
import sys
import time
import json
import subprocess
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

# ─── PATHS ────────────────────────────────────────────────────────────────────

_HERE       = os.path.dirname(os.path.abspath(__file__))
CAPSULE_DIR = os.path.join(_HERE, 'capsule-4214075-data')

def _te(f):  return os.path.join(CAPSULE_DIR, 'TE_REL_Endogenous_Cao', f)
def _mrl(f): return os.path.join(CAPSULE_DIR, 'MRL_Random50Nuc_SynthesisLibrary_Sample', f)
def _exp(f): return os.path.join(CAPSULE_DIR, 'Experimental_Data', f)

MRL_LIBS = {
    'eGFP_unmod_1':    (_mrl('4.1_train_data_GSM3130435_egfp_unmod_1.csv'),    _mrl('4.1_test_data_GSM3130435_egfp_unmod_1.csv')),
    'eGFP_unmod_2':    (_mrl('4.4_train_data_GSM3130436_egfp_unmod_2.csv'),    _mrl('4.4_test_data_GSM3130436_egfp_unmod_2.csv')),
    'eGFP_pseudo_1':   (_mrl('4.7_train_data_GSM3130437_egfp_pseudo_1.csv'),   _mrl('4.7_test_data_GSM3130437_egfp_pseudo_1.csv')),
    'eGFP_pseudo_2':   (_mrl('4.10_train_data_GSM3130438_egfp_pseudo_2.csv'),  _mrl('4.10_test_data_GSM3130438_egfp_pseudo_2.csv')),
    'mCherry_1':       (_mrl('4.13_train_data_GSM3130441_mcherry_1.csv'),      _mrl('4.13_test_data_GSM3130441_mcherry_1.csv')),
    'mCherry_2':       (_mrl('4.16_train_data_GSM3130442_mcherry_2.csv'),      _mrl('4.16_test_data_GSM3130442_mcherry_2.csv')),
    'eGFP_m1pseudo_1': (_mrl('4.19_train_data_GSM3130439_egfp_m1pseudo_1.csv'),_mrl('4.19_test_data_GSM3130439_egfp_m1pseudo_1.csv')),
    'eGFP_m1pseudo_2': (_mrl('4.22_train_data_GSM3130440_egfp_m1pseudo_2.csv'),_mrl('4.22_test_data_GSM3130440_egfp_m1pseudo_2.csv')),
}
TE_FILES = {
    'HEK':    _te('HEK_sequence.csv'),
    'Muscle': _te('Muscle_sequence.csv'),
    'pc3':    _te('pc3_sequence.csv'),
}
RLU_CSV = _exp('Experimental_data_revised_label.csv')

# ─── HARDWARE ─────────────────────────────────────────────────────────────────

HAS_GPU     = torch.cuda.is_available()
DEVICE      = 'cuda' if HAS_GPU else 'cpu'
BATCH_SIZE  = 256 if HAS_GPU else 64
NUM_WORKERS = 4   if HAS_GPU else 2
BPP_CACHE_DIR = os.path.expanduser('~/bpp_cache')

# ─── THREE CONDITIONS ─────────────────────────────────────────────────────────

# Each condition is a letter (A/B/C), a short label, and the relevant flags.
# The comparison table is keyed by these labels.
#
# Backbone is identical across all three: model_dim=128, num_layers=6, reduced_dim=16
# Only bpp_backend and aux_struct vary, keeping the comparison strictly controlled.
#
# Condition B training overrides:
#   lambda_ss/mfe are smaller so aux heads don't swamp the primary MRL signal.
#   min_epochs=120 gives extra room since multitask convergence is slower; early
#   stopping will cut it short if the model plateaus before that.
CONDITIONS = {
    'A': dict(label='seq_only', bpp='zero', aux_struct=False,
              description='Sequence-only baseline (no structure, no aux)'),
    'B': dict(label='utrlm',   bpp='zero', aux_struct=True,
              description='UTR-LM style: seq input + SS/MFE targets, bpp=zero',
              lambda_ss=0.05, lambda_mfe=0.001,
              min_epochs=120, min_patience=15),
    'C': dict(label='plucker', bpp='mfe',  aux_struct=False,
              description='Your model: structure-as-graph-input, no aux targets'),
}

# ─── Experiment dataclass ─────────────────────────────────────────────────────

@dataclass
class Experiment:
    name:        str
    task:        str
    data:        str
    condition:   str            # 'A' | 'B' | 'C'
    benchmark:   str            # human-readable benchmark group key
    test_data:   Optional[str] = None
    folds:       int            = 1
    bpp:         str            = 'mfe'
    aux_struct:  bool           = False
    lambda_ss:   float          = 0.05
    lambda_mfe:  float          = 0.001
    epochs:      int            = 100
    patience:    int            = 15
    model_dim:   int            = 128
    num_layers:  int            = 6
    reduced_dim: int            = 16
    dropout:     float          = 0.1


# ─── Experiment builder ───────────────────────────────────────────────────────

def _triplet(
    benchmark:  str,
    task:       str,
    data:       str,
    conditions: List[str],
    test_data:  Optional[str] = None,
    folds:      int           = 1,
    epochs:     int           = 60,
    patience:   int           = 10,
    dropout:    float         = 0.1,
    folds_gpu:  Optional[int] = None,   # override folds when GPU is available
) -> List[Experiment]:
    """Create one Experiment per requested condition for a single benchmark."""
    actual_folds = (folds_gpu if folds_gpu and HAS_GPU else folds)
    exps = []
    for cond_key in conditions:
        cond = CONDITIONS[cond_key]
        # Apply per-condition epoch/patience floors (condition B needs more room)
        cond_epochs  = max(epochs,   cond.get('min_epochs',   0))
        cond_patience = max(patience, cond.get('min_patience', 0))
        exps.append(Experiment(
            name       = f'{benchmark}_{cond["label"]}',
            task       = task,
            data       = data,
            condition  = cond_key,
            benchmark  = benchmark,
            test_data  = test_data,
            folds      = actual_folds,
            bpp        = cond['bpp'],
            aux_struct = cond['aux_struct'],
            lambda_ss  = cond.get('lambda_ss',  0.05),
            lambda_mfe = cond.get('lambda_mfe', 0.001),
            epochs     = cond_epochs,
            patience   = cond_patience,
            dropout    = dropout,
        ))
    return exps


def build_experiment_list(
    run_tasks:      Optional[List[str]] = None,
    run_conditions: Optional[List[str]] = None,
) -> List[Experiment]:
    """
    Build the full experiment list.

    Args:
        run_tasks:      if given, only include these task types (mrl/te/el/rlu)
        run_conditions: if given, only include these condition keys (A/B/C)
    """
    conditions = run_conditions or list(CONDITIONS.keys())
    all_tasks  = run_tasks or ['mrl', 'te', 'rlu']  # el shares data with te; omit by default

    exps: List[Experiment] = []

    # ── MRL ──────────────────────────────────────────────────────────────────
    if 'mrl' in all_tasks:
        for lib_name, (train_csv, test_csv) in MRL_LIBS.items():
            if not os.path.exists(train_csv):
                print(f'  [skip] MRL {lib_name}: {train_csv} not found')
                continue
            exps += _triplet(
                benchmark = f'mrl_{lib_name}',
                task      = 'mrl',
                data      = train_csv,
                test_data = test_csv,
                folds     = 1,
                conditions= conditions,
                epochs    = 60,
                patience  = 10,
            )

    # ── TE ───────────────────────────────────────────────────────────────────
    if 'te' in all_tasks:
        for cell, csv_path in TE_FILES.items():
            if not os.path.exists(csv_path):
                print(f'  [skip] TE {cell}: {csv_path} not found')
                continue
            exps += _triplet(
                benchmark  = f'te_{cell}',
                task       = 'te',
                data       = csv_path,
                folds      = 5,       # 5-fold on CPU to keep runtime sane
                folds_gpu  = 10,      # 10-fold on GPU (paper protocol)
                conditions = conditions,
            )

    # ── EL ───────────────────────────────────────────────────────────────────
    if 'el' in all_tasks:
        for cell, csv_path in TE_FILES.items():
            if not os.path.exists(csv_path):
                continue
            exps += _triplet(
                benchmark  = f'el_{cell}',
                task       = 'el',
                data       = csv_path,
                folds      = 5,
                folds_gpu  = 10,
                conditions = conditions,
            )

    # ── RLU ──────────────────────────────────────────────────────────────────
    if 'rlu' in all_tasks:
        if not os.path.exists(RLU_CSV):
            print(f'  [skip] RLU: {RLU_CSV} not found')
        else:
            exps += _triplet(
                benchmark  = 'rlu',
                task       = 'rlu',
                data       = RLU_CSV,
                folds      = 5,
                conditions = conditions,
                dropout    = 0.2,
                epochs     = 150,
                patience   = 20,
            )

    return exps


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_experiment(exp: Experiment, output_dir: str,
                   resume_from: Optional[str] = None) -> Optional[Dict]:
    log_path    = os.path.join(output_dir, f'{exp.name}.log')
    exp_out_dir = os.path.join(output_dir, exp.name)
    eval_every  = 1

    cmd = [
        sys.executable, '-u', os.path.join(_HERE, 'train_utr.py'),
        '--task',         exp.task,
        '--data',         exp.data,
        '--bpp_backend',  exp.bpp,
        '--bpp_cache_dir',BPP_CACHE_DIR,
        '--folds',        str(exp.folds),
        '--epochs',       str(exp.epochs),
        '--patience',     str(exp.patience),
        '--batch_size',   str(BATCH_SIZE),
        '--num_workers',  str(NUM_WORKERS),
        '--eval_every',   str(eval_every),
        '--model_dim',    str(exp.model_dim),
        '--num_layers',   str(exp.num_layers),
        '--reduced_dim',  str(exp.reduced_dim),
        '--dropout',      str(exp.dropout),
        '--device',       DEVICE,
        '--output_dir',   exp_out_dir,
        '--seed',         '42',
    ]

    if exp.aux_struct:
        cmd += [
            '--aux_struct',
            '--lambda_ss',  str(exp.lambda_ss),
            '--lambda_mfe', str(exp.lambda_mfe),
        ]
    if exp.test_data:
        cmd += ['--test_data', exp.test_data]
    if resume_from and os.path.exists(resume_from):
        cmd += ['--resume_from', resume_from]

    cond_info = CONDITIONS[exp.condition]
    print(f'\n{"=" * 65}')
    print(f'  {exp.name}')
    print(f'  condition {exp.condition}: {cond_info["description"]}')
    print(f'  task={exp.task}  bpp={exp.bpp}  '
          f'aux_struct={exp.aux_struct}  folds={exp.folds}')
    if exp.test_data:
        print(f'  train: {os.path.basename(exp.data)}')
        print(f'  test:  {os.path.basename(exp.test_data)}')
    else:
        print(f'  data:  {os.path.basename(exp.data)}')
    print(f'{"=" * 65}')

    t0 = time.time()
    try:
        lines: List[str] = []
        with open(log_path, 'w') as log_fh:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_fh.write(line)
                log_fh.flush()
                lines.append(line)
            proc.wait()

        output  = ''.join(lines)
        elapsed = time.time() - t0
        print(f'  Finished in {elapsed / 60:.1f} min  ->  {log_path}')

        metrics = _parse_summary(output)
        metrics['elapsed_min'] = round(elapsed / 60, 1)
        return metrics

    except Exception as e:
        print(f'  ERROR: {e}')
        return None


def _parse_summary(output: str) -> Dict:
    """Extract metrics from training output.

    Priority:
    1. 'Test set:' line  (fixed hold-out, most reliable)
    2. Cross-validation summary block
    3. 'Best @' line     (validation metrics, fallback)
    """
    results = {}

    # 1) Prefer test-set line if present
    for line in output.splitlines():
        if line.strip().startswith('Test set:'):
            for part in line.split('|'):
                if '=' in part:
                    k, v = part.strip().split('=', 1)
                    try:
                        results[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
            return results

    # 2) CV summary block
    in_summary = False
    for line in output.splitlines():
        if 'Cross-validation summary' in line:
            in_summary = True
            continue
        if in_summary and ':' in line:
            k, v = line.split(':', 1)
            try:
                results[k.strip()] = float(v.split('+-')[0].split('±')[0].strip())
            except ValueError:
                pass
        elif in_summary and not line.strip():
            in_summary = False
    if results:
        return results

    # 3) Fall back to Best@ (val metrics)
    for line in output.splitlines():
        if line.strip().startswith('Best @'):
            for part in line.split('|'):
                if '=' in part:
                    k, v = part.strip().split('=', 1)
                    try:
                        results[k.strip()] = float(v.strip())
                    except ValueError:
                        pass
            break
    return results


# ─── Comparison summary table ─────────────────────────────────────────────────

def print_comparison_table(
    experiments:  List[Experiment],
    all_results:  Dict[str, Optional[Dict]],
):
    """
    Print a side-by-side comparison of all three conditions per benchmark.

    Metrics shown per task:
        mrl        -> spearman_r  AND  pearson_r  (two rows)
        te / el    -> spearman_r
        rlu        -> pearson_r
    """
    # List of (metric_key, label) tuples per task
    TASK_METRICS = {
        'mrl': [('spearman_r', 'spearman'), ('pearson_r', 'pearson')],
        'te':  [('spearman_r', 'spearman')],
        'el':  [('spearman_r', 'spearman')],
        'rlu': [('pearson_r',  'pearson')],
    }

    # Collect unique benchmarks in insertion order
    seen = {}
    for exp in experiments:
        if exp.benchmark not in seen:
            seen[exp.benchmark] = exp.task
    benchmarks = list(seen.items())

    cond_labels = [CONDITIONS[k]['label'] for k in sorted(CONDITIONS)]

    print('\n' + '=' * 72)
    print('  COMPARISON SUMMARY')
    print('  Backbone: model_dim=128  num_layers=6  reduced_dim=16 (same for A/B/C)')
    print('  Conditions:')
    for k, v in sorted(CONDITIONS.items()):
        print(f'    {k} [{v["label"]:<10}] {v["description"]}')
    print('=' * 72)

    col_w = 12
    print(f'\n  {"Benchmark":<32}' + ''.join(f'{lbl:>{col_w}}' for lbl in cond_labels))
    print('  ' + '-' * (32 + col_w * len(cond_labels)))

    for bench, task in benchmarks:
        metrics_list = TASK_METRICS.get(task, [('spearman_r', 'spearman')])
        for metric_key, metric_label in metrics_list:
            row_label = f'{bench} ({metric_label})'
            row = f'  {row_label:<32}'
            for k in sorted(CONDITIONS):
                exp_name = f'{bench}_{CONDITIONS[k]["label"]}'
                res = all_results.get(exp_name)
                if res is None:
                    val = 'n/a'
                else:
                    v = res.get(metric_key)
                    val = f'{v:.4f}' if v is not None else '---'
                row += f'{val:>{col_w}}'
            print(row)

    print('\n  ' + '-' * (32 + col_w * len(cond_labels)))
    print('  Elapsed times (min):')
    for bench, _ in benchmarks:
        row = f'  {bench:<32}'
        for k in sorted(CONDITIONS):
            exp_name = f'{bench}_{CONDITIONS[k]["label"]}'
            res = all_results.get(exp_name)
            val = f'{res["elapsed_min"]:.0f}' if res and 'elapsed_min' in res else '---'
            row += f'{val:>{col_w}}'
        print(row)

    print('=' * 72)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='UTR-LM comparison runner — three controlled conditions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--resume_dir', default=None,
                   help='Resume a previous run directory; completed experiments '
                        'are skipped, interrupted ones resume from checkpoint.')
    p.add_argument('--tasks', nargs='+', default=None,
                   choices=['mrl', 'te', 'el', 'rlu'],
                   help='Restrict to these task types (default: mrl te rlu)')
    p.add_argument('--conditions', nargs='+', default=None,
                   choices=['A', 'B', 'C'],
                   help='Restrict to these conditions (default: A B C)')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Output directory ──────────────────────────────────────────────────────
    if args.resume_dir:
        output_dir = os.path.abspath(args.resume_dir)
        if not os.path.isdir(output_dir):
            print(f'ERROR: --resume_dir {output_dir!r} does not exist.')
            sys.exit(1)
        print(f'Resuming run: {output_dir}')
    else:
        ts         = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(_HERE, 'outputs', f'comparison_{ts}')
        os.makedirs(output_dir, exist_ok=True)

    os.makedirs(BPP_CACHE_DIR, exist_ok=True)

    print(f'Device:     {DEVICE}')
    print(f'Batch size: {BATCH_SIZE}')
    print(f'BPP cache:  {BPP_CACHE_DIR}')
    print(f'Output dir: {output_dir}')
    print(f'Capsule:    {CAPSULE_DIR}')
    print()
    print('Conditions:')
    for k, v in sorted(CONDITIONS.items()):
        if not args.conditions or k in args.conditions:
            print(f'  {k} [{v["label"]}]  bpp={v["bpp"]}  '
                  f'aux_struct={v["aux_struct"]}')
            print(f'      {v["description"]}')
    print()

    experiments = build_experiment_list(
        run_tasks      = args.tasks,
        run_conditions = args.conditions,
    )

    # ── Load prior results ────────────────────────────────────────────────────
    summary_path = os.path.join(output_dir, 'results.json')
    all_results: Dict[str, Optional[Dict]] = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            all_results = json.load(f)
        done = sum(1 for v in all_results.values() if v is not None)
        print(f'Loaded existing results: {done}/{len(experiments)} done.')

    print(f'Planned: {len(experiments)} experiments')
    for exp in experiments:
        status = 'DONE' if all_results.get(exp.name) else 'pending'
        print(f'  [{status:>7}]  {exp.name}')
    print()

    # ── Run ───────────────────────────────────────────────────────────────────
    for exp in experiments:
        if all_results.get(exp.name) is not None:
            print(f'  [skip]  {exp.name}')
            continue

        exp_out_dir = os.path.join(output_dir, exp.name)
        resume_ckpt = os.path.join(exp_out_dir, f'{exp.task}_fold1_resume.pt')
        resume_from = resume_ckpt if os.path.exists(resume_ckpt) else None
        if resume_from:
            print(f'  [resume] {exp.name} — using {resume_ckpt}')

        result = run_experiment(exp, output_dir, resume_from=resume_from)
        all_results[exp.name] = result

        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_comparison_table(experiments, all_results)
    print(f'\nEnd time:     {datetime.now().strftime("%H:%M:%S")}')
    print(f'All logs in:  {output_dir}')
    print(f'Results JSON: {summary_path}')


if __name__ == '__main__':
    main()
