"""
Overnight experiment runner — runs all UTR benchmark experiments sequentially,
logs results to timestamped files, and prints a summary table at the end.

Datasets are read from  capsule-4214075-data/  (assumed to be a sibling of
this script).  Edit CAPSULE_DIR below if your layout differs.

    python run_overnight.py                        # fresh run
    python run_overnight.py --resume_dir outputs/overnight_20250101_220000

The BPP cache is written to  outputs/bpp_cache/  (shared across runs so it
only needs to be computed once).  BPPs are computed lazily by DataLoader
workers during the first epoch — no pre-warming step.

Results are written to  outputs/overnight_<timestamp>/
Each experiment saves a  *_fold1_resume.pt  checkpoint after every epoch so
training can be resumed automatically if the run is interrupted.
"""

import argparse
import os
import sys
import time
import subprocess
import json
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import torch
import numpy as np

# ─── PATHS ────────────────────────────────────────────────────────────────────

# Root of the capsule data folder (relative to this script's directory).
_HERE       = os.path.dirname(os.path.abspath(__file__))
CAPSULE_DIR = os.path.join(_HERE, 'capsule-4214075-data')

# Convenience path builders
def _te(fname):   return os.path.join(CAPSULE_DIR, 'TE_REL_Endogenous_Cao', fname)
def _mrl(fname):  return os.path.join(CAPSULE_DIR, 'MRL_Random50Nuc_SynthesisLibrary_Sample', fname)
def _exp(fname):  return os.path.join(CAPSULE_DIR, 'Experimental_Data', fname)


# ─── MRL LIBRARY CATALOGUE ────────────────────────────────────────────────────
#
# 8 GEO libraries, each with a fixed train / test CSV split.
# Column names: utr (sequence), rl (mean ribosome load label).
#
# The numbered prefix (4.X) corresponds to figure panels in the source paper.
# We use the first replicate per library (4.1, 4.4, 4.7, … ) as the canonical
# experiment; the second replicate can be uncommented to train all 16 files.

MRL_LIBS = {
    # name                   train CSV                                          test CSV
    'eGFP_unmod_1':    (_mrl('4.1_train_data_GSM3130435_egfp_unmod_1.csv'),   _mrl('4.1_test_data_GSM3130435_egfp_unmod_1.csv')),
    'eGFP_unmod_2':    (_mrl('4.4_train_data_GSM3130436_egfp_unmod_2.csv'),   _mrl('4.4_test_data_GSM3130436_egfp_unmod_2.csv')),
    'eGFP_pseudo_1':   (_mrl('4.7_train_data_GSM3130437_egfp_pseudo_1.csv'),  _mrl('4.7_test_data_GSM3130437_egfp_pseudo_1.csv')),
    'eGFP_pseudo_2':   (_mrl('4.10_train_data_GSM3130438_egfp_pseudo_2.csv'), _mrl('4.10_test_data_GSM3130438_egfp_pseudo_2.csv')),
    'mCherry_1':       (_mrl('4.13_train_data_GSM3130441_mcherry_1.csv'),     _mrl('4.13_test_data_GSM3130441_mcherry_1.csv')),
    'mCherry_2':       (_mrl('4.16_train_data_GSM3130442_mcherry_2.csv'),     _mrl('4.16_test_data_GSM3130442_mcherry_2.csv')),
    'eGFP_m1pseudo_1': (_mrl('4.19_train_data_GSM3130439_egfp_m1pseudo_1.csv'),_mrl('4.19_test_data_GSM3130439_egfp_m1pseudo_1.csv')),
    'eGFP_m1pseudo_2': (_mrl('4.22_train_data_GSM3130440_egfp_m1pseudo_2.csv'),_mrl('4.22_test_data_GSM3130440_egfp_m1pseudo_2.csv')),
}

# ─── TE / EL FILES (per-cell-line CSVs) ──────────────────────────────────────
#
# Columns:  utr (sequence), te_log (log TE), rnaseq_log (log expression level)
# Evaluation: 10-fold CV + Spearman ρ (paper protocol).

TE_FILES = {
    'HEK':    _te('HEK_sequence.csv'),
    'Muscle': _te('Muscle_sequence.csv'),
    'pc3':    _te('pc3_sequence.csv'),
}

# ─── RLU (EXPERIMENTAL VALIDATION SET) ───────────────────────────────────────
#
# Experimental_data_revised_label.csv: N=211 designed UTRs with normalised
# luciferase expression (label column, range 0.48–1.33).
# Columns: utr_originial_varylength (sequence), label.

RLU_CSV = _exp('Experimental_data_revised_label.csv')

# ─── HARDWARE-ADAPTIVE SETTINGS ───────────────────────────────────────────────

HAS_GPU    = torch.cuda.is_available()
DEVICE     = 'cuda' if HAS_GPU else 'cpu'
BATCH_SIZE = 256  if HAS_GPU else 64
NUM_WORKERS= 4    if HAS_GPU else 2

# BPP cache is shared across ALL runs (stable path, not timestamped).
# This means BPPs computed in one run are reused in the next.
BPP_CACHE_DIR = os.path.expanduser('~/bpp_cache')

# ─── EXPERIMENT CATALOGUE ─────────────────────────────────────────────────────

@dataclass
class Experiment:
    name:        str
    task:        str           # mrl | te | el | rlu
    data:        str           # path to train CSV
    test_data:   Optional[str] = None   # fixed hold-out (MRL)
    folds:       int           = 1
    cell_line:   Optional[str] = None
    bpp:         str           = 'mfe'  # mfe | viennarna | zero
    epochs:      int           = 100
    patience:    int           = 15
    model_dim:   int           = 128
    num_layers:  int           = 4
    reduced_dim: int           = 32
    dropout:     float         = 0.1


def build_experiment_list() -> List[Experiment]:
    exps: List[Experiment] = []

    # ── MRL: one experiment per library, fixed train / test split ─────────────
    for lib_name, (train_csv, test_csv) in MRL_LIBS.items():
        if not os.path.exists(train_csv):
            print(f'  [skip] MRL {lib_name}: {train_csv} not found')
            continue
        exps.append(Experiment(
            name      = f'mrl_{lib_name}_mfe',
            task      = 'mrl',
            data      = train_csv,
            test_data = test_csv,
            folds     = 1,          # train_utr will use test_data as hold-out
            bpp       = 'mfe',
            epochs    = 60,         # MRL ~260k rows: fewer epochs needed
            patience  = 10,
        ))

    # ── TE / EL: 3 cell lines × 2 metrics, 10-fold CV ────────────────────────
    for cell, csv_path in TE_FILES.items():
        if not os.path.exists(csv_path):
            print(f'  [skip] TE {cell}: {csv_path} not found')
            continue
        for label in ('te', 'el'):
            exps.append(Experiment(
                name    = f'{label}_{cell}_mfe',
                task    = label,
                data    = csv_path,
                folds   = 10 if HAS_GPU else 5,
                bpp     = 'mfe',
            ))

    # Key ablation: structure vs local-only on HEK TE
    if os.path.exists(TE_FILES.get('HEK', '')):
        exps.append(Experiment(
            name  = 'te_HEK_zero',
            task  = 'te',
            data  = TE_FILES['HEK'],
            folds = 10 if HAS_GPU else 5,
            bpp   = 'zero',
        ))

    # ── RLU: 5-fold CV on N=211 designed UTRs ────────────────────────────────
    if os.path.exists(RLU_CSV):
        exps.append(Experiment(
            name    = 'rlu_mfe',
            task    = 'rlu',
            data    = RLU_CSV,
            folds   = 5,
            bpp     = 'mfe',
            dropout = 0.2,   # stronger regularisation for N=211
            epochs  = 150,
            patience= 20,
        ))

    return exps


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_experiment(exp: Experiment, output_dir: str,
                   resume_from: Optional[str] = None) -> Optional[Dict]:
    """
    Call train_utr.py for one experiment via subprocess.
    Captures stdout; parses the final CV summary line.
    Returns a dict of metric_name → value or None on failure.
    """
    log_path    = os.path.join(output_dir, f'{exp.name}.log')
    exp_out_dir = os.path.join(output_dir, exp.name)

    # eval_every: use 1 for small datasets (RLU), 5 for large (MRL/TE)
    eval_every = 1 if exp.task == 'rlu' else 5

    cmd = [
        sys.executable, '-u', os.path.join(_HERE, 'train_utr.py'),
        '--task',        exp.task,
        '--data',        exp.data,
        '--bpp_backend', exp.bpp,
        '--bpp_cache_dir', BPP_CACHE_DIR,   # stable shared cache
        '--folds',       str(exp.folds),
        '--epochs',      str(exp.epochs),
        '--patience',    str(exp.patience),
        '--batch_size',  str(BATCH_SIZE),
        '--num_workers', str(NUM_WORKERS),
        '--eval_every',  str(eval_every),
        '--model_dim',   str(exp.model_dim),
        '--num_layers',  str(exp.num_layers),
        '--reduced_dim', str(exp.reduced_dim),
        '--dropout',     str(exp.dropout),
        '--device',      DEVICE,
        '--output_dir',  exp_out_dir,
        '--seed',        '42',
    ]

    if exp.test_data:
        cmd += ['--test_data', exp.test_data]
    if exp.cell_line:
        cmd += ['--cell_line', exp.cell_line]
    if resume_from and os.path.exists(resume_from):
        cmd += ['--resume_from', resume_from]

    print(f'\n{"="*60}')
    print(f'  {exp.name}')
    print(f'  task={exp.task}  bpp={exp.bpp}  folds={exp.folds}')
    if exp.test_data:
        print(f'  train: {os.path.basename(exp.data)}')
        print(f'  test:  {os.path.basename(exp.test_data)}')
    else:
        print(f'  data:  {os.path.basename(exp.data)}')
    print(f'{"="*60}')

    t0 = time.time()
    try:
        lines: list[str] = []
        with open(log_path, 'w') as log_fh:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,          # line-buffered on the pipe side
            )
            for line in proc.stdout:    # streams in real-time
                sys.stdout.write(line)
                sys.stdout.flush()
                log_fh.write(line)
                log_fh.flush()
                lines.append(line)
            proc.wait()

        output  = ''.join(lines)
        elapsed = time.time() - t0
        print(f'  Finished in {elapsed/60:.1f} min → {log_path}')

        metrics = _parse_summary(output)
        metrics['elapsed_min'] = round(elapsed / 60, 1)
        return metrics

    except Exception as e:
        print(f'  ERROR: {e}')
        return None


def _parse_summary(output: str) -> Dict:
    """Extract mean metric values from the 'Cross-validation summary' section."""
    results = {}
    in_summary = False
    for line in output.splitlines():
        if 'Cross-validation summary' in line:
            in_summary = True
            continue
        if in_summary and ':' in line:
            parts = line.strip().split(':')
            if len(parts) == 2:
                key   = parts[0].strip()
                value = parts[1].strip()
                try:
                    mean_val = float(value.split('±')[0].strip())
                    results[key] = mean_val
                except ValueError:
                    pass
        elif in_summary and line.strip() == '':
            in_summary = False

    # For fixed hold-out (no CV summary), parse the "Best @" line instead
    if not results:
        for line in output.splitlines():
            if line.strip().startswith('Best @'):
                for part in line.split('|'):
                    kv = part.strip()
                    if '=' in kv:
                        k, v = kv.split('=', 1)
                        try:
                            results[k.strip()] = float(v.strip())
                        except ValueError:
                            pass
    return results


# ─── Summary table ────────────────────────────────────────────────────────────

def print_summary_table(results: Dict[str, Optional[Dict]]):
    print('\n' + '═' * 75)
    print('  OVERNIGHT RESULTS SUMMARY')
    print('═' * 75)
    header_metrics = ['spearman_r', 'pearson_r', 'r2', 'mse', 'aupr', 'elapsed_min']
    print(f'  {"Experiment":<35}' + ''.join(f'{m:>10}' for m in header_metrics))
    print('  ' + '─' * 73)
    for name, res in results.items():
        if res is None:
            print(f'  {name:<35}  FAILED')
            continue
        vals = ''.join(
            f'{res[m]:>10.4f}' if m in res else f'{"—":>10}'
            for m in header_metrics
        )
        print(f'  {name:<35}{vals}')
    print('═' * 75)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Overnight UTR benchmark runner')
    p.add_argument('--resume_dir', default=None,
                   help='Path to a previous overnight output directory to resume. '
                        'Completed experiments (present in results.json) are skipped; '
                        'the first incomplete experiment resumes from its latest '
                        '_resume.pt checkpoint.')
    return p.parse_args()


def main():
    args = parse_args()

    # ── Determine output directory ────────────────────────────────────────────
    if args.resume_dir:
        output_dir = os.path.abspath(args.resume_dir)
        if not os.path.isdir(output_dir):
            print(f'ERROR: --resume_dir {output_dir!r} does not exist.')
            sys.exit(1)
        print(f'Resuming run in: {output_dir}')
    else:
        timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(_HERE, 'outputs', f'overnight_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)

    print(f'Device:      {DEVICE}')
    print(f'Batch size:  {BATCH_SIZE}')
    print(f'BPP cache:   {BPP_CACHE_DIR}')
    print(f'Output dir:  {output_dir}')
    print(f'Capsule:     {CAPSULE_DIR}')
    print(f'Start time:  {datetime.now().strftime("%H:%M:%S")}')
    print()

    experiments = build_experiment_list()

    # ── Load prior results if resuming ────────────────────────────────────────
    summary_path = os.path.join(output_dir, 'results.json')
    all_results: Dict[str, Optional[Dict]] = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            all_results = json.load(f)
        done = [k for k, v in all_results.items() if v is not None]
        print(f'Loaded existing results: {len(done)} experiments already done.')

    print(f'Planned {len(experiments)} experiments total:')
    for exp in experiments:
        status = 'DONE' if all_results.get(exp.name) is not None else 'pending'
        print(f'  [{status:>7}] {exp.name}')
    print()

    # ── Run experiments ───────────────────────────────────────────────────────
    # BPPs are computed lazily by DataLoader workers; no pre-warm step.
    # The shared BPP_CACHE_DIR means any sequence computed in a prior run
    # (or earlier in this run) is immediately reused.
    os.makedirs(BPP_CACHE_DIR, exist_ok=True)

    for exp in experiments:
        # Skip already-completed experiments
        if all_results.get(exp.name) is not None:
            print(f'  [skip] {exp.name} (already completed)')
            continue

        # Check for a resume checkpoint from an interrupted run
        exp_out_dir   = os.path.join(output_dir, exp.name)
        resume_ckpt   = os.path.join(exp_out_dir,
                                     f'{exp.task}_fold1_resume.pt')
        resume_from   = resume_ckpt if os.path.exists(resume_ckpt) else None
        if resume_from:
            print(f'  [resume] {exp.name} — found checkpoint {resume_ckpt}')

        result = run_experiment(exp, output_dir, resume_from=resume_from)
        all_results[exp.name] = result

        # Save after each experiment so a subsequent --resume_dir skips it
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    print_summary_table(all_results)
    print(f'\nEnd time:     {datetime.now().strftime("%H:%M:%S")}')
    print(f'All logs in:  {output_dir}')
    print(f'Results JSON: {summary_path}')


if __name__ == '__main__':
    main()
