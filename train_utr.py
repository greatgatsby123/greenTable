"""
Unified training script for RNA 5'UTR prediction benchmarks.

Usage examples:

  # MRL regression (single library, MFE structure, lazy loading):
  python train_utr.py --task mrl --data eGFP_U1.csv --bpp_backend mfe

  # MRL cross-library (merged CSV with library column):
  python train_utr.py --task mrl --data all_libraries.csv --lib_col library

  # TE regression, HEK cell line, 10-fold CV + Spearman (paper protocol):
  python train_utr.py --task te --data te_data.csv --cell_line HEK --folds 10

  # IRES binary classification (AUPR):
  python train_utr.py --task ires --data ires_data.csv

  # Luciferase small-data test:
  python train_utr.py --task rlu --data luciferase.csv --folds 5

  # Quick sanity check with structure disabled (local edges only):
  python train_utr.py --task mrl --data eGFP_U1.csv --bpp_backend zero
"""

import argparse
import dataclasses
import os
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# LambdaLR calls step() in __init__ before any optimizer step, which triggers
# a spurious PyTorch warning.  Suppress it — the training loop order is correct.
warnings.filterwarnings(
    'ignore',
    message=r'Detected call of .lr_scheduler\.step\(\). before .optimizer\.step\(\)',
)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from rna_structure_plucker import RNAStructureGrassmann
from utr_datasets import (
    BPPCache, MRLDataset, TEDataset, IRESDataset, RLUDataset,
    NUM_LIBRARIES, collate_utr, compute_metrics,
    kfold_indices, stratified_kfold_indices,
)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # Data
    task:         str   = 'mrl'        # mrl | te | el | ires | rlu
    data:         str   = ''           # path to CSV
    bpp_backend:  str   = 'mfe'        # viennarna | mfe | zero
    bpp_cache_dir:str   = 'bpp_cache'
    seq_col:      Optional[str] = None # None → auto from task (see _auto_fill)
    label_col:    Optional[str] = None # None → auto from task
    lib_col:      Optional[str] = None # column with library name (MRL)
    cell_line:    Optional[str] = None # Muscle | PC3 | HEK (TE only)
    max_len:      Optional[int] = None # None → auto from task

    # Model
    model_dim:    int   = 128
    num_layers:   int   = 4
    reduced_dim:  int   = 32
    ff_dim:       Optional[int] = None # default: 4*model_dim
    dropout:      float = 0.1
    pooling:      str   = 'attention'

    # Training
    epochs:       int   = 60
    batch_size:   int   = 64
    lr:           float = 3e-4
    weight_decay: float = 1e-2
    clip_grad:    float = 1.0
    patience:     int   = 10          # early stopping on primary metric
    warmup_steps: int   = 200

    # Evaluation
    folds:        int   = 1           # 1 = single train/val split (80/20)
    val_frac:     float = 0.2         # used only when folds == 1
    stratify:     bool  = True        # stratified fold split for regression
    seed:         int   = 42

    # Speed / precision
    use_amp:      bool  = True         # mixed-precision (CUDA only; auto-disabled on CPU)
    eval_every:   int   = 5            # evaluate val set every N epochs (saves time on large data)

    # Runtime
    device:       str   = 'auto'
    num_workers:  int   = 2
    output_dir:   str   = 'outputs'
    save_best:    bool  = True
    test_data:    Optional[str] = None  # if set, use as fixed hold-out instead of CV
    resume_from:  Optional[str] = None  # path to a resume checkpoint to continue training


def _auto_fill(cfg: TrainConfig) -> TrainConfig:
    """Fill task-dependent defaults for fields left as None."""
    task_defaults = {
        # Column names match capsule-4214075-data files exactly
        'mrl':  dict(label_col='rl',                      seq_col='utr',                     max_len=50),
        'te':   dict(label_col='te_log',                  seq_col='utr',                     max_len=100),
        'el':   dict(label_col='rnaseq_log',              seq_col='utr',                     max_len=100),
        'ires': dict(label_col='label',                   seq_col='sequence',                max_len=None),
        'rlu':  dict(label_col='label',                   seq_col='utr_originial_varylength', max_len=50),
    }
    d = task_defaults[cfg.task]
    # Only fill fields the user did not explicitly set (sentinel = None).
    if cfg.seq_col   is None: cfg.seq_col   = d['seq_col']
    if cfg.label_col is None: cfg.label_col = d['label_col']
    if cfg.max_len   is None: cfg.max_len   = d['max_len']   # stays None for mrl/ires/rlu
    if cfg.device == 'auto':
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg


# ─── Dataset factory ──────────────────────────────────────────────────────────

def build_dataset(cfg: TrainConfig):
    """Instantiate the right dataset class for the given task."""
    cache = BPPCache(cfg.bpp_cache_dir, backend=cfg.bpp_backend)
    common = dict(
        bpp_cache=cache,
        top_k_struct=4,
        bp_threshold=0.05,
    )

    if cfg.task == 'mrl':
        return MRLDataset(
            cfg.data,
            seq_col=cfg.seq_col,
            label_col=cfg.label_col,
            lib_col=cfg.lib_col,
            lazy=True,                  # large dataset
            **common,
        )
    elif cfg.task in ('te', 'el'):
        return TEDataset(
            cfg.data,
            seq_col=cfg.seq_col,
            label_col=cfg.label_col,
            cell_filter=cfg.cell_line,
            max_len=cfg.max_len or 100,
            lazy=False,
            **common,
        )
    elif cfg.task == 'ires':
        return IRESDataset(
            cfg.data,
            seq_col=cfg.seq_col,
            label_col=cfg.label_col,
            lazy=False,
            **common,
        )
    elif cfg.task == 'rlu':
        return RLUDataset(
            cfg.data,
            seq_col=cfg.seq_col,
            label_col=cfg.label_col,
            **common,
        )
    else:
        raise ValueError(f'Unknown task: {cfg.task!r}')


# ─── Model factory ────────────────────────────────────────────────────────────

def build_model(cfg: TrainConfig) -> RNAStructureGrassmann:
    task_type = 'classification' if cfg.task == 'ires' else 'regression'
    num_libraries = NUM_LIBRARIES if cfg.task == 'mrl' and cfg.lib_col else 0

    return RNAStructureGrassmann(
        model_dim    = cfg.model_dim,
        num_layers   = cfg.num_layers,
        reduced_dim  = cfg.reduced_dim,
        ff_dim       = cfg.ff_dim,
        dropout      = cfg.dropout,
        pooling      = cfg.pooling,
        task         = task_type,
        num_libraries= num_libraries,
    )


# ─── Training helpers ─────────────────────────────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup followed by cosine decay."""

    def __init__(self, optimiser, warmup_steps: int, total_steps: int):
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        super().__init__(optimiser, lr_lambda)


def train_epoch(
    model: RNAStructureGrassmann,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    clip_grad: float = 1.0,
    scaler: Optional['torch.amp.GradScaler'] = None,
) -> float:
    model.train()
    total_loss = 0.0
    use_amp    = scaler is not None
    for batch in loader:
        batch   = {k: v.to(device) for k, v in batch.items()}
        labels  = batch.pop('labels', None)
        lib_ids = batch.pop('library_ids', None)

        optimiser.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            _, loss = model(**batch, labels=labels, library_ids=lib_ids)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimiser)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimiser.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: RNAStructureGrassmann,
    loader: DataLoader,
    device: torch.device,
    task: str = 'regression',
) -> Dict[str, float]:
    model.eval()
    all_preds:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for batch in loader:
        batch   = {k: v.to(device) for k, v in batch.items()}
        labels  = batch.pop('labels', None)
        lib_ids = batch.pop('library_ids', None)
        logits, _ = model(**batch, library_ids=lib_ids)
        all_preds.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds  = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    return compute_metrics(preds, labels, task=task)


# ─── Primary metric (for early stopping and fold ranking) ─────────────────────

def primary_metric(metrics: Dict[str, float], task: str) -> float:
    """Higher is always better (negate MSE for regression)."""
    if task == 'classification':
        return metrics['aupr']
    return metrics.get('spearman_r', metrics.get('pearson_r', -metrics['mse']))


# ─── Single fold training ─────────────────────────────────────────────────────

def train_fold(
    cfg:          TrainConfig,
    dataset,
    train_idx:    np.ndarray,
    val_idx:      np.ndarray,
    fold_num:     int = 1,
    val_dataset   = None,    # if given, use instead of Subset(dataset, val_idx)
    test_dataset  = None,    # evaluated ONCE after training; never used for selection
) -> Dict[str, float]:
    """
    Train one fold; return best-val or final-test metrics.

    Early stopping uses val_loader (either val_dataset or Subset(dataset,
    val_idx)).  If test_dataset is provided it is evaluated exactly once at
    the end with the best checkpoint — it is never seen during model selection.
    """
    device  = torch.device(cfg.device)
    task    = 'classification' if cfg.task == 'ires' else 'regression'

    train_ds = Subset(dataset, train_idx)
    val_ds   = val_dataset if val_dataset is not None else Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_utr, num_workers=cfg.num_workers,
        pin_memory=(cfg.device != 'cpu'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        collate_fn=collate_utr, num_workers=cfg.num_workers,
        pin_memory=(cfg.device != 'cpu'),
    )

    model    = build_model(cfg).to(device)
    n_params = model.get_num_params()

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    total_steps = cfg.epochs * len(train_loader)
    sched  = WarmupCosineScheduler(opt, cfg.warmup_steps, total_steps)
    scaler = (torch.amp.GradScaler('cuda')
              if cfg.use_amp and device.type == 'cuda' else None)

    best_score   = -np.inf
    best_state:  Optional[Dict] = None
    best_epoch   = 0
    best_metrics: Dict[str, float] = {}
    no_improve   = 0
    start_epoch  = 1

    os.makedirs(cfg.output_dir, exist_ok=True)
    resume_path = os.path.join(cfg.output_dir, f'{cfg.task}_fold{fold_num}_resume.pt')

    # ── Resume from checkpoint ────────────────────────────────────────────────
    ckpt_to_load = cfg.resume_from or (resume_path if os.path.exists(resume_path) else None)
    if ckpt_to_load and os.path.exists(ckpt_to_load):
        ckpt = torch.load(ckpt_to_load, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        opt.load_state_dict(ckpt['optimizer'])
        sched.load_state_dict(ckpt['scheduler'])
        if scaler and ckpt.get('scaler'):
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch  = ckpt['epoch'] + 1
        best_score   = ckpt['best_score']
        best_epoch   = ckpt.get('best_epoch', 0)
        best_metrics = ckpt['best_metrics']
        best_state   = ckpt['best_state']   # already on CPU
        no_improve   = ckpt['no_improve']
        print(f'  Resumed from epoch {ckpt["epoch"]} '
              f'(best_score={best_score:.4f} @ epoch {best_epoch})')

    val_size = len(val_dataset) if val_dataset is not None else len(val_idx)
    amp_tag  = 'AMP' if scaler else 'fp32'
    print(f'\n  Fold {fold_num} | {len(train_idx)} train / {val_size} val '
          f'| {n_params:,} params | {amp_tag} | eval_every={cfg.eval_every}')

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, opt, sched, device,
                                 cfg.clip_grad, scaler)
        elapsed    = time.time() - t0

        # ── Periodic evaluation & early stopping ─────────────────────────────
        should_eval = (epoch % cfg.eval_every == 0) or (epoch == cfg.epochs)
        if should_eval:
            metrics = evaluate(model, val_loader, device, task)
            score   = primary_metric(metrics, task)

            if score > best_score:
                best_score   = score
                best_epoch   = epoch
                best_metrics = metrics.copy()
                best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve  += 1

            m_str = ' | '.join(f'{k}={v:.4f}' for k, v in metrics.items())
            print(f'    E{epoch:03d} loss={train_loss:.4f} | {m_str} '
                  f'[{elapsed:.1f}s] {"*" if no_improve == 0 else ""}')

            if no_improve >= cfg.patience:
                print(f'    Early stop at epoch {epoch} '
                      f'(no improvement for {cfg.patience} evals)')
                # Save final resume checkpoint before exiting
                torch.save(_resume_ckpt(epoch, model, opt, sched, scaler,
                                        best_score, best_epoch, best_metrics,
                                        best_state, no_improve, fold_num),
                           resume_path)
                break
        else:
            # Non-eval epoch: always print so progress is visible
            print(f'    E{epoch:03d} loss={train_loss:.4f} [{elapsed:.1f}s]')

        # ── Save resume checkpoint after every epoch ──────────────────────────
        torch.save(_resume_ckpt(epoch, model, opt, sched, scaler,
                                best_score, best_epoch, best_metrics,
                                best_state, no_improve, fold_num),
                   resume_path)

    print(f'  Best @ epoch {best_epoch}: '
          + ' | '.join(f'{k}={v:.4f}' for k, v in best_metrics.items()))

    # ── Save best model weights ───────────────────────────────────────────────
    if cfg.save_best and best_state is not None:
        best_path = os.path.join(cfg.output_dir, f'{cfg.task}_fold{fold_num}_best.pt')
        torch.save({'state_dict': best_state, 'metrics': best_metrics, 'cfg': cfg},
                   best_path)
        print(f'  Saved → {best_path}')

    # ── Final test-set evaluation (once, not used for selection) ──────────────
    if test_dataset is not None and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test_loader = DataLoader(
            test_dataset, batch_size=cfg.batch_size * 2, shuffle=False,
            collate_fn=collate_utr, num_workers=cfg.num_workers,
            pin_memory=(cfg.device != 'cpu'),
        )
        test_metrics = evaluate(model, test_loader, device, task)
        t_str = ' | '.join(f'{k}={v:.4f}' for k, v in test_metrics.items())
        print(f'  Test set:  {t_str}')
        return test_metrics

    return best_metrics


def _resume_ckpt(epoch, model, opt, sched, scaler,
                 best_score, best_epoch, best_metrics, best_state, no_improve,
                 fold_num) -> Dict:
    """Pack the full training state needed to resume."""
    return {
        'epoch':        epoch,
        'state_dict':   model.state_dict(),
        'optimizer':    opt.state_dict(),
        'scheduler':    sched.state_dict(),
        'scaler':       scaler.state_dict() if scaler else None,
        'best_score':   best_score,
        'best_epoch':   best_epoch,
        'best_metrics': best_metrics,
        'best_state':   best_state,   # CPU tensors of best model
        'no_improve':   no_improve,
        'fold_num':     fold_num,
    }


# ─── Cross-validation orchestrator ───────────────────────────────────────────

def run_cv(cfg: TrainConfig):
    """Full cross-validation run; prints aggregate statistics."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f'Task: {cfg.task} | Data: {cfg.data} | Device: {cfg.device}')
    print(f'Model: dim={cfg.model_dim} layers={cfg.num_layers} r={cfg.reduced_dim}')
    print(f'BPP backend: {cfg.bpp_backend} | Folds: {cfg.folds}')

    dataset = build_dataset(cfg)
    n       = len(dataset)
    task    = 'classification' if cfg.task == 'ires' else 'regression'
    print(f'Dataset: {n} sequences | task={task}')

    # Build fold splits
    hold_out_ds: Optional[object] = None   # test set, evaluated once at the end
    if cfg.test_data is not None:
        # Fixed hold-out protocol (e.g. MRL pre-split libraries):
        #   • Carve a val split from the TRAINING data for early stopping.
        #   • The test CSV is only evaluated once, after training, with the
        #     best checkpoint — it never touches model selection.
        test_cfg    = dataclasses.replace(cfg, data=cfg.test_data)
        hold_out_ds = build_dataset(test_cfg)
        idx   = np.random.permutation(n)
        split = int(n * (1 - cfg.val_frac))   # e.g. 90 % train, 10 % val
        folds = [(idx[:split], idx[split:])]
        val_datasets = [None]                  # use Subset(dataset, val_idx)
        print(f'Hold-out split: {split} train / {n - split} val '
              f'(val from train CSV, test CSV evaluated once at end)')
    elif cfg.folds == 1:
        # Single 80/20 random split
        idx     = np.random.permutation(n)
        split   = int(n * (1 - cfg.val_frac))
        folds   = [(idx[:split], idx[split:])]
        val_datasets = [None]
    elif cfg.stratify and task == 'regression':
        labels = np.array([dataset[i]['label'] for i in range(n)])
        folds  = stratified_kfold_indices(labels, k=cfg.folds, seed=cfg.seed)
        val_datasets = [None] * len(folds)
    else:
        folds  = kfold_indices(n, k=cfg.folds, seed=cfg.seed)
        val_datasets = [None] * len(folds)

    all_metrics: List[Dict[str, float]] = []
    for fold_i, ((tr_idx, va_idx), val_ds) in enumerate(zip(folds, val_datasets)):
        metrics = train_fold(cfg, dataset, tr_idx, va_idx,
                             fold_num=fold_i + 1, val_dataset=val_ds,
                             test_dataset=hold_out_ds)
        all_metrics.append(metrics)

    # Aggregate
    if len(all_metrics) > 1:
        print('\n── Cross-validation summary ' + '─' * 30)
        for k in all_metrics[0]:
            vals = [m[k] for m in all_metrics]
            print(f'  {k:>12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(
        description='Train structure-edge Plücker model on UTR benchmarks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument('--task',         default='mrl',
                   choices=['mrl', 'te', 'el', 'ires', 'rlu'])
    p.add_argument('--data',         required=True,
                   help='Path to input CSV file')
    p.add_argument('--bpp_backend',  default='mfe',
                   choices=['viennarna', 'mfe', 'zero'],
                   help='Structure computation: viennarna (slow/accurate), '
                        'mfe (fast/binary), zero (ablation — local edges only)')
    p.add_argument('--bpp_cache_dir',default='bpp_cache')
    p.add_argument('--seq_col',      default=None,
                   help='Sequence column name (default: auto per task)')
    p.add_argument('--label_col',    default=None,
                   help='Label column name (default: auto per task)')
    p.add_argument('--lib_col',      default=None,
                   help='CSV column with library name (MRL cross-library)')
    p.add_argument('--cell_line',    default=None,
                   help='Filter TE data to one cell line (Muscle/PC3/HEK)')
    p.add_argument('--max_len',      type=int, default=None,
                   help='Truncate sequences to this length')
    p.add_argument('--test_data',    default=None,
                   help='Fixed hold-out test CSV; if set, trains on --data and '
                        'evaluates on --test_data instead of doing CV')
    # Model
    p.add_argument('--model_dim',    type=int,   default=128)
    p.add_argument('--num_layers',   type=int,   default=4)
    p.add_argument('--reduced_dim',  type=int,   default=32)
    p.add_argument('--dropout',      type=float, default=0.1)
    p.add_argument('--pooling',      default='attention',
                   choices=['attention', 'mean'])
    # Training
    p.add_argument('--epochs',       type=int,   default=60)
    p.add_argument('--batch_size',   type=int,   default=64)
    p.add_argument('--lr',           type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--patience',     type=int,   default=10)
    p.add_argument('--warmup_steps', type=int,   default=200)
    # Evaluation
    p.add_argument('--folds',        type=int,   default=1,
                   help='Number of CV folds (1 = single 80/20 split)')
    p.add_argument('--val_frac',     type=float, default=0.2)
    p.add_argument('--no_stratify',  action='store_true',
                   help='Disable stratified k-fold for regression tasks')
    p.add_argument('--seed',         type=int,   default=42)
    # Speed / precision
    p.add_argument('--no_amp',       action='store_true',
                   help='Disable mixed-precision (AMP); use full fp32')
    p.add_argument('--eval_every',   type=int, default=5,
                   help='Evaluate validation set every N epochs')
    # Runtime
    p.add_argument('--device',       default='auto')
    p.add_argument('--num_workers',  type=int,   default=2)
    p.add_argument('--output_dir',   default='outputs')
    p.add_argument('--no_save',      action='store_true',
                   help='Do not save checkpoints')
    p.add_argument('--resume_from',  default=None,
                   help='Path to a _resume.pt checkpoint to continue training')

    args = p.parse_args()
    cfg  = TrainConfig(
        task         = args.task,
        data         = args.data,
        bpp_backend  = args.bpp_backend,
        bpp_cache_dir= args.bpp_cache_dir,
        seq_col      = args.seq_col,
        lib_col      = args.lib_col,
        cell_line    = args.cell_line,
        max_len      = args.max_len,
        model_dim    = args.model_dim,
        num_layers   = args.num_layers,
        reduced_dim  = args.reduced_dim,
        dropout      = args.dropout,
        pooling      = args.pooling,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        patience     = args.patience,
        warmup_steps = args.warmup_steps,
        use_amp      = not args.no_amp,
        eval_every   = args.eval_every,
        folds        = args.folds,
        val_frac     = args.val_frac,
        stratify     = not args.no_stratify,
        seed         = args.seed,
        device       = args.device,
        num_workers  = args.num_workers,
        output_dir   = args.output_dir,
        save_best    = not args.no_save,
        test_data    = args.test_data,
        resume_from  = args.resume_from,
    )
    if args.label_col:
        cfg.label_col = args.label_col
    return _auto_fill(cfg)


if __name__ == '__main__':
    cfg = parse_args()
    run_cv(cfg)
