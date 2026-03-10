"""
Self-supervised pretraining for the RNA Plucker encoder.

Trains RNAPretrainModel with three objectives:
  MLM   masked language modelling on nucleotide tokens      (lambda_mlm)
  SS    per-token secondary-structure classification        (lambda_ss)
  MFE   sequence-level MFE scalar regression               (lambda_mfe)

After pretraining, the encoder weights (token_emb, pos_emb, blocks, ln_f,
pool_attn) are saved and can be loaded into train_utr.py via
--pretrained_backbone for fine-tuning on any downstream task.

Usage examples:

  # UTR-LM style (default): seq input only, structure only as SS/MFE targets:
  python pretrain_utr.py \\
      --sources "4.1_train.csv:utr" "4.4_train.csv:utr" "te_hek.csv:utr" \\
      --epochs 200 --output_dir pretrain_out

  # Structure-as-input style (bpp=mfe): structure also flows in as graph edges:
  python pretrain_utr.py \\
      --sources "4.1_train.csv:utr" \\
      --bpp_backend mfe --epochs 200

  # Fine-tune afterwards:
  python train_utr.py --task mrl --data 4.1_train.csv \\
      --pretrained_backbone pretrain_out/best_encoder.pt

Sources are specified as  path/to/file.csv:seq_col_name.
Multiple --sources arguments are accepted; sequences are deduplicated across all.
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from rna_structure_plucker import RNAPretrainModel, VOCAB_SIZE, N_EDGE_FEATS
from utr_datasets import BPPCache, PretrainDataset, collate_pretrain, kfold_indices
from train_utr import WarmupCosineScheduler   # reuse scheduler


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class PretrainConfig:
    # Data
    sources:         List[str] = field(default_factory=list)
    # Each source is "csv_path:seq_col" (colon-separated).
    exclude_sources: List[str] = field(default_factory=list)
    # Same format as sources.  Sequences from these CSVs are removed from the
    # pretraining corpus.  Pass your downstream test CSVs here for a clean eval.
    bpp_backend:     str   = 'zero'       # viennarna | mfe | zero
    # 'zero' = UTR-LM style: no structural edges as input; structure is only
    # a supervision signal (SS/MFE targets).  Use 'mfe' if you want structure
    # to also flow in as graph edges during pretraining (different regime).
    bpp_cache_dir: str   = 'bpp_cache'
    max_len:       Optional[int] = None
    mlm_prob:      float = 0.15
    aux_struct:    bool  = True         # add SS + MFE objectives
    deduplicate:   bool  = True

    # Model (must match fine-tune model architecture exactly)
    model_dim:     int   = 128
    num_layers:    int   = 6
    reduced_dim:   int   = 16
    ff_dim:        Optional[int] = None
    dropout:       float = 0.1
    pooling:       str   = 'attention'

    # Loss weights
    lambda_mlm:    float = 1.0
    lambda_ss:     float = 0.1
    lambda_mfe:    float = 0.01

    # Training
    epochs:        int   = 200
    batch_size:    int   = 256
    lr:            float = 3e-4
    weight_decay:  float = 1e-2
    clip_grad:     float = 1.0
    patience:      int   = 20           # early stop on val loss
    warmup_steps:  int   = 500
    val_frac:      float = 0.05

    # Speed
    use_amp:       bool  = True
    eval_every:    int   = 1

    # Runtime
    device:        str   = 'auto'
    num_workers:   int   = 4
    output_dir:    str   = 'pretrain_outputs'
    seed:          int   = 42
    resume_from:   Optional[str] = None


def _resolve_device(cfg: PretrainConfig) -> PretrainConfig:
    if cfg.device == 'auto':
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg


def _parse_sources(source_strs: List[str]) -> List[Tuple[str, str]]:
    """Convert ["path.csv:col", ...] -> [(path, col), ...]."""
    out = []
    for s in source_strs:
        if ':' not in s:
            raise ValueError(
                f"Source '{s}' must be in 'csv_path:seq_col' format."
            )
        path, col = s.rsplit(':', 1)
        out.append((path, col))
    return out


# ─── Dataset & model builders ─────────────────────────────────────────────────

def build_pretrain_dataset(cfg: PretrainConfig) -> PretrainDataset:
    cache    = BPPCache(cfg.bpp_cache_dir, backend=cfg.bpp_backend)
    sources  = _parse_sources(cfg.sources)
    excludes = _parse_sources(cfg.exclude_sources) if cfg.exclude_sources else None
    top_k    = 0 if cfg.bpp_backend == 'zero' else 4
    return PretrainDataset(
        sources          = sources,
        bpp_cache        = cache,
        max_len          = cfg.max_len,
        top_k_struct     = top_k,
        bp_threshold     = 0.05,
        mlm_prob         = cfg.mlm_prob,
        aux_struct       = cfg.aux_struct,
        deduplicate      = cfg.deduplicate,
        rng_seed         = cfg.seed,
        exclude_sources  = excludes,
    )


def build_pretrain_model(cfg: PretrainConfig) -> RNAPretrainModel:
    return RNAPretrainModel(
        vocab_size   = VOCAB_SIZE,
        model_dim    = cfg.model_dim,
        num_layers   = cfg.num_layers,
        reduced_dim  = cfg.reduced_dim,
        ff_dim       = cfg.ff_dim,
        n_edge_feats = N_EDGE_FEATS,
        dropout      = cfg.dropout,
        pooling      = cfg.pooling,
        lambda_mlm   = cfg.lambda_mlm,
        lambda_ss    = cfg.lambda_ss if cfg.aux_struct else 0.0,
        lambda_mfe   = cfg.lambda_mfe if cfg.aux_struct else 0.0,
    )


# ─── Training helpers ─────────────────────────────────────────────────────────

def pretrain_epoch(
    model:     RNAPretrainModel,
    loader:    DataLoader,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device:    torch.device,
    clip_grad: float,
    scaler:    Optional['torch.amp.GradScaler'],
) -> float:
    model.train()
    total_loss = 0.0
    use_amp    = scaler is not None

    for batch in loader:
        batch      = {k: v.to(device) for k, v in batch.items()}
        mlm_labels = batch.pop('mlm_labels', None)
        ss_labels  = batch.pop('ss_ids', None)
        mfe_labels = batch.pop('mfe', None)

        optimiser.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=use_amp):
            loss = model(
                **batch,
                mlm_labels = mlm_labels,
                ss_labels  = ss_labels,
                mfe_labels = mfe_labels,
            )

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
def pretrain_eval(
    model:  RNAPretrainModel,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Return mean total loss on the validation split."""
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch      = {k: v.to(device) for k, v in batch.items()}
        mlm_labels = batch.pop('mlm_labels', None)
        ss_labels  = batch.pop('ss_ids', None)
        mfe_labels = batch.pop('mfe', None)
        loss = model(
            **batch,
            mlm_labels = mlm_labels,
            ss_labels  = ss_labels,
            mfe_labels = mfe_labels,
        )
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def _save_pretrain_ckpt(
    path:            str,
    epoch:           int,
    model:           RNAPretrainModel,
    optimiser:       torch.optim.Optimizer,
    scheduler,
    scaler,
    best_val_loss:   float,
    best_epoch:      int,
    best_encoder_sd: Dict,
    no_improve:      int,
):
    torch.save({
        'epoch':           epoch,
        'state_dict':      model.state_dict(),
        'encoder_state_dict': best_encoder_sd,
        'optimizer':       optimiser.state_dict(),
        'scheduler':       scheduler.state_dict(),
        'scaler':          scaler.state_dict() if scaler else None,
        'best_val_loss':   best_val_loss,
        'best_epoch':      best_epoch,
        'no_improve':      no_improve,
    }, path)


# ─── Main training loop ───────────────────────────────────────────────────────

def run_pretrain(cfg: PretrainConfig):
    cfg = _resolve_device(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    resume_path  = os.path.join(cfg.output_dir, 'pretrain_resume.pt')
    best_path    = os.path.join(cfg.output_dir, 'best_encoder.pt')
    last_path    = os.path.join(cfg.output_dir, 'last_encoder.pt')
    cfg_path     = os.path.join(cfg.output_dir, 'pretrain_cfg.json')

    # Save config for reference
    with open(cfg_path, 'w') as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)

    device = torch.device(cfg.device)
    amp_ok = cfg.use_amp and cfg.device == 'cuda'

    print(f'Device:      {cfg.device}')
    print(f'Output dir:  {cfg.output_dir}')
    print(f'BPP backend: {cfg.bpp_backend}')
    print(f'Objectives:  MLM (lambda={cfg.lambda_mlm})', end='')
    if cfg.aux_struct:
        print(f' + SS (lambda={cfg.lambda_ss}) + MFE (lambda={cfg.lambda_mfe})')
    else:
        print()

    # ── Dataset split ─────────────────────────────────────────────────────────
    print('Loading dataset...')
    dataset = build_pretrain_dataset(cfg)
    n       = len(dataset)
    print(f'Sequences:   {n:,} (after dedup={cfg.deduplicate})')

    idx    = np.random.permutation(n)
    split  = int(n * (1 - cfg.val_frac))
    tr_idx = idx[:split]
    va_idx = idx[split:]

    train_loader = DataLoader(
        Subset(dataset, tr_idx),
        batch_size  = cfg.batch_size,
        shuffle     = True,
        collate_fn  = collate_pretrain,
        num_workers = cfg.num_workers,
        pin_memory  = (cfg.device != 'cpu'),
    )
    val_loader = DataLoader(
        Subset(dataset, va_idx),
        batch_size  = cfg.batch_size * 2,
        shuffle     = False,
        collate_fn  = collate_pretrain,
        num_workers = cfg.num_workers,
        pin_memory  = (cfg.device != 'cpu'),
    )

    # ── Model & optimiser ─────────────────────────────────────────────────────
    model    = build_pretrain_model(cfg).to(device)
    n_params = model.get_num_params()
    print(f'Model:       dim={cfg.model_dim} layers={cfg.num_layers} '
          f'r={cfg.reduced_dim} | {n_params:,} params')

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    total_steps = cfg.epochs * len(train_loader)
    sched  = WarmupCosineScheduler(opt, cfg.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda') if amp_ok else None

    best_val_loss   = np.inf
    best_epoch      = 0
    best_encoder_sd = model.get_encoder_state_dict()
    no_improve      = 0
    start_epoch     = 1

    # ── Resume ────────────────────────────────────────────────────────────────
    ckpt_to_load = cfg.resume_from or (resume_path if os.path.exists(resume_path) else None)
    if ckpt_to_load and os.path.exists(ckpt_to_load):
        ckpt = torch.load(ckpt_to_load, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        opt.load_state_dict(ckpt['optimizer'])
        sched.load_state_dict(ckpt['scheduler'])
        if scaler and ckpt.get('scaler'):
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch     = ckpt['epoch'] + 1
        best_val_loss   = ckpt['best_val_loss']
        best_epoch      = ckpt['best_epoch']
        best_encoder_sd = ckpt['encoder_state_dict']
        no_improve      = ckpt['no_improve']
        print(f'Resumed from epoch {ckpt["epoch"]} '
              f'(best_val_loss={best_val_loss:.4f} @ epoch {best_epoch})')

    print(f'\nTraining {len(tr_idx):,} / val {len(va_idx):,} | '
          f'eval_every={cfg.eval_every} | patience={cfg.patience}\n')

    # ── Training loop ─────────────────────────────────────────────────────────
    last_epoch = start_epoch - 1   # tracks the actual last epoch run
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0         = time.time()
        train_loss = pretrain_epoch(
            model, train_loader, opt, sched, device, cfg.clip_grad, scaler
        )
        elapsed    = time.time() - t0

        should_eval = (epoch % cfg.eval_every == 0) or (epoch == cfg.epochs)
        if should_eval:
            val_loss = pretrain_eval(model, val_loader, device)
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss   = val_loss
                best_epoch      = epoch
                best_encoder_sd = model.get_encoder_state_dict()
                no_improve      = 0
                # Save best encoder immediately
                torch.save(
                    {'encoder_state_dict': best_encoder_sd,
                     'epoch': epoch,
                     'val_loss': val_loss,
                     'cfg': dataclasses.asdict(cfg)},
                    best_path,
                )
            else:
                no_improve += 1

            mark = '*' if improved else ''
            print(f'  E{epoch:03d} train={train_loss:.4f} val={val_loss:.4f} '
                  f'[{elapsed:.1f}s] {mark}')

            if no_improve >= cfg.patience:
                print(f'  Early stop at epoch {epoch} '
                      f'(no improvement for {cfg.patience} evals)')
                _save_pretrain_ckpt(
                    resume_path, epoch, model, opt, sched, scaler,
                    best_val_loss, best_epoch, best_encoder_sd, no_improve,
                )
                break
        else:
            print(f'  E{epoch:03d} train={train_loss:.4f} [{elapsed:.1f}s]')

        last_epoch = epoch
        _save_pretrain_ckpt(
            resume_path, epoch, model, opt, sched, scaler,
            best_val_loss, best_epoch, best_encoder_sd, no_improve,
        )

    # ── Save last encoder ─────────────────────────────────────────────────────
    torch.save(
        {'encoder_state_dict': model.get_encoder_state_dict(),
         'epoch': last_epoch,
         'cfg': dataclasses.asdict(cfg)},
        last_path,
    )

    print(f'\nBest val loss: {best_val_loss:.4f} @ epoch {best_epoch}')
    print(f'Best encoder:  {best_path}')
    print(f'Last encoder:  {last_path}')


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> PretrainConfig:
    p = argparse.ArgumentParser(
        description='Self-supervised pretraining for the RNA Plucker encoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument('--sources', nargs='+', required=True,
                   metavar='CSV:COL',
                   help='One or more "csv_path:seq_col" sources. '
                        'Sequences are combined and deduplicated across all.')
    p.add_argument('--exclude_sources', nargs='+', default=[],
                   metavar='CSV:COL',
                   help='Held-out test/val CSVs (same "csv_path:seq_col" format) '
                        'whose sequences are removed from the pretraining corpus. '
                        'Pass your downstream test CSVs here to avoid data leakage.')
    p.add_argument('--bpp_backend',   default='zero',
                   choices=['viennarna', 'mfe', 'zero'],
                   help="'zero' (default) = UTR-LM style: no structural graph edges; "
                        "structure only as SS/MFE supervision targets. "
                        "'mfe'/'viennarna' = structure also flows in as graph edges.")
    p.add_argument('--bpp_cache_dir', default='bpp_cache')
    p.add_argument('--max_len',       type=int, default=None)
    p.add_argument('--mlm_prob',      type=float, default=0.15,
                   help='Fraction of tokens masked for MLM')
    p.add_argument('--no_aux_struct', action='store_true',
                   help='Disable SS and MFE objectives (MLM only)')
    p.add_argument('--no_dedup',      action='store_true',
                   help='Keep duplicate sequences across sources')
    # Model
    p.add_argument('--model_dim',   type=int,   default=128)
    p.add_argument('--num_layers',  type=int,   default=6)
    p.add_argument('--reduced_dim', type=int,   default=16)
    p.add_argument('--dropout',     type=float, default=0.1)
    p.add_argument('--pooling',     default='attention',
                   choices=['attention', 'mean'])
    # Loss weights
    p.add_argument('--lambda_mlm',  type=float, default=1.0)
    p.add_argument('--lambda_ss',   type=float, default=0.1)
    p.add_argument('--lambda_mfe',  type=float, default=0.01)
    # Training
    p.add_argument('--epochs',       type=int,   default=200)
    p.add_argument('--batch_size',   type=int,   default=256)
    p.add_argument('--lr',           type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--patience',     type=int,   default=20)
    p.add_argument('--warmup_steps', type=int,   default=500)
    p.add_argument('--val_frac',     type=float, default=0.05)
    # Speed
    p.add_argument('--no_amp',      action='store_true')
    p.add_argument('--eval_every',  type=int, default=1)
    # Runtime
    p.add_argument('--device',      default='auto')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--output_dir',  default='pretrain_outputs')
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--resume_from', default=None,
                   help='Path to pretrain_resume.pt to continue a run')

    args = p.parse_args()
    return PretrainConfig(
        sources          = args.sources,
        exclude_sources  = args.exclude_sources,
        bpp_backend      = args.bpp_backend,
        bpp_cache_dir = args.bpp_cache_dir,
        max_len       = args.max_len,
        mlm_prob      = args.mlm_prob,
        aux_struct    = not args.no_aux_struct,
        deduplicate   = not args.no_dedup,
        model_dim     = args.model_dim,
        num_layers    = args.num_layers,
        reduced_dim   = args.reduced_dim,
        dropout       = args.dropout,
        pooling       = args.pooling,
        lambda_mlm    = args.lambda_mlm,
        lambda_ss     = args.lambda_ss,
        lambda_mfe    = args.lambda_mfe,
        epochs        = args.epochs,
        batch_size    = args.batch_size,
        lr            = args.lr,
        weight_decay  = args.weight_decay,
        patience      = args.patience,
        warmup_steps  = args.warmup_steps,
        val_frac      = args.val_frac,
        use_amp       = not args.no_amp,
        eval_every    = args.eval_every,
        device        = args.device,
        num_workers   = args.num_workers,
        output_dir    = args.output_dir,
        seed          = args.seed,
        resume_from   = args.resume_from,
    )


if __name__ == '__main__':
    cfg = parse_args()
    run_pretrain(cfg)
