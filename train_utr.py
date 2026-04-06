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

  # UTR-LM comparison: add SS + MFE auxiliary supervision heads:
  python train_utr.py --task mrl --data eGFP_U1.csv --aux_struct
  python train_utr.py --task mrl --data eGFP_U1.csv --aux_struct --lambda_ss 0.2 --lambda_mfe 0.05
"""

import argparse
import dataclasses
import json
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

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
from rna_bender import RNABenderModel, VOCAB_SIZE
from rna_baseline import RNATransformerBaseline
from rna_moe_mrl import RNAMoEMRLModel
from rna_hybrid import RNAHybridModel
from rna_fold import (
    RNAstralignDataset, collate_rnastralign,
    aggregate_structure_metrics, structure_metrics,
    folding_loss, family_kfold_indices, random_family_split,
)
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
    bpp_cache_dir:str   = '~/bpp_cache'
    seq_col:      Optional[str] = None # None → auto from task (see _auto_fill)
    label_col:    Optional[str] = None # None → auto from task
    lib_col:      Optional[str] = None # column with library name (MRL)
    cell_line:    Optional[str] = None # Muscle | PC3 | HEK (TE only)
    max_len:      Optional[int] = None # None → auto from task

    # Model
    model_type:   str   = 'plucker'   # plucker | bender | transformer
    model_dim:    int   = 128
    num_layers:   int   = 4
    num_heads:    int   = 8            # transformer baseline only
    reduced_dim:  int   = 32
    ff_dim:       Optional[int] = None # default: 4*model_dim
    dropout:      float = 0.1
    pooling:      str   = 'attention'

    # Auxiliary structure supervision (UTR-LM comparison mode)
    # When enabled, the model also predicts secondary structure (per-token)
    # and MFE (sequence-level) as auxiliary targets computed by ViennaRNA,
    # exactly mirroring what UTR-LM does during pretraining.
    aux_struct:   bool  = False        # add SS + MFE auxiliary prediction heads
    lambda_ss:    float = 0.1          # weight for per-token SS cross-entropy
    lambda_mfe:   float = 0.01         # weight for scalar MFE regression

    # RNA Bender geometric losses (active only when model_type='bender')
    lambda_curv:  float = 0.01         # curvature regularisation weight
    lambda_cons:  float = 0.0          # backbone–pairing consistency weight (disabled by default)
    lambda_pair:  float = 0.1          # pair-map (BPP supervision) weight
    use_pair_head:bool  = True         # include pair-map head

    # MoE model (model_type='moe')
    gate_type:           str           = 'scalar'  # 'scalar' | 'vector'
    gate_bias:           float         = 0.0       # initial gate bias (0 = balanced)
    pretrained_geom_encoder: Optional[str] = None  # path to geom encoder checkpoint
    freeze_geom_epochs:  int           = 0         # freeze geom branch for N epochs
    geom_lr_scale:       float         = 0.1       # LR multiplier for geom encoder
    geom_num_layers:     int           = 4         # geometry branch depth (can differ from seq)

    # Hybrid model (model_type='hybrid')
    seq_dim:               int         = 128       # Stage B sequence encoder hidden dim
    seq_num_layers_hybrid: int         = 2         # Stage B sequence encoder depth
    struct_bottleneck_dim: int         = 64        # per-token structure bottleneck dim
    glob_bottleneck_dim:   int         = 128       # global structure bottleneck dim
    bottleneck_mode:       str         = 'full'    # 'full' | 'simple' (ablation)

    # RNAstralign / folding task
    data_format:  str   = 'csv'        # 'csv' or 'bpseq' (rnastralign task only)
    struct_col:   Optional[str] = None # dot-bracket column name (csv format)
    family_col:   Optional[str] = None # family column name (csv format)
    family_split: bool  = True         # use family-aware GroupKFold (rnastralign)
    oracle_edges: bool  = True         # include ground-truth base pairs as graph edges

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
    split_file:   Optional[str] = None  # path to save/load split indices (JSON)

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

    # Pretrain-then-finetune
    pretrained_backbone: Optional[str] = None  # path to pretrain checkpoint; if set,
                                               # encoder weights are loaded before training

    # Eval-only mode (no training)
    eval_checkpoint: Optional[str] = None  # if set, load checkpoint and evaluate on --data; no training

    # Bender positional encoding
    pos_emb_type: str = 'sinusoidal'  # 'sinusoidal' | 'learned'


def _auto_fill(cfg: TrainConfig) -> TrainConfig:
    """Fill task-dependent defaults for fields left as None."""
    if cfg.task == 'rnastralign':
        # Folding task: column defaults differ from UTR tasks
        if cfg.seq_col    is None: cfg.seq_col    = 'sequence'
        if cfg.struct_col is None: cfg.struct_col = 'structure'
        if cfg.family_col is None: cfg.family_col = 'family'
        if cfg.device == 'auto':
            cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return cfg

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
    if cfg.task == 'rnastralign':
        return RNAstralignDataset(
            cfg.data,
            data_format             = cfg.data_format,
            seq_col                 = cfg.seq_col    or 'sequence',
            struct_col              = cfg.struct_col or 'structure',
            family_col              = cfg.family_col or 'family',
            max_len                 = cfg.max_len,
            top_k_struct            = 4,
            use_oracle_struct_edges = cfg.oracle_edges,
        )

    cache = BPPCache(os.path.expanduser(cfg.bpp_cache_dir), backend=cfg.bpp_backend)
    common = dict(
        bpp_cache=cache,
        top_k_struct=0 if cfg.bpp_backend == 'zero' else 4,
        bp_threshold=0.05,
        aux_struct=cfg.aux_struct,
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


# ─── Pretrained encoder loader ────────────────────────────────────────────────

def check_pretrain_arch(ckpt_path: str, cfg: 'TrainConfig'):
    """
    Raise ValueError if the pretrain checkpoint's architecture does not match
    the fine-tune config.

    Checks model_dim, num_layers, reduced_dim.  If the checkpoint does not
    contain a saved config (old format) the check is skipped with a warning.
    """
    ckpt       = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    saved_cfg  = ckpt.get('cfg')
    if saved_cfg is None:
        import warnings
        warnings.warn(
            f'Pretrain checkpoint {ckpt_path!r} has no saved config; '
            f'cannot verify architecture match.  Proceeding anyway.',
            stacklevel=3,
        )
        return

    mismatches = []
    for key in ('model_dim', 'num_layers', 'reduced_dim'):
        pretrain_val = saved_cfg.get(key)
        finetune_val = getattr(cfg, key, None)
        if pretrain_val is not None and pretrain_val != finetune_val:
            mismatches.append(
                f'  {key}: pretrain={pretrain_val}, fine-tune={finetune_val}'
            )

    if mismatches:
        raise ValueError(
            f'Architecture mismatch between pretrain checkpoint and fine-tune config:\n'
            + '\n'.join(mismatches)
            + '\n\nFix: pass --model_dim --num_layers --reduced_dim matching the '
            f'pretrain run, or use a checkpoint trained with the same architecture.'
        )


def load_pretrained_encoder(model: 'RNAStructureGrassmann', ckpt_path: str) -> int:
    """
    Copy encoder weights from a pretrain checkpoint into a fine-tune model.

    The checkpoint can be either:
      - A file saved by pretrain_utr.py (contains 'encoder_state_dict')
      - A raw state dict

    Only keys that exist in the model with matching shapes are loaded
    (strict=False), so task-specific heads and lib_emb are safely ignored.

    Returns the number of parameters loaded.
    """
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    encoder_sd = ckpt.get('encoder_state_dict', ckpt.get('state_dict', ckpt))

    model_sd = model.state_dict()
    to_load, skipped = {}, []
    for k, v in encoder_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            to_load[k] = v
        else:
            skipped.append(k)

    model.load_state_dict(to_load, strict=False)
    n_loaded  = sum(v.numel() for v in to_load.values())
    n_model   = sum(v.numel() for v in model_sd.values())
    frac      = n_loaded / max(n_model, 1)
    if skipped:
        print(f'  Pretrain load: skipped {len(skipped)} keys '
              f'(shape mismatch or not in fine-tune model)')
    if frac < 0.5:
        import warnings
        warnings.warn(
            f'Only {frac:.0%} of fine-tune model parameters were initialised '
            f'from the pretrained checkpoint ({n_loaded:,}/{n_model:,}). '
            f'This usually means the pretrain and fine-tune architectures differ '
            f'(num_layers, reduced_dim, model_dim). Make sure both scripts use '
            f'the same --model_dim --num_layers --reduced_dim.',
            stacklevel=3,
        )
    return n_loaded


# ─── Architecture validation ──────────────────────────────────────────────────

def _check_pretrained_geom_arch(cfg: 'TrainConfig') -> None:
    """
    Warn (not error) when key geometry-branch hyperparams don't match the
    pretrained checkpoint.  A mismatch loads partially and can silently corrupt
    experiments, so we surface it explicitly.

    Checked fields (all present in pretrain_bender.py checkpoints):
        model_dim, num_layers (geom_num_layers), reduced_dim
    """
    if not cfg.pretrained_geom_encoder:
        return
    try:
        ckpt = torch.load(cfg.pretrained_geom_encoder, map_location='cpu',
                          weights_only=False)
    except Exception as e:
        print(f'  [warn] Could not open pretrained geom checkpoint for arch check: {e}')
        return

    ckpt_cfg = ckpt.get('cfg', {})
    if not ckpt_cfg:
        return

    mismatches = []
    checks = [
        ('model_dim',    'model_dim',   cfg.model_dim),
        ('num_layers',   'num_layers',  cfg.geom_num_layers),
        ('reduced_dim',  'reduced_dim', cfg.reduced_dim),
    ]
    for label, key, current_val in checks:
        stored = ckpt_cfg.get(key)
        if stored is not None and stored != current_val:
            mismatches.append(f'{label}: ckpt={stored} vs current={current_val}')

    oracle_ckpt = ckpt_cfg.get('oracle_edges', None)
    if oracle_ckpt is not None:
        edge_mode = 'oracle' if oracle_ckpt else 'sequence-only'
        print(f'  [info] Pretrained geom encoder used {edge_mode} edges')

    if mismatches:
        print(f'  [WARN] Geometry encoder architecture mismatch — partial load likely:')
        for m in mismatches:
            print(f'         {m}')
    else:
        print(f'  [info] Pretrained geom encoder arch matches current config ✓')


# ─── Model factory ────────────────────────────────────────────────────────────

def build_model(cfg: TrainConfig):
    is_folding    = cfg.task == 'rnastralign'
    task_type     = 'classification' if cfg.task == 'ires' else 'regression'
    num_libraries = NUM_LIBRARIES if cfg.task == 'mrl' and cfg.lib_col else 0

    if cfg.model_type == 'transformer':
        # Standard MHA baseline — sequence-only, same output heads as Bender.
        # Edge inputs are accepted but ignored; curvature/consistency losses = 0.
        tf_task    = 'folding' if is_folding else task_type
        aux_struct = True      if is_folding else cfg.aux_struct
        return RNATransformerBaseline(
            model_dim     = cfg.model_dim,
            num_layers    = cfg.num_layers,
            num_heads     = cfg.num_heads,
            ff_dim        = cfg.ff_dim,
            dropout       = cfg.dropout,
            pooling       = cfg.pooling,
            task          = tf_task,
            num_libraries = num_libraries,
            aux_struct    = aux_struct,
            use_pair_head = cfg.use_pair_head,
        )

    if cfg.model_type == 'moe':
        # Two-branch MoE: sequence encoder + geometry (Bender) encoder + learned gate.
        # When a pretrained geom encoder is supplied, auto-read its architecture so
        # the geom branch is always built to match the saved weights exactly.
        geom_layers  = cfg.geom_num_layers
        geom_r       = cfg.reduced_dim
        geom_max_len = None   # None → use shared max_len

        if cfg.pretrained_geom_encoder:
            try:
                _ckpt = torch.load(cfg.pretrained_geom_encoder, map_location='cpu',
                                   weights_only=False)
                _cc = _ckpt.get('cfg', {})
                if _cc:
                    geom_layers  = _cc.get('num_layers',  geom_layers)
                    geom_r       = _cc.get('reduced_dim', geom_r)
                    geom_max_len = _cc.get('max_len') or geom_max_len
                    print(f'  [info] Geom branch arch from checkpoint: '
                          f'layers={geom_layers} r={geom_r} max_len={geom_max_len}')
            except Exception as e:
                print(f'  [warn] Could not read pretrained geom cfg: {e}')
            _check_pretrained_geom_arch(cfg)

        model = RNAMoEMRLModel(
            model_dim        = cfg.model_dim,
            seq_num_layers   = cfg.num_layers,
            seq_num_heads    = cfg.num_heads,
            geom_num_layers  = geom_layers,
            geom_reduced_dim = geom_r,
            geom_max_len     = geom_max_len,
            vocab_size       = VOCAB_SIZE,
            max_len          = cfg.max_len or 256,
            dropout          = cfg.dropout,
            pooling          = cfg.pooling,
            num_libraries    = num_libraries,
            gate_type        = cfg.gate_type,
            gate_bias        = cfg.gate_bias,
        )
        if cfg.pretrained_geom_encoder:
            model.load_pretrained_geom(cfg.pretrained_geom_encoder)
        return model

    if cfg.model_type == 'hybrid':
        # Two-stage hybrid: Stage A = geometry encoder + structure bottleneck,
        # Stage B = small sequence encoder + cross-attention bridge from structure.
        # Same auto-read-arch-from-checkpoint logic as 'moe'.
        geom_layers  = cfg.geom_num_layers
        geom_r       = cfg.reduced_dim
        geom_max_len = None
        geom_dim     = cfg.model_dim   # may be overridden from checkpoint below

        if cfg.pretrained_geom_encoder:
            try:
                _ckpt = torch.load(cfg.pretrained_geom_encoder, map_location='cpu',
                                   weights_only=False)
                _cc = _ckpt.get('cfg', {})
                if _cc:
                    geom_layers  = _cc.get('num_layers',  geom_layers)
                    geom_r       = _cc.get('reduced_dim', geom_r)
                    geom_max_len = _cc.get('max_len') or geom_max_len
                    # model_dim is critical — mismatching it causes a broken load
                    _ckpt_dim = _cc.get('model_dim')
                    if _ckpt_dim is not None and _ckpt_dim != geom_dim:
                        print(f'  [info] geom_dim auto-corrected: '
                              f'{geom_dim} → {_ckpt_dim} (from checkpoint; '
                              f'pass --model_dim {_ckpt_dim} to silence this)')
                        geom_dim = _ckpt_dim
                    print(f'  [info] Geom branch arch from checkpoint: '
                          f'dim={geom_dim} layers={geom_layers} r={geom_r} max_len={geom_max_len}')
            except Exception as e:
                print(f'  [warn] Could not read pretrained geom cfg: {e}')
            _check_pretrained_geom_arch(cfg)

        model = RNAHybridModel(
            vocab_size            = VOCAB_SIZE,
            max_len               = cfg.max_len or 256,
            geom_dim              = geom_dim,
            geom_num_layers       = geom_layers,
            geom_reduced_dim      = geom_r,
            geom_ff_dim           = cfg.ff_dim,
            geom_max_len          = geom_max_len,
            struct_bottleneck_dim = cfg.struct_bottleneck_dim,
            glob_bottleneck_dim   = cfg.glob_bottleneck_dim,
            seq_dim               = cfg.seq_dim,
            seq_num_layers        = cfg.seq_num_layers_hybrid,
            seq_num_heads         = cfg.num_heads,
            seq_ff_dim            = cfg.ff_dim,
            dropout               = cfg.dropout,
            pooling               = cfg.pooling,
            num_libraries         = num_libraries,
            lambda_pair           = cfg.lambda_pair,
            lambda_ss             = cfg.lambda_ss,
            lambda_mfe            = cfg.lambda_mfe,
            lambda_curv           = cfg.lambda_curv,
            lambda_cons           = cfg.lambda_cons,
            bottleneck_mode       = cfg.bottleneck_mode,
        )
        if cfg.pretrained_geom_encoder:
            model.load_pretrained_geom(cfg.pretrained_geom_encoder)
        return model

    if cfg.model_type == 'bender':
        # For the folding task:
        #   task='folding'    → skips building the pooled task head entirely
        #   aux_struct=True   → SS head is a primary output, not optional
        bender_task = 'folding' if is_folding else task_type
        aux_struct  = True      if is_folding else cfg.aux_struct
        return RNABenderModel(
            model_dim     = cfg.model_dim,
            num_layers    = cfg.num_layers,
            reduced_dim   = cfg.reduced_dim,
            ff_dim        = cfg.ff_dim,
            dropout       = cfg.dropout,
            pooling       = cfg.pooling,
            task          = bender_task,
            num_libraries = num_libraries,
            aux_struct    = aux_struct,
            lambda_ss     = cfg.lambda_ss,
            lambda_mfe    = cfg.lambda_mfe,
            use_pair_head = cfg.use_pair_head,
            lambda_pair   = cfg.lambda_pair,
            lambda_curv   = cfg.lambda_curv,
            lambda_cons   = cfg.lambda_cons,
            max_len       = cfg.max_len or 4096,
            pos_emb_type  = cfg.pos_emb_type,
        )

    # Default: original structure-edge Plücker model
    return RNAStructureGrassmann(
        model_dim    = cfg.model_dim,
        num_layers   = cfg.num_layers,
        reduced_dim  = cfg.reduced_dim,
        ff_dim       = cfg.ff_dim,
        dropout      = cfg.dropout,
        pooling      = cfg.pooling,
        task         = task_type,
        num_libraries= num_libraries,
        aux_struct   = cfg.aux_struct,
        lambda_ss    = cfg.lambda_ss,
        lambda_mfe   = cfg.lambda_mfe,
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
    model,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    clip_grad:        float = 1.0,
    scaler:           Optional['torch.cuda.amp.GradScaler'] = None,
    compute_loss_fn   = None,   # callable(outputs_dict, batch) -> Tensor
                                # if None, falls back to old (logits, loss) API
) -> float:
    """
    One training epoch.

    Two calling conventions are supported:

    Old API (RNAStructureGrassmann):
        _, loss = model(**inputs, labels=labels, ...)
        compute_loss_fn should be None.

    New API (RNABenderModel / folding task):
        outputs = model(**inputs)        # returns dict
        loss    = compute_loss_fn(outputs, batch)
        compute_loss_fn must be provided.
    """
    model.train()
    total_loss = 0.0
    use_amp    = scaler is not None

    for batch in loader:
        # Move all tensor values to device; keep non-tensor items (e.g. 'families') aside
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}

        if compute_loss_fn is not None:
            # ── New dict API ──────────────────────────────────────────────────
            optimiser.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                # Forward: strip non-tensor keys the model doesn't accept
                model_inputs = {k: v for k, v in batch.items()
                                if isinstance(v, torch.Tensor)
                                and k not in ('pair_targets', 'ss_labels', 'families')}
                outputs = model(**model_inputs)
                loss    = compute_loss_fn(outputs, batch)
        else:
            # ── Old (logits, loss) API ────────────────────────────────────────
            labels     = batch.pop('labels', None)
            lib_ids    = batch.pop('library_ids', None)
            ss_labels  = batch.pop('ss_ids', None)
            mfe_labels = batch.pop('mfe', None)

            optimiser.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=use_amp):
                result = model(**batch, labels=labels, library_ids=lib_ids,
                               ss_labels=ss_labels, mfe_labels=mfe_labels)
                # result is either (logits, loss) or a dict with 'loss'
                if isinstance(result, dict):
                    loss = result['loss']
                else:
                    _, loss = result

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
    model,
    loader:    DataLoader,
    device:    torch.device,
    task:      str = 'regression',   # 'regression' | 'classification' | 'rnastralign'
) -> Dict[str, float]:
    model.eval()

    if task == 'rnastralign':
        return _evaluate_structure(model, loader, device)

    all_preds:  List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_gates:  List[np.ndarray] = []

    for batch in loader:
        batch   = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                   for k, v in batch.items()}
        labels  = batch.pop('labels', None)
        lib_ids = batch.pop('library_ids', None)
        batch.pop('ss_ids', None)
        batch.pop('mfe', None)
        batch.pop('families', None)

        result = model(**batch, library_ids=lib_ids)
        if isinstance(result, dict):
            logits = result['task_logits']
            if 'gate' in result:
                all_gates.append(result['gate'].squeeze(-1).cpu().numpy())
        else:
            logits, _ = result

        all_preds.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    preds   = np.concatenate(all_preds)
    labels  = np.concatenate(all_labels)
    metrics = compute_metrics(preds, labels, task=task)

    # Gate diagnostics (MoE only): mean/std can hide bimodal collapse, so also
    # track tail fractions — if gate_lo_frac + gate_hi_frac ≈ 1, one branch dominates.
    if all_gates:
        g = np.concatenate(all_gates)
        metrics['gate_mean']    = float(g.mean())
        metrics['gate_std']     = float(g.std())
        metrics['gate_lo_frac'] = float((g < 0.1).mean())   # fraction using geom only
        metrics['gate_hi_frac'] = float((g > 0.9).mean())   # fraction using seq only

    return metrics


@torch.no_grad()
def _evaluate_structure(model, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate pair P/R/F1/MCC and SS accuracy for the rnastralign task."""
    per_seq: List[Dict[str, float]] = []

    for batch in loader:
        batch        = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                        for k, v in batch.items()}
        pair_targets = batch.pop('pair_targets')       # (B, L, L)
        ss_labels    = batch.pop('ss_labels', None)    # (B, L)
        seq_mask     = batch['seq_mask']               # (B, L)
        batch.pop('families', None)

        result = model(**{k: v for k, v in batch.items() if isinstance(v, torch.Tensor)})
        if isinstance(result, dict):
            pair_logits = result.get('pair_logits')
            ss_logits   = result.get('ss_logits')
        else:
            pair_logits, ss_logits = None, None

        if pair_logits is None:
            continue

        B = pair_logits.shape[0]
        for b in range(B):
            seq_len = int(seq_mask[b].sum().item())
            m = structure_metrics(
                pair_logits[b].cpu().numpy(),
                pair_targets[b].cpu().numpy(),
                ss_logits[b].cpu().numpy()   if ss_logits   is not None else None,
                ss_labels[b].cpu().numpy()   if ss_labels   is not None else None,
                seq_len=seq_len,
            )
            per_seq.append(m)

    return aggregate_structure_metrics(per_seq)


# ─── Primary metric (for early stopping and fold ranking) ─────────────────────

def primary_metric(metrics: Dict[str, float], task: str) -> float:
    """Higher is always better."""
    if task == 'classification':
        return metrics['aupr']
    if task == 'rnastralign':
        return metrics.get('pair_f1', 0.0)
    return metrics.get('spearman_r', metrics.get('pearson_r', -metrics['mse']))


# ─── Folding loss factory ─────────────────────────────────────────────────────

def _make_folding_loss_fn(cfg: 'TrainConfig'):
    """Return a compute_loss_fn(outputs, batch) callable for the rnastralign task."""
    lp   = cfg.lambda_pair
    ls   = cfg.lambda_ss
    lc   = cfg.lambda_curv
    lco  = cfg.lambda_cons

    def _loss(outputs, batch):
        return folding_loss(
            outputs,
            pair_targets = batch['pair_targets'],
            ss_labels    = batch.get('ss_labels'),
            seq_mask     = batch.get('seq_mask'),
            lambda_pair  = lp,
            lambda_ss    = ls,
            lambda_curv  = lc,
            lambda_cons  = lco,
        )

    return _loss


# ─── Freeze / unfreeze helpers ────────────────────────────────────────────────

def _freeze_pretrained(model) -> str:
    """Freeze the right pretrained components and return a description string.

    For RNAHybridModel: if structure heads were loaded from a checkpoint, freeze
    encoder + heads (struct_bottleneck stays trainable — it was not pretrained).
    Otherwise fall back to encoder-backbone-only freeze.

    For all other models (MoE, etc.): freeze_geom_encoder() (backbone only).
    """
    if hasattr(model, '_heads_loaded') and model._heads_loaded:
        model.freeze_encoder_and_heads()
        return 'encoder + structure heads frozen (bottleneck stays trainable)'
    model.freeze_geom_encoder()
    return 'geometry encoder backbone frozen'


def _unfreeze_pretrained(model) -> str:
    """Reverse of _freeze_pretrained — unfreeze the same component set."""
    if hasattr(model, '_heads_loaded') and model._heads_loaded:
        model.unfreeze_encoder_and_heads()
        return 'encoder + structure heads unfrozen'
    model.unfreeze_geom_encoder()
    return 'geometry encoder backbone unfrozen'


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
    device = torch.device(cfg.device)
    if cfg.task == 'rnastralign':
        task        = 'rnastralign'
        collate_fn  = collate_rnastralign
        loss_fn     = _make_folding_loss_fn(cfg)
    elif cfg.task == 'ires':
        task        = 'classification'
        collate_fn  = collate_utr
        loss_fn     = None
    else:
        task        = 'regression'
        collate_fn  = collate_utr
        loss_fn     = None

    train_ds = Subset(dataset, train_idx)
    val_ds   = val_dataset if val_dataset is not None else Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=cfg.num_workers,
        pin_memory=(cfg.device != 'cpu'),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        collate_fn=collate_fn, num_workers=cfg.num_workers,
        pin_memory=(cfg.device != 'cpu'),
    )

    model    = build_model(cfg).to(device)
    n_params = model.get_num_params()

    if cfg.pretrained_backbone:
        check_pretrain_arch(cfg.pretrained_backbone, cfg)
        n_loaded = load_pretrained_encoder(model, cfg.pretrained_backbone)
        print(f'  Loaded pretrained encoder: {n_loaded:,} params '
              f'from {cfg.pretrained_backbone}')

    # MoE / Hybrid: differential LR and optional initial freeze of geometry encoder
    if cfg.model_type in ('moe', 'hybrid') and cfg.pretrained_geom_encoder:
        if cfg.freeze_geom_epochs > 0:
            freeze_desc = _freeze_pretrained(model)
            print(f'  Freeze schedule: {freeze_desc} for {cfg.freeze_geom_epochs} epochs')
        param_groups = model.get_optimizer_groups(cfg.lr, cfg.geom_lr_scale)
        opt = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    sched  = WarmupCosineScheduler(opt, cfg.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda') if (cfg.use_amp and device.type == 'cuda') else None

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
        # Restore freeze state: pretrained components may still be frozen at this epoch
        if (cfg.model_type in ('moe', 'hybrid')
                and cfg.pretrained_geom_encoder
                and cfg.freeze_geom_epochs > 0):
            if ckpt['epoch'] < cfg.freeze_geom_epochs:
                freeze_desc = _freeze_pretrained(model)
                print(f'  Resumed with {freeze_desc} '
                      f'(will unfreeze at epoch {cfg.freeze_geom_epochs + 1})')
            else:
                _unfreeze_pretrained(model)

    val_size = len(val_dataset) if val_dataset is not None else len(val_idx)
    amp_tag  = 'AMP' if scaler else 'fp32'
    aux_tag  = f' | aux_struct λ_ss={cfg.lambda_ss} λ_mfe={cfg.lambda_mfe}' if cfg.aux_struct else ''
    print(f'\n  Fold {fold_num} | {len(train_idx)} train / {val_size} val '
          f'| {n_params:,} params | {amp_tag} | eval_every={cfg.eval_every}{aux_tag}')

    for epoch in range(start_epoch, cfg.epochs + 1):
        # MoE / Hybrid freeze/thaw: unfreeze pretrained components after freeze_geom_epochs
        if (cfg.model_type in ('moe', 'hybrid')
                and cfg.pretrained_geom_encoder
                and cfg.freeze_geom_epochs > 0
                and epoch == cfg.freeze_geom_epochs + 1):
            unfreeze_desc = _unfreeze_pretrained(model)
            print(f'    Epoch {epoch}: {unfreeze_desc}')

        t0         = time.time()
        train_loss = train_epoch(model, train_loader, opt, sched, device,
                                 cfg.clip_grad, scaler, compute_loss_fn=loss_fn)
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
        ckpt_payload: Dict = {
            'state_dict':  best_state,
            'metrics':     best_metrics,
            'cfg':         cfg,
            'best_epoch':  best_epoch,
        }
        if cfg.model_type == 'moe':
            ckpt_payload['moe_meta'] = {
                'pretrained_geom_encoder': cfg.pretrained_geom_encoder,
                'freeze_geom_epochs':      cfg.freeze_geom_epochs,
                'geom_lr_scale':           cfg.geom_lr_scale,
                'geom_num_layers':         cfg.geom_num_layers,
                'gate_type':               cfg.gate_type,
                'gate_bias':               cfg.gate_bias,
                # Gate summary at best epoch (from best_metrics if present)
                'gate_mean':    best_metrics.get('gate_mean'),
                'gate_std':     best_metrics.get('gate_std'),
                'gate_lo_frac': best_metrics.get('gate_lo_frac'),
                'gate_hi_frac': best_metrics.get('gate_hi_frac'),
            }
        torch.save(ckpt_payload, best_path)
        print(f'  Saved → {best_path}')

    # ── Final test-set evaluation (once, not used for selection) ──────────────
    if test_dataset is not None and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test_loader = DataLoader(
            test_dataset, batch_size=cfg.batch_size * 2, shuffle=False,
            collate_fn=collate_fn, num_workers=cfg.num_workers,
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


# ─── Eval-only mode ──────────────────────────────────────────────────────────

def run_eval(cfg: TrainConfig):
    """Load a saved checkpoint and evaluate on cfg.data. No training."""
    device = torch.device(cfg.device)

    print(f'Eval mode | checkpoint: {cfg.eval_checkpoint}')
    print(f'Task: {cfg.task} | Data: {cfg.data} | Device: {cfg.device}')

    # ── Detect checkpoint type and extract weights ────────────────────────────
    ckpt = torch.load(cfg.eval_checkpoint, map_location='cpu', weights_only=False)
    ckpt_keys = set(ckpt.keys())

    if 'best_state' in ckpt_keys:
        # Resume checkpoint (_resume.pt) — best_state = weights at best val epoch
        raw_state = ckpt['best_state']
        epoch_info = (f'best_state (best epoch {ckpt.get("best_epoch", "?")} '
                      f'/ total {ckpt.get("epoch", "?")})')
        stored_metrics = ckpt.get('best_metrics', {})
    elif 'state_dict' in ckpt_keys:
        # Best-model checkpoint (_best.pt)
        raw_state = ckpt['state_dict']
        epoch_info = f'state_dict (epoch {ckpt.get("best_epoch", "?")})'
        stored_metrics = ckpt.get('metrics', {})
        # Warn if stored cfg disagrees with CLI on model architecture
        if 'cfg' in ckpt:
            stored_cfg = ckpt['cfg']
            mismatches = []
            for attr in ('model_type', 'model_dim', 'num_layers', 'reduced_dim'):
                sv = getattr(stored_cfg, attr, None)
                cv = getattr(cfg, attr)
                if sv is not None and sv != cv:
                    mismatches.append(f'{attr}: ckpt={sv} CLI={cv}')
            if mismatches:
                print(f'  WARNING: model arch may differ from checkpoint: '
                      + ', '.join(mismatches))
    else:
        raise ValueError(
            f'Unrecognized checkpoint format. Keys found: {sorted(ckpt_keys)}\n'
            f'Expected either a _resume.pt (has "best_state") or _best.pt (has "state_dict").'
        )

    print(f'  Weights: {epoch_info}')
    if stored_metrics:
        print('  Stored val metrics: '
              + ' | '.join(f'{k}={v:.4f}' for k, v in stored_metrics.items()))

    # ── Build model and load weights ──────────────────────────────────────────
    model = build_model(cfg).to(device)
    n_params = model.get_num_params()
    model.load_state_dict({k: v.to(device) for k, v in raw_state.items()})
    model.eval()
    print(f'  Model: {cfg.model_type} | {n_params:,} params')

    # ── Build eval dataset ────────────────────────────────────────────────────
    dataset = build_dataset(cfg)
    print(f'  Eval dataset: {len(dataset)} sequences')

    if cfg.task == 'rnastralign':
        task       = 'rnastralign'
        collate_fn = collate_rnastralign
    elif cfg.task == 'ires':
        task       = 'classification'
        collate_fn = collate_utr
    else:
        task       = 'regression'
        collate_fn = collate_utr

    loader = DataLoader(
        dataset,
        batch_size  = cfg.batch_size * 2,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = cfg.num_workers,
        pin_memory  = (device.type != 'cpu'),
    )

    metrics = evaluate(model, loader, device, task)
    print('\n── Eval results ' + '─' * 40)
    for k, v in metrics.items():
        print(f'  {k:>20s}: {v:.4f}')


# ─── Cross-validation orchestrator ───────────────────────────────────────────

def run_cv(cfg: TrainConfig):
    """Full cross-validation run; prints aggregate statistics."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f'Task: {cfg.task} | Data: {cfg.data} | Device: {cfg.device}')
    if cfg.model_type == 'transformer':
        print(f'Model: {cfg.model_type} | dim={cfg.model_dim} layers={cfg.num_layers} heads={cfg.num_heads}')
    elif cfg.model_type == 'moe':
        geom_src = f'pretrained({cfg.pretrained_geom_encoder})' if cfg.pretrained_geom_encoder else 'scratch'
        print(f'Model: moe | dim={cfg.model_dim} '
              f'seq_layers={cfg.num_layers} geom_layers={cfg.geom_num_layers} r={cfg.reduced_dim} '
              f'gate={cfg.gate_type} geom={geom_src} freeze={cfg.freeze_geom_epochs}ep '
              f'geom_lr×{cfg.geom_lr_scale}')
    elif cfg.model_type == 'hybrid':
        geom_src = f'pretrained({cfg.pretrained_geom_encoder})' if cfg.pretrained_geom_encoder else 'scratch'
        print(f'Model: hybrid | geom_dim={cfg.model_dim} geom_layers={cfg.geom_num_layers} r={cfg.reduced_dim} '
              f'seq_dim={cfg.seq_dim} seq_layers={cfg.seq_num_layers_hybrid} heads={cfg.num_heads} '
              f'btok={cfg.struct_bottleneck_dim} bglob={cfg.glob_bottleneck_dim} '
              f'geom={geom_src} freeze={cfg.freeze_geom_epochs}ep geom_lr×{cfg.geom_lr_scale}')
    else:
        print(f'Model: {cfg.model_type} | dim={cfg.model_dim} layers={cfg.num_layers} r={cfg.reduced_dim}')
    print(f'BPP backend: {cfg.bpp_backend} | Folds: {cfg.folds}')
    if cfg.aux_struct:
        print(f'Aux struct: ON  (λ_ss={cfg.lambda_ss}, λ_mfe={cfg.lambda_mfe})')
    if cfg.model_type == 'bender':
        print(f'Bender losses: λ_curv={cfg.lambda_curv} λ_cons={cfg.lambda_cons} λ_pair={cfg.lambda_pair}')

    dataset = build_dataset(cfg)
    n = len(dataset)
    if cfg.task == 'rnastralign':
        task = 'rnastralign'
    elif cfg.task == 'ires':
        task = 'classification'
    else:
        task = 'regression'
    print(f'Dataset: {n} sequences | task={task}')

    # Build fold splits
    hold_out_ds: Optional[object] = None   # test set, evaluated once at the end
    if cfg.split_file and os.path.exists(cfg.split_file):
        # ── Load previously saved split (guarantees identical train/val for comparison) ──
        with open(cfg.split_file) as _f:
            _saved = json.load(_f)
        folds        = [(np.array(s['train']), np.array(s['val'])) for s in _saved['folds']]
        val_datasets = [None] * len(folds)
        print(f'Split loaded: {cfg.split_file} ({len(folds)} fold(s), '
              f'{len(folds[0][0])} train / {len(folds[0][1])} val)')
    elif cfg.test_data is not None:
        test_cfg    = dataclasses.replace(cfg, data=cfg.test_data)
        hold_out_ds = build_dataset(test_cfg)
        idx   = np.random.permutation(n)
        split = int(n * (1 - cfg.val_frac))
        folds = [(idx[:split], idx[split:])]
        val_datasets = [None]
        print(f'Hold-out split: {split} train / {n - split} val '
              f'(val from train CSV, test CSV evaluated once at end)')
    elif cfg.task == 'rnastralign' and cfg.family_split:
        # Family-aware splitting: keeps every RNA family entirely in one split
        families = [dataset[i]['family'] for i in range(n)]
        if cfg.folds == 1:
            tr, va   = random_family_split(families, cfg.val_frac, cfg.seed)
            folds    = [(tr, va)]
        else:
            folds    = family_kfold_indices(families, k=cfg.folds, seed=cfg.seed)
        val_datasets = [None] * len(folds)
        n_fam = len(set(families))
        print(f'Family-aware split: {n_fam} families across {len(folds)} fold(s)')
    elif cfg.folds == 1:
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

    # ── Optionally save split for reproducibility ──────────────────────────────
    if cfg.split_file and not os.path.exists(cfg.split_file):
        os.makedirs(os.path.dirname(os.path.abspath(cfg.split_file)), exist_ok=True)
        with open(cfg.split_file, 'w') as _f:
            json.dump({'folds': [{'train': tr.tolist(), 'val': va.tolist()}
                                 for tr, va in folds]}, _f)
        print(f'Split saved: {cfg.split_file}')

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
                   choices=['mrl', 'te', 'el', 'ires', 'rlu', 'rnastralign'])
    p.add_argument('--data',         required=True,
                   help='Path to input CSV file')
    p.add_argument('--bpp_backend',  default='mfe',
                   choices=['viennarna', 'mfe', 'zero'],
                   help='Structure computation: viennarna (slow/accurate), '
                        'mfe (fast/binary), zero (ablation — local edges only)')
    p.add_argument('--bpp_cache_dir',default='~/bpp_cache')
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
    p.add_argument('--model_type',   default='plucker',
                   choices=['plucker', 'bender', 'transformer', 'moe', 'hybrid', 'fcgrcnn'],
                   help='plucker  = original StructureEdgePlucker; '
                        'bender = RNA Bender with Grassmann curvature; '
                        'transformer  = standard MHA baseline (sequence-only); '
                        'moe  = seq + geom mixture-of-experts; '
                        'hybrid   = two-stage: geometry bottleneck + seq cross-attn'
                        'fcgrcnn = fcgr based cnn approach')
    p.add_argument('--model_dim',    type=int,   default=128)
    p.add_argument('--num_layers',   type=int,   default=4)
    p.add_argument('--num_heads',    type=int,   default=8,
                   help='[transformer] Number of attention heads')
    p.add_argument('--reduced_dim',  type=int,   default=32)
    p.add_argument('--dropout',      type=float, default=0.1)
    p.add_argument('--pooling',      default='attention',
                   choices=['attention', 'mean'])
    # Auxiliary structure supervision
    p.add_argument('--aux_struct',   action='store_true',
                   help='Add SS (per-token) and MFE (scalar) auxiliary prediction '
                        'heads, trained alongside the primary task.  Mirrors the '
                        'auxiliary supervision strategy used in UTR-LM.')
    p.add_argument('--lambda_ss',    type=float, default=0.1,
                   help='Loss weight for the auxiliary SS cross-entropy term')
    p.add_argument('--lambda_mfe',   type=float, default=0.01,
                   help='Loss weight for the auxiliary MFE MSE term')
    # RNA Bender geometric losses
    p.add_argument('--lambda_curv',  type=float, default=0.01,
                   help='[bender] Curvature regularisation weight')
    p.add_argument('--lambda_cons',  type=float, default=0.0,
                   help='[bender] Backbone–pairing consistency loss weight (default off)')
    p.add_argument('--lambda_pair',  type=float, default=0.1,
                   help='[bender] Pair-map (BPP supervision) loss weight')
    p.add_argument('--no_pair_head', action='store_true',
                   help='[bender] Disable the pair-map output head')
    p.add_argument('--pos_emb_type', default='learned', choices=['sinusoidal', 'learned'],
                   help='[bender] Positional encoding type: sinusoidal (no params, any length) '
                        'or learned (fixed max_len, adds max_len×dim parameters)')
    # RNAstralign / folding
    p.add_argument('--data_format',  default='csv', choices=['csv', 'json', 'bpseq'],
                   help='[rnastralign] Input format: csv, json dict, or bpseq directory')
    p.add_argument('--struct_col',   default=None,
                   help='[rnastralign/csv] Dot-bracket column name')
    p.add_argument('--family_col',   default=None,
                   help='[rnastralign/csv] Family label column name')
    p.add_argument('--no_family_split', action='store_true',
                   help='[rnastralign] Disable family-aware splits; use random splits instead')
    p.add_argument('--no_oracle_edges', action='store_true',
                   help='[rnastralign] Use only local/backbone edges; no ground-truth base '
                        'pairs in the input graph (sequence-only ablation)')
    # MoE model
    p.add_argument('--gate_type',    default='scalar', choices=['scalar', 'vector'],
                   help='[moe] Gate output shape: scalar (B,1) or vector (B,d)')
    p.add_argument('--gate_bias',    type=float, default=0.0,
                   help='[moe] Initial gate bias (0=balanced, <0=trust seq more)')
    p.add_argument('--pretrained_geom_encoder', default=None,
                   help='[moe] Path to pretrained geometry encoder checkpoint '
                        '(from pretrain_bender.py)')
    p.add_argument('--freeze_geom_epochs', type=int, default=0,
                   help='[moe] Keep geometry encoder frozen for first N epochs '
                        'to protect the pretrained signal')
    p.add_argument('--geom_lr_scale', type=float, default=0.1,
                   help='[moe] LR multiplier for geometry encoder (prevents forgetting)')
    p.add_argument('--geom_num_layers', type=int, default=4,
                   help='[moe/hybrid] Depth of geometry branch (can differ from seq branch)')
    # Hybrid model
    p.add_argument('--seq_dim',        type=int, default=128,
                   help='[hybrid] Stage B sequence encoder hidden dim')
    p.add_argument('--seq_num_layers_hybrid', type=int, default=2,
                   help='[hybrid] Stage B sequence encoder depth')
    p.add_argument('--struct_bottleneck_dim', type=int, default=64,
                   help='[hybrid] Per-token structure bottleneck dimension')
    p.add_argument('--glob_bottleneck_dim',   type=int, default=128,
                   help='[hybrid] Global structure bottleneck dimension')
    p.add_argument('--bottleneck_mode',  default='v2', choices=['v2', 'v1', 'full', 'simple'],
                   help='[hybrid] Bottleneck variant: full=[H∥partner_ctx∥degree∥ss∥curv], '
                        'simple=[H∥partner_ctx∥curv] (cleaner ablation)')
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
    p.add_argument('--split_file',   default=None,
                   help='JSON file to save (first run) or load (subsequent runs) exact '
                        'train/val indices.  Guarantees both models train on the same split.')
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
    p.add_argument('--eval_checkpoint', default=None,
                   help='Path to a _resume.pt or _best.pt checkpoint. '
                        'Evaluates on --data with no training and exits. '
                        'Specify the same model arch flags used during training.')
    p.add_argument('--pretrained_backbone', default=None,
                   help='Path to a pretrain checkpoint (from pretrain_utr.py); '
                        'encoder weights are loaded before fine-tuning begins')

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
        data_format  = args.data_format,
        struct_col   = args.struct_col,
        family_col   = args.family_col,
        family_split = not args.no_family_split,
        oracle_edges = not args.no_oracle_edges,
        model_type   = args.model_type,
        model_dim    = args.model_dim,
        num_layers   = args.num_layers,
        num_heads    = args.num_heads,
        reduced_dim  = args.reduced_dim,
        dropout      = args.dropout,
        pooling      = args.pooling,
        aux_struct   = args.aux_struct,
        lambda_ss    = args.lambda_ss,
        lambda_mfe   = args.lambda_mfe,
        lambda_curv  = args.lambda_curv,
        lambda_cons  = args.lambda_cons,
        lambda_pair  = args.lambda_pair,
        use_pair_head= not args.no_pair_head,
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
        split_file   = args.split_file,
        device       = args.device,
        num_workers  = args.num_workers,
        output_dir   = args.output_dir,
        save_best            = not args.no_save,
        test_data            = args.test_data,
        resume_from          = args.resume_from,
        pretrained_backbone  = args.pretrained_backbone,
        eval_checkpoint      = args.eval_checkpoint,
        pos_emb_type         = args.pos_emb_type,
        gate_type            = args.gate_type,
        gate_bias            = args.gate_bias,
        pretrained_geom_encoder = args.pretrained_geom_encoder,
        freeze_geom_epochs   = args.freeze_geom_epochs,
        geom_lr_scale        = args.geom_lr_scale,
        geom_num_layers      = args.geom_num_layers,
        seq_dim              = args.seq_dim,
        seq_num_layers_hybrid = args.seq_num_layers_hybrid,
        struct_bottleneck_dim = args.struct_bottleneck_dim,
        glob_bottleneck_dim  = args.glob_bottleneck_dim,
        bottleneck_mode      = args.bottleneck_mode,
    )
    if args.label_col:
        cfg.label_col = args.label_col
    return _auto_fill(cfg)


if __name__ == '__main__':
    cfg = parse_args()
    if cfg.eval_checkpoint:
        run_eval(cfg)
    else:
        run_cv(cfg)
