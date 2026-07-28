"""
Ablation harness for RNA Bender's geometric machinery.

Standalone script — imports from train_utr.py / rna_bender.py / rna_baseline.py
but edits none of them.  All new classes and logic live here.

Answers one question: is the Grassmann/Plücker geometry in RNABenderModel
actually load-bearing, or would something simpler do just as well?

Variants (all trained from scratch, same data split, same hyperparameters
unless noted):

  full             Unmodified RNABenderModel.  Control / reproduces the
                    reference run.

  no_bb_curv       BackboneCurvatureMixer runs exactly as before (so φ_bb and
                    φ_curv still exist and burn the same number of parameters)
                    but g_bb and g_curv are zeroed immediately before the
                    Step-D aggregator (rna_bender.py:293-378, "g_bb" /
                    "g_curv").  Isolates how much the backbone-Plücker and
                    curvature channels contribute to the task, holding
                    parameter count fixed.

  bilinear_pair    plucker_coords(z_i, z_j) — the fixed, parameter-free,
                    antisymmetric wedge product — is replaced by a learned
                    nn.Bilinear(r, r, plu_dim) in *both* the backbone and
                    structural-edge mixers.  Same output dimensionality, so
                    every downstream MLP (φ_bb, φ_curv, φ_bp) is unchanged.
                    This ADDS parameters relative to `full` (the wedge has
                    zero parameters; a generic bilinear form does not) — the
                    comparison is deliberately generous to the alternative.
                    Tests whether the antisymmetric "plane" structure beats
                    an unconstrained quadratic interaction of matched output
                    size.

  no_struct_edges  Identical architecture to `full`, trained on a dataset
                    built with --bpp_backend zero instead of mfe.  This is
                    the same "structure disabled" ablation path documented in
                    train_utr.py's own usage examples and used as condition A
                    in run_comparison.py: BPPCache returns an all-zero matrix,
                    so top_k_struct is forced to 0 (no structural edges at
                    all) AND every local edge's bp_prob channel is zero too —
                    i.e. no MFE-derived information survives anywhere in the
                    graph.  Only plain sequence-adjacency edges (offsets
                    ±1, ±2, bp_prob≡0) remain, plus the backbone-Plücker path
                    (which is computed from z directly, not from the edge
                    graph, so it is unaffected).  Isolates how much the MFE
                    structure contributes vs. sequence alone.

  transformer      RNATransformerBaseline (rna_baseline.py) — standard
                    multi-head self-attention, sequence-only by construction
                    (edge inputs are accepted but ignored).  The single most
                    informative comparison: does any of this geometry beat a
                    generic Transformer at all?

                    Parameter count will NOT automatically match `full` —
                    a plain nn.TransformerEncoder at the same model_dim/
                    num_layers lands at a different parameter count than
                    Bender's geometric layers.  Use --transformer_num_layers /
                    --transformer_ff_dim / --transformer_model_dim to close
                    the gap; every variant's parameter count is printed at
                    startup so the mismatch (if any) is visible, not hidden.

Usage (mirrors train_utr.py's flags for the shared config):

    python ablation_bender.py \\
        --data capsule_data/.../4.1_train_data_GSM3130435_egfp_unmod_1.csv \\
        --test_data capsule_data/.../4.1_test_data_GSM3130435_egfp_unmod_1.csv \\
        --model_dim 96 --num_layers 3 --reduced_dim 16 \\
        --epochs 60 --batch_size 64 --lr 3e-4 --patience 12 --eval_every 1 \\
        --split_file outputs/mrl_4.1_seed42_split.json \\
        --output_dir outputs/ablation_bender_d96_l3_r16 \\
        --variants full no_bb_curv bilinear_pair no_struct_edges transformer

Reusing the same --split_file as the reference run guarantees every variant
(and the original) trains/validates on the identical partition — required
for the R² comparison to isolate architecture, not sampling noise.

Reusing the same --bpp_cache_dir (default ~/bpp_cache, same as train_utr.py)
means the `full`/`no_bb_curv`/`bilinear_pair`/`transformer` variants hit a
warm BPP cache if you already ran the reference command.

Results: outputs/<output_dir>/<variant>/{task}_fold{n}_best.pt + resume
checkpoint, outputs/<output_dir>/results.json (variant -> metrics, for
resuming a partially-completed sweep), and a final comparison table.
"""

import argparse
import dataclasses
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from train_utr import (
    TrainConfig, _auto_fill, build_dataset,
    WarmupCosineScheduler, train_epoch, evaluate, primary_metric,
)
from utr_datasets import (
    NUM_LIBRARIES, collate_utr, kfold_indices, stratified_kfold_indices,
)
from rna_bender import (
    RNABenderModel, VOCAB_SIZE, N_EDGE_FEATS, BACKBONE_OFFSETS,
    BackboneCurvatureMixer,
)
from rna_baseline import RNATransformerBaseline

VARIANT_CHOICES = ['full', 'no_bb_curv', 'bilinear_pair', 'no_struct_edges', 'transformer']


# ─── Ablation model pieces ─────────────────────────────────────────────────────

class ZeroingBackboneCurvatureMixer(BackboneCurvatureMixer):
    """
    Identical to BackboneCurvatureMixer (same φ_bb / φ_curv parameters, same
    forward computation) except g_bb and g_curv are zeroed right before they
    would reach the layer's Step-D aggregator.  p_bb1 and kappa are returned
    unchanged so curvature/consistency loss terms still see real geometry —
    only the *residual-path contribution* of these two channels is cut.

    Parameter count is unchanged vs. `full`: φ_bb/φ_curv still exist, they
    just compute a dead end.
    """

    def forward(self, h: torch.Tensor, seq_mask: torch.Tensor):
        z, g_bb, g_curv, p_bb1, kappa = super().forward(h, seq_mask)
        return z, torch.zeros_like(g_bb), torch.zeros_like(g_curv), p_bb1, kappa


class LearnedPairFeature(nn.Module):
    """
    Generic learned quadratic interaction of (z_i, z_j), output-dimension-
    matched to plucker_coords' r*(r-1)/2 so it drops into the same
    downstream φ_bb / φ_curv / φ_bp MLPs unchanged.

    plucker_coords has zero parameters (it's a fixed antisymmetric wedge);
    this module has real parameters (nn.Bilinear), so it is a strictly more
    expressive — not merely different — competitor.  If it does not beat
    the wedge, that is evidence the antisymmetric structure itself matters,
    not just "any quadratic interaction of this size."
    """

    def __init__(self, reduced_dim: int):
        super().__init__()
        plu_dim = reduced_dim * (reduced_dim - 1) // 2
        self.bilinear = nn.Bilinear(reduced_dim, reduced_dim, plu_dim, bias=True)

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        p = self.bilinear(u, v)
        norm = p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        return p / norm


class BilinearBackboneCurvatureMixer(nn.Module):
    """
    Copy of BackboneCurvatureMixer (rna_bender.py) with the single line that
    calls plucker_coords(z, zj) replaced by a LearnedPairFeature.  Everything
    else — offsets, curvature construction, φ_bb/φ_curv MLPs — is identical.
    """

    def __init__(
        self,
        reduced_dim: int,
        model_dim: int,
        offsets: Tuple[int, ...] = BACKBONE_OFFSETS,
    ):
        super().__init__()
        self.offsets = offsets
        self.r = reduced_dim
        plu_dim = reduced_dim * (reduced_dim - 1) // 2
        self.plu_dim = plu_dim
        hidden = model_dim // 2

        self.W_red = nn.Linear(model_dim, reduced_dim, bias=True)
        self.pair_feature = LearnedPairFeature(reduced_dim)

        self.phi_bb = nn.Sequential(
            nn.Linear(len(offsets) * plu_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, model_dim),
        )
        self.phi_curv = nn.Sequential(
            nn.Linear(plu_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, model_dim),
        )

    def forward(self, h: torch.Tensor, seq_mask: torch.Tensor):
        B, L, _d = h.shape
        z = self.W_red(h)
        mask_f = seq_mask.float().unsqueeze(-1)

        p_list: List[torch.Tensor] = []
        for delta in self.offsets:
            zj = torch.zeros_like(z)
            zj[:, :L - delta, :] = z[:, delta:, :]
            mj = torch.zeros_like(seq_mask, dtype=torch.float)
            mj[:, :L - delta] = seq_mask[:, delta:].float()
            m_ij = (seq_mask.float() * mj).unsqueeze(-1)

            p = self.pair_feature(z, zj) * m_ij
            p_list.append(p)

        p_bb1 = p_list[0]
        p_fwd = torch.zeros_like(p_bb1)
        p_bwd = torch.zeros_like(p_bb1)
        p_fwd[:, :L - 1, :] = p_bb1[:, 1:, :]
        p_bwd[:, 1:, :] = p_bb1[:, :L - 1, :]
        kappa = (p_fwd - 2 * p_bb1 + p_bwd) * mask_f

        p_all = torch.cat(p_list, dim=-1)
        g_bb = self.phi_bb(p_all)
        g_curv = self.phi_curv(kappa)

        return z, g_bb, g_curv, p_bb1, kappa


class BilinearStructuralEdgeMixer(nn.Module):
    """
    Copy of StructuralEdgeMixer (rna_bender.py) with plucker_coords(z_i, z_j)
    replaced by the same LearnedPairFeature substitution (a separate instance
    from the backbone mixer's, mirroring how plucker_coords is called
    independently — and without shared weights — in both places).
    """

    def __init__(
        self,
        reduced_dim: int,
        model_dim: int,
        edge_feat_dim: int = N_EDGE_FEATS,
    ):
        super().__init__()
        plu_dim = reduced_dim * (reduced_dim - 1) // 2
        d = model_dim

        self.pair_feature = LearnedPairFeature(reduced_dim)
        self.phi_bp = nn.Sequential(
            nn.Linear(plu_dim + edge_feat_dim, d),
            nn.GELU(),
        )
        self.attn_score = nn.Linear(d + d + edge_feat_dim, 1)

    def forward(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        edge_idx: torch.Tensor,
        edge_feat: torch.Tensor,
    ):
        B, L, r = z.shape
        K = edge_idx.shape[-1]

        valid_mask = (edge_idx >= 0)
        clamped_idx = edge_idx.clamp(min=0).long()

        batch_idx = torch.arange(B, device=z.device).view(B, 1, 1).expand(B, L, K)
        z_j = z[batch_idx, clamped_idx]
        z_i = z.unsqueeze(2).expand(B, L, K, r)

        p_struct = self.pair_feature(z_i, z_j)
        p_struct = p_struct * valid_mask.unsqueeze(-1).float()

        feat = torch.cat([p_struct, edge_feat], dim=-1)
        msg = self.phi_bp(feat)

        h_j = h[batch_idx, clamped_idx]
        scores = self.attn_score(torch.cat([msg, h_j, edge_feat], dim=-1)).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, -1e4)
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        g_bp = (attn.unsqueeze(-1) * msg).sum(dim=2)
        return g_bp, p_struct


# ─── Model factory ────────────────────────────────────────────────────────────

def _bender_kwargs(cfg: TrainConfig) -> Dict:
    """Mirrors build_model()'s 'bender' branch in train_utr.py exactly."""
    task_type = 'classification' if cfg.task == 'ires' else 'regression'
    num_libraries = NUM_LIBRARIES if cfg.task == 'mrl' and cfg.lib_col else 0
    return dict(
        vocab_size=VOCAB_SIZE,
        max_len=cfg.max_len or 256,
        model_dim=cfg.model_dim,
        num_layers=cfg.num_layers,
        reduced_dim=cfg.reduced_dim,
        ff_dim=cfg.ff_dim,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
        task=task_type,
        num_libraries=num_libraries,
        aux_struct=cfg.aux_struct,
        lambda_ss=cfg.lambda_ss,
        lambda_mfe=cfg.lambda_mfe,
        use_pair_head=cfg.use_pair_head,
        lambda_pair=cfg.lambda_pair,
        lambda_curv=cfg.lambda_curv,
        lambda_cons=cfg.lambda_cons,
        pos_emb_type=cfg.pos_emb_type,
    )


def _transformer_kwargs(cfg: TrainConfig, args: argparse.Namespace) -> Dict:
    task_type = 'classification' if cfg.task == 'ires' else 'regression'
    num_libraries = NUM_LIBRARIES if cfg.task == 'mrl' and cfg.lib_col else 0
    return dict(
        vocab_size=VOCAB_SIZE,
        max_len=cfg.max_len or 256,
        model_dim=args.transformer_model_dim or cfg.model_dim,
        num_layers=args.transformer_num_layers or cfg.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.transformer_ff_dim or cfg.ff_dim,
        dropout=cfg.dropout,
        pooling=cfg.pooling,
        task=task_type,
        num_libraries=num_libraries,
        aux_struct=cfg.aux_struct,
        use_pair_head=cfg.use_pair_head,
    )


def build_variant_model(variant: str, cfg: TrainConfig, args: argparse.Namespace) -> nn.Module:
    if variant == 'transformer':
        return RNATransformerBaseline(**_transformer_kwargs(cfg, args))

    kwargs = _bender_kwargs(cfg)

    if variant in ('full', 'no_struct_edges'):
        return RNABenderModel(**kwargs)

    if variant == 'no_bb_curv':
        model = RNABenderModel(**kwargs)
        for block in model.blocks:
            block.bb_mixer = ZeroingBackboneCurvatureMixer(model.reduced_dim, model.model_dim)
        return model

    if variant == 'bilinear_pair':
        model = RNABenderModel(**kwargs)
        for block in model.blocks:
            block.bb_mixer = BilinearBackboneCurvatureMixer(model.reduced_dim, model.model_dim)
            block.edge_mixer = BilinearStructuralEdgeMixer(model.reduced_dim, model.model_dim)
        return model

    raise ValueError(f'Unknown variant: {variant!r}')


# ─── Fold splitting (mirrors run_cv's split logic in train_utr.py) ────────────

def build_folds(cfg: TrainConfig, dataset, n: int):
    """
    Returns (folds, val_datasets, hold_out_ds).

    folds: list of (train_idx, val_idx) arrays
    hold_out_ds: test dataset if cfg.test_data is set, else None

    Copied from train_utr.run_cv (rnastralign family-split branch omitted —
    not applicable to mrl/te/el/ires/rlu).
    """
    hold_out_ds = None
    if cfg.split_file and os.path.exists(cfg.split_file):
        with open(cfg.split_file) as f:
            saved = json.load(f)
        folds = [(np.array(s['train']), np.array(s['val'])) for s in saved['folds']]
        val_datasets = [None] * len(folds)
        print(f'Split loaded: {cfg.split_file} ({len(folds)} fold(s), '
              f'{len(folds[0][0])} train / {len(folds[0][1])} val)')
    elif cfg.test_data is not None:
        test_cfg = dataclasses.replace(cfg, data=cfg.test_data)
        hold_out_ds = build_dataset(test_cfg)
        idx = np.random.permutation(n)
        split = int(n * (1 - cfg.val_frac))
        folds = [(idx[:split], idx[split:])]
        val_datasets = [None]
        print(f'Hold-out split: {split} train / {n - split} val '
              f'(val from train CSV, test CSV evaluated once at end)')
    elif cfg.folds == 1:
        idx = np.random.permutation(n)
        split = int(n * (1 - cfg.val_frac))
        folds = [(idx[:split], idx[split:])]
        val_datasets = [None]
    elif cfg.stratify:
        labels = np.array([dataset[i]['label'] for i in range(n)])
        folds = stratified_kfold_indices(labels, k=cfg.folds, seed=cfg.seed)
        val_datasets = [None] * len(folds)
    else:
        folds = kfold_indices(n, k=cfg.folds, seed=cfg.seed)
        val_datasets = [None] * len(folds)

    if cfg.split_file and not os.path.exists(cfg.split_file):
        os.makedirs(os.path.dirname(os.path.abspath(cfg.split_file)), exist_ok=True)
        with open(cfg.split_file, 'w') as f:
            json.dump({'folds': [{'train': tr.tolist(), 'val': va.tolist()}
                                  for tr, va in folds]}, f)
        print(f'Split saved: {cfg.split_file}')

    return folds, val_datasets, hold_out_ds


# ─── Single-variant training loop (trimmed copy of train_utr.train_fold) ──────

def run_variant_fold(
    cfg: TrainConfig,
    variant: str,
    model: nn.Module,
    dataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold_num: int,
    val_dataset=None,
    test_dataset=None,
    out_dir: str = 'outputs',
) -> Tuple[Dict[str, float], int, float]:
    """Returns (metrics, n_params, elapsed_seconds)."""
    device = torch.device(cfg.device)
    task = 'classification' if cfg.task == 'ires' else 'regression'

    train_ds = Subset(dataset, train_idx)
    val_ds = val_dataset if val_dataset is not None else Subset(dataset, val_idx)

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

    model = model.to(device)
    n_params = model.get_num_params()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_loader)
    sched = WarmupCosineScheduler(opt, cfg.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler('cuda') if (cfg.use_amp and device.type == 'cuda') else None

    best_score = -np.inf
    best_state: Optional[Dict] = None
    best_epoch = 0
    best_metrics: Dict[str, float] = {}
    no_improve = 0
    start_epoch = 1

    os.makedirs(out_dir, exist_ok=True)
    resume_path = os.path.join(out_dir, f'{cfg.task}_fold{fold_num}_resume.pt')

    # ── Resume from a Ctrl-C / crash within this variant, if a checkpoint exists ──
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        opt.load_state_dict(ckpt['optimizer'])
        sched.load_state_dict(ckpt['scheduler'])
        if scaler and ckpt.get('scaler'):
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        best_score = ckpt['best_score']
        best_epoch = ckpt['best_epoch']
        best_metrics = ckpt['best_metrics']
        best_state = ckpt['best_state']
        no_improve = ckpt['no_improve']
        print(f'  [{variant}] Resumed from epoch {ckpt["epoch"]} '
              f'(best_score={best_score:.4f} @ epoch {best_epoch})')

    val_size = len(val_dataset) if val_dataset is not None else len(val_idx)
    amp_tag = 'AMP' if scaler else 'fp32'
    print(f'\n  [{variant}] Fold {fold_num} | {len(train_idx)} train / {val_size} val '
          f'| {n_params:,} params | {amp_tag} | eval_every={cfg.eval_every}')

    def _save_resume(epoch: int):
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': opt.state_dict(),
            'scheduler': sched.state_dict(),
            'scaler': scaler.state_dict() if scaler else None,
            'best_score': best_score,
            'best_epoch': best_epoch,
            'best_metrics': best_metrics,
            'best_state': best_state,
            'no_improve': no_improve,
        }, resume_path)

    t_start = time.time()
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, opt, sched, device,
                                  cfg.clip_grad, scaler, compute_loss_fn=None)
        elapsed = time.time() - t0

        should_eval = (epoch % cfg.eval_every == 0) or (epoch == cfg.epochs)
        if should_eval:
            metrics = evaluate(model, val_loader, device, task)
            score = primary_metric(metrics, task)

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_metrics = metrics.copy()
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            m_str = ' | '.join(f'{k}={v:.4f}' for k, v in metrics.items())
            print(f'    [{variant}] E{epoch:03d} loss={train_loss:.4f} | {m_str} '
                  f'[{elapsed:.1f}s] {"*" if no_improve == 0 else ""}')

            _save_resume(epoch)

            if no_improve >= cfg.patience:
                print(f'    [{variant}] Early stop at epoch {epoch} '
                      f'(no improvement for {cfg.patience} evals)')
                break
        else:
            print(f'    [{variant}] E{epoch:03d} loss={train_loss:.4f} [{elapsed:.1f}s]')
            _save_resume(epoch)

    total_elapsed = time.time() - t_start
    print(f'  [{variant}] Best @ epoch {best_epoch}: '
          + ' | '.join(f'{k}={v:.4f}' for k, v in best_metrics.items()))

    if best_state is not None:
        best_path = os.path.join(out_dir, f'{cfg.task}_fold{fold_num}_best.pt')
        torch.save({
            'state_dict': best_state,
            'metrics': best_metrics,
            'variant': variant,
            'best_epoch': best_epoch,
        }, best_path)
        print(f'  [{variant}] Saved -> {best_path}')

    if test_dataset is not None and best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        test_loader = DataLoader(
            test_dataset, batch_size=cfg.batch_size * 2, shuffle=False,
            collate_fn=collate_utr, num_workers=cfg.num_workers,
            pin_memory=(cfg.device != 'cpu'),
        )
        test_metrics = evaluate(model, test_loader, device, task)
        t_str = ' | '.join(f'{k}={v:.4f}' for k, v in test_metrics.items())
        print(f'  [{variant}] Test set:  {t_str}')
        return test_metrics, n_params, total_elapsed

    return best_metrics, n_params, total_elapsed


# ─── Comparison table ──────────────────────────────────────────────────────────

def print_ablation_table(results: Dict[str, Dict]):
    print('\n' + '=' * 88)
    print('  ABLATION SUMMARY')
    print('=' * 88)
    metric_keys = ['mse', 'r2', 'pearson_r', 'spearman_r', 'aupr']
    present_keys = [k for k in metric_keys
                    if any(k in r.get('metrics', {}) for r in results.values())]

    header = f'  {"variant":<18}' + ''.join(f'{k:>12}' for k in present_keys) \
             + f'{"params":>14}{"train_min":>12}'
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for variant, r in results.items():
        m = r.get('metrics', {})
        row = f'  {variant:<18}'
        for k in present_keys:
            v = m.get(k)
            row += f'{v:>12.4f}' if v is not None else f'{"---":>12}'
        row += f'{r.get("n_params", 0):>14,}'
        row += f'{r.get("elapsed_min", 0):>12.1f}'
        print(row)
    print('=' * 88)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> Tuple[TrainConfig, argparse.Namespace]:
    p = argparse.ArgumentParser(
        description='RNA Bender geometry ablation harness (standalone; edits no existing files)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data (same names/semantics as train_utr.py)
    p.add_argument('--task', default='mrl', choices=['mrl', 'te', 'el', 'ires', 'rlu'])
    p.add_argument('--data', required=True)
    p.add_argument('--test_data', default=None)
    p.add_argument('--bpp_backend', default='mfe', choices=['viennarna', 'mfe', 'zero'])
    p.add_argument('--bpp_cache_dir', default='~/bpp_cache')
    p.add_argument('--seq_col', default=None)
    p.add_argument('--label_col', default=None)
    p.add_argument('--lib_col', default=None)
    p.add_argument('--cell_line', default=None)
    p.add_argument('--max_len', type=int, default=None)
    # Model (shared by full / no_bb_curv / bilinear_pair / no_struct_edges)
    p.add_argument('--model_dim', type=int, default=96)
    p.add_argument('--num_layers', type=int, default=3)
    p.add_argument('--reduced_dim', type=int, default=16)
    p.add_argument('--ff_dim', type=int, default=None)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--pooling', default='attention', choices=['attention', 'mean'])
    p.add_argument('--pos_emb_type', default='sinusoidal', choices=['sinusoidal', 'learned'])
    p.add_argument('--no_pair_head', action='store_true')
    # Aux / geometric losses
    p.add_argument('--aux_struct', action='store_true')
    p.add_argument('--lambda_ss', type=float, default=0.1)
    p.add_argument('--lambda_mfe', type=float, default=0.05)
    p.add_argument('--lambda_curv', type=float, default=0.001)
    p.add_argument('--lambda_cons', type=float, default=0.0)
    p.add_argument('--lambda_pair', type=float, default=0.1)
    # Transformer-only overrides (parameter matching)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--transformer_model_dim', type=int, default=None)
    p.add_argument('--transformer_num_layers', type=int, default=None)
    p.add_argument('--transformer_ff_dim', type=int, default=None)
    # Training
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--clip_grad', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--warmup_steps', type=int, default=200)
    p.add_argument('--eval_every', type=int, default=1)
    # Evaluation
    p.add_argument('--folds', type=int, default=1)
    p.add_argument('--val_frac', type=float, default=0.2)
    p.add_argument('--no_stratify', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--split_file', default=None)
    # Speed / runtime
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--device', default='auto')
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--output_dir', default='outputs/ablation_bender')
    # Which variants to run
    p.add_argument('--variants', nargs='+', default=VARIANT_CHOICES, choices=VARIANT_CHOICES)

    args = p.parse_args()

    cfg = TrainConfig(
        task=args.task, data=args.data, test_data=args.test_data,
        bpp_backend=args.bpp_backend, bpp_cache_dir=args.bpp_cache_dir,
        seq_col=args.seq_col, label_col=args.label_col, lib_col=args.lib_col,
        cell_line=args.cell_line, max_len=args.max_len,
        model_type='bender',   # unused by build_dataset/_auto_fill; kept for clarity
        model_dim=args.model_dim, num_layers=args.num_layers,
        reduced_dim=args.reduced_dim, ff_dim=args.ff_dim, dropout=args.dropout,
        pooling=args.pooling, pos_emb_type=args.pos_emb_type,
        use_pair_head=not args.no_pair_head,
        aux_struct=args.aux_struct, lambda_ss=args.lambda_ss, lambda_mfe=args.lambda_mfe,
        lambda_curv=args.lambda_curv, lambda_cons=args.lambda_cons, lambda_pair=args.lambda_pair,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, clip_grad=args.clip_grad,
        patience=args.patience, warmup_steps=args.warmup_steps, eval_every=args.eval_every,
        folds=args.folds, val_frac=args.val_frac, stratify=not args.no_stratify,
        seed=args.seed, split_file=args.split_file,
        use_amp=not args.no_amp, device=args.device, num_workers=args.num_workers,
        output_dir=args.output_dir,
    )
    return _auto_fill(cfg), args


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg, args = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    print(f'Task: {cfg.task} | Data: {cfg.data} | Device: {cfg.device}')
    print(f'Backbone: dim={cfg.model_dim} layers={cfg.num_layers} r={cfg.reduced_dim}')
    print(f'Variants: {args.variants}')

    os.makedirs(cfg.output_dir, exist_ok=True)
    results_path = os.path.join(cfg.output_dir, 'results.json')
    results: Dict[str, Dict] = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        print(f'Loaded existing results: {list(results.keys())}')

    # ── Build datasets once, shared across variants that use them ────────────
    # 'full' / 'no_bb_curv' / 'bilinear_pair' / 'transformer' all read the same
    # bpp_backend=mfe dataset (transformer ignores edges entirely but sharing
    # the object avoids re-folding sequences). 'no_struct_edges' needs its own
    # dataset built with bpp_backend='zero' (see module docstring).
    print(f'\nBuilding primary dataset (bpp_backend={cfg.bpp_backend})...')
    dataset_mfe = build_dataset(cfg)
    n = len(dataset_mfe)
    print(f'Dataset: {n} sequences')

    folds, val_datasets, hold_out_ds = build_folds(cfg, dataset_mfe, n)
    if len(folds) > 1:
        print(f'WARNING: {len(folds)} folds requested — this script runs and '
              f'reports fold 1 only per variant for simplicity. Re-run with '
              f'--folds 1 and a --split_file for the documented workflow.')
    train_idx, val_idx = folds[0]
    val_dataset0 = val_datasets[0]

    dataset_zero = None
    hold_out_zero = None
    if 'no_struct_edges' in args.variants:
        print(f'\nBuilding structure-free dataset (bpp_backend=zero)...')
        cfg_zero = dataclasses.replace(cfg, bpp_backend='zero')
        dataset_zero = build_dataset(cfg_zero)
        if cfg.test_data is not None:
            test_cfg_zero = dataclasses.replace(cfg_zero, data=cfg.test_data)
            hold_out_zero = build_dataset(test_cfg_zero)

    # ── Run each requested variant ────────────────────────────────────────────
    for variant in args.variants:
        if variant in results:
            print(f'\n[skip] {variant} (already in results.json)')
            continue

        dataset = dataset_zero if variant == 'no_struct_edges' else dataset_mfe
        test_ds = hold_out_zero if variant == 'no_struct_edges' else hold_out_ds

        model = build_variant_model(variant, cfg, args)
        out_dir = os.path.join(cfg.output_dir, variant)

        metrics, n_params, elapsed = run_variant_fold(
            cfg, variant, model, dataset, train_idx, val_idx,
            fold_num=1, val_dataset=val_dataset0, test_dataset=test_ds,
            out_dir=out_dir,
        )
        results[variant] = {
            'metrics': metrics,
            'n_params': n_params,
            'elapsed_min': round(elapsed / 60, 2),
        }
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

    print_ablation_table(results)
    print(f'\nResults JSON: {results_path}')


if __name__ == '__main__':
    main()
