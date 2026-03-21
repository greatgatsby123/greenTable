"""
RNA Hybrid Model V2: Geometry Encoder → Structure Bottleneck → MRL Predictor

Two-stage design:

  Stage A — Geometry encoder + Structure bottleneck (V2)
    BenderEncoder produces H_geom, pair_logits P, SS logits S, MFE scalar,
    and last-layer curvature K.  V2 StructureBottleneck compresses these into:

        Per-token bottleneck B_tok (R^{struct_bottleneck_dim}):
            Content branch  [H_geom ∥ partner_ctx_mean ∥ partner_ctx_topk ∥ curv]
            Stats branch    [pair_mass ∥ max_pair_prob ∥ unpaired ∥ pair_entropy
                             ∥ ss_entropy ∥ local_pair_mass ∥ distal_pair_mass
                             ∥ win_up_w3 ∥ win_up_w7 ∥ win_ent_w7
                             ∥ ss_prob ∥ ss_logit_proj]
            B_tok = merge(content_mlp(content_in), stats_mlp(stats_in))

        Per-token stats B_stats_tok (R^{N_STATS_V2=10}):
            Interpretable scalar channels exported alongside B_tok for
            direct injection into Stage B.

        Global bottleneck B_glob (R^{glob_bottleneck_dim}):
            AttnPool(B_tok) ∥ mean_pair_degree ∥ mfe_hat
            ∥ pe_mean ∥ pe_std ∥ up_mean ∥ up_min ∥ lb_mean ∥ lb_max
            ∥ frac_high_pair ∥ frac_uncertain

  Stage B — MRL predictor
    A small sequence encoder produces H_seq.
    Structural stats are injected directly: H_seq ← H_seq + proj(B_stats_tok)
    A cross-attention bridge injects structure content:

        H' = CrossAttn(Q=H_seq + proj(B_stats_tok), K=V=B_tok)
        y  = MRL_head([AttnPool(H') ∥ B_glob])

Training phases (what is actually implemented)
    Phase 1  Pretrain BenderPretrainModel on folding via pretrain_bender.py.
             Checkpoint saves: geom_encoder, pair_head, ss_head, mfe_head, mfe_pool.
             NOT saved: struct_bottleneck, global_bottleneck (hybrid-specific; random).

    Phase 2  Build RNAHybridModel, load Phase 1 checkpoint.
             freeze_encoder_and_heads() freezes pretrained components only.
             struct_bottleneck, global_bottleneck, and Stage B train at full LR.
             (--freeze_geom_epochs N)

    Phase 3  unfreeze_encoder_and_heads(); full model trains jointly.
             Pretrained components: base_lr × geom_lr_scale (prevent forgetting).
             Bottlenecks + Stage B: base_lr.
             (--geom_lr_scale 0.1)

Forward API
    Accepts same dict API as RNABenderModel and RNAMoEMRLModel — both
    edge_idx/edge_feat (folding collate) and edge_index/edge_attrs (UTR collate).

Bottleneck modes
    'v2'      Default.  Full uncertainty + multi-view + windowed features.
    'v1'/'full' Original pair_degree + ss_prob version (kept for ablation).
    'simple'  Minimal: H_geom + partner_ctx + curv (when heads are untrained).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from rna_encoders import RNABenderEncoder, RNASequenceEncoder, _pool, _init_weights
from rna_bender import (
    VOCAB_SIZE, SS_IGNORE_IDX,
    PairMapHead, _consistency_loss,
)

# Scalar channels in B_stats_tok (per-token interpretable stats, mode='v2')
N_STATS_V2 = 10
# Global heterogeneity scalars added to GlobalBottleneck input when n_stats > 0
_N_GLOB_STATS = 8


# ─── Windowed masked mean helper ─────────────────────────────────────────────

def _win_mean(vals: torch.Tensor, mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Masked windowed mean over a sequence.

    For each position i, averages vals over the window [i-radius, i+radius],
    counting only valid (unmasked) positions in the denominator.

    Args:
        vals  : (B, L) — scalar values; should already be 0 at padding positions
        mask  : (B, L) — float mask  (1 = valid token, 0 = padding)
        radius: half-window size; full window = 2*radius + 1

    Returns:
        (B, L) — windowed mean, normalised by count of valid positions in window
    """
    kernel = 2 * radius + 1
    ones   = vals.new_ones(1, 1, kernel)           # (1, 1, k)  — tiny; OK to create inline
    x  = (vals * mask).unsqueeze(1)                # (B, 1, L)
    m  = mask.unsqueeze(1)                         # (B, 1, L)
    sx = F.conv1d(x, ones, padding=radius)         # (B, 1, L)
    sm = F.conv1d(m, ones, padding=radius)         # (B, 1, L)
    return (sx / sm.clamp(min=1)).squeeze(1)       # (B, L)


# ─── Structure bottleneck V2 ──────────────────────────────────────────────────

class StructureBottleneck(nn.Module):
    """
    Per-token bottleneck: compresses geometry encoder outputs into b_i ∈ R^{bottleneck_dim}.

    Modes
    -----
    'v2' (default)
        Two-branch fusion with temperature-softened features, uncertainty channels,
        top-k multi-view partner context, and windowed accessibility.

        Content branch  : geometric representation content
            [H_geom ∥ partner_ctx_mean ∥ partner_ctx_topk ∥ curv_compressed]

        Stats branch    : structural statistics and accessibility
            scalar stats (10): pair_mass, max_pair_prob, unpaired_score,
                                pair_entropy, ss_entropy, local_pair_mass,
                                distal_pair_mass, win_up_±2, win_up_±5, win_ent_±5
            ss_prob (3)  : softmax(ss_logits / tau_ss)
            ss_logit_proj: linear projection of raw ss_logits (preserves confidence scale)

        B_stats_tok (B, L, N_STATS_V2=10) is also returned for direct injection
        into Stage B and for global bottleneck heterogeneity features.

    'v1' / 'full'
        Original version: [H_geom ∥ partner_ctx ∥ pair_degree ∥ ss_prob ∥ curv].
        Returns (B_tok, None).

    'simple'
        Minimal: [H_geom ∥ partner_ctx ∥ curv].  Useful when pair/SS heads are
        not yet pretrained.  Returns (B_tok, None).
    """

    def __init__(
        self,
        geom_dim:            int,
        reduced_dim:         int,      # geom encoder's reduced_dim (determines plu_dim)
        bottleneck_dim:      int   = 64,
        curv_out:            int   = 16,
        mode:                str   = 'v2',
        # V2 softening temperatures
        tau_pair:            float = 1.5,    # sigmoid temperature for pair probabilities
        tau_attn:            float = 1.5,    # softmax temperature for partner attention
        tau_ss:              float = 1.25,   # softmax temperature for SS probabilities
        # V2 partner context
        topk_partners:       int   = 3,
        # V2 local/distal boundary (positions within ±window are "local")
        local_distal_window: int   = 8,
        # V2 SS logit projection dim
        ss_proj_dim:         int   = 8,
        # V2 internal branch hidden sizes
        content_dim:         int   = 64,
        stats_dim:           int   = 32,
    ):
        super().__init__()
        assert mode in ('simple', 'v1', 'full', 'v2'), f'Unknown bottleneck mode: {mode!r}'
        self.mode                = mode
        self.tau_pair            = tau_pair
        self.tau_attn            = tau_attn
        self.tau_ss              = tau_ss
        self.topk_partners       = topk_partners
        self.local_distal_window = local_distal_window

        plu_dim = reduced_dim * (reduced_dim - 1) // 2

        # Curvature compressor — shared across all modes
        self.curv_mlp = nn.Sequential(
            nn.Linear(plu_dim, max(curv_out * 2, plu_dim // 2)),
            nn.GELU(),
            nn.Linear(max(curv_out * 2, plu_dim // 2), curv_out),
        )

        if mode == 'v2':
            self.ss_logit_proj = nn.Linear(3, ss_proj_dim)
            # Normalise 10 scalar stats before stats_mlp (they live on very different scales)
            self.stats_input_ln = nn.LayerNorm(N_STATS_V2)

            # Content branch: H_geom + partner_ctx_mean + partner_ctx_topk + curv
            content_in_dim = geom_dim + geom_dim + geom_dim + curv_out
            self.content_mlp = nn.Sequential(
                nn.Linear(content_in_dim, content_dim * 2),
                nn.GELU(),
                nn.Linear(content_dim * 2, content_dim),
            )

            # Stats branch: 10 scalar stats + ss_prob(3) + ss_logit_proj(ss_proj_dim)
            stats_in_dim = N_STATS_V2 + 3 + ss_proj_dim
            self.stats_mlp = nn.Sequential(
                nn.Linear(stats_in_dim, stats_dim * 2),
                nn.GELU(),
                nn.Linear(stats_dim * 2, stats_dim),
            )

            # Merge branches
            self.merge_mlp = nn.Sequential(
                nn.Linear(content_dim + stats_dim, bottleneck_dim * 2),
                nn.GELU(),
                nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            )

        else:
            # v1/full and simple share a single fuse_mlp
            if mode in ('v1', 'full'):
                fuse_in = geom_dim + geom_dim + 1 + 3 + curv_out
            else:   # 'simple'
                fuse_in = geom_dim + geom_dim + curv_out
            self.fuse_mlp = nn.Sequential(
                nn.Linear(fuse_in, bottleneck_dim * 2),
                nn.GELU(),
                nn.Linear(bottleneck_dim * 2, bottleneck_dim),
            )

        _init_weights(self)

    def forward(
        self,
        H_geom:      torch.Tensor,   # (B, L, geom_dim)
        pair_logits: torch.Tensor,   # (B, L, L)
        ss_logits:   torch.Tensor,   # (B, L, 3)
        kappa_last:  torch.Tensor,   # (B, L, plu_dim)
        seq_mask:    torch.Tensor,   # (B, L) bool  True=valid
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns
        -------
        B_tok       : (B, L, bottleneck_dim)
        B_stats_tok : (B, L, N_STATS_V2) — interpretable scalar channels (v2 only)
                      None for v1/simple modes
        """
        mf = seq_mask.float()          # (B, L)
        B, L, _ = H_geom.shape

        # Mask diagonal (self-pairing), invalid column positions, and invalid row positions.
        # Row masking (~seq_mask.unsqueeze(2)) ensures that pair-attention from padding
        # tokens is zero before the softmax, not just zeroed out afterwards.
        diag  = torch.eye(L, device=pair_logits.device, dtype=torch.bool).unsqueeze(0)
        clean = pair_logits.masked_fill(
            diag | ~seq_mask.unsqueeze(1) | ~seq_mask.unsqueeze(2), -1e4
        )

        # Curvature (shared, all modes)
        curv = self.curv_mlp(kappa_last)                               # (B, L, curv_out)

        # ── V1 / simple ────────────────────────────────────────────────────────
        if self.mode in ('simple', 'v1', 'full'):
            partner_w = torch.softmax(clean, dim=-1)
            partner_w = torch.nan_to_num(partner_w, nan=0.0)
            partner_ctx = torch.bmm(partner_w, H_geom)                # (B, L, geom_dim)

            if self.mode in ('v1', 'full'):
                pair_degree = torch.sigmoid(clean).sum(-1, keepdim=True)  # (B, L, 1)
                ss_prob     = torch.softmax(ss_logits, dim=-1)            # (B, L, 3)
                fuse_in = torch.cat([H_geom, partner_ctx, pair_degree, ss_prob, curv], dim=-1)
            else:
                fuse_in = torch.cat([H_geom, partner_ctx, curv], dim=-1)

            B_tok = self.fuse_mlp(fuse_in) * mf.unsqueeze(-1)
            return B_tok, None

        # ── V2 ─────────────────────────────────────────────────────────────────

        # Temperature-softened pair tensors
        pair_prob = torch.sigmoid(clean / self.tau_pair)               # (B, L, L)  ~0 at diag/pad
        pair_attn = torch.softmax(clean / self.tau_attn, dim=-1)
        pair_attn = torch.nan_to_num(pair_attn, nan=0.0)              # (B, L, L)

        # ── Uncertainty / accessibility statistics ─────────────────────────────
        eps = 1e-8
        pair_entropy   = -(pair_attn * (pair_attn + eps).log()).sum(-1, keepdim=True)  # (B, L, 1)
        max_pair_prob  = pair_prob.max(-1).values.unsqueeze(-1)                        # (B, L, 1)
        unpaired_score = 1.0 - max_pair_prob                                           # (B, L, 1)

        # Valid partner count per token: non-self positions where BOTH i and j are valid.
        # seq_mask.unsqueeze(1): valid query rows; seq_mask.unsqueeze(2): valid columns (partners).
        valid_pair_mask = (~diag) & seq_mask.unsqueeze(1) & seq_mask.unsqueeze(2)  # (B, L, L)
        n_valid_j       = valid_pair_mask.float().sum(-1, keepdim=True).clamp(min=1)  # (B, L, 1)

        # Normalised pair mass: average pair probability per eligible partner
        pair_mass = pair_prob.sum(-1, keepdim=True) / n_valid_j        # (B, L, 1)  ∈ [0, 1]

        # SS features
        ss_prob    = torch.softmax(ss_logits / self.tau_ss, dim=-1)            # (B, L, 3)
        ss_entropy = -(ss_prob * (ss_prob + eps).log()).sum(-1, keepdim=True)  # (B, L, 1)

        # Normalised pair entropy: raw / log(eligible_partners).
        # Raw entropy grows with n_valid_j, so a fixed threshold confounds length and ambiguity.
        # Normalised entropy ∈ [0, 1]: 0 = perfectly paired, 1 = maximally diffuse.
        pair_entropy_norm = pair_entropy / (n_valid_j.float().log().clamp(min=eps))  # (B, L, 1)
        ss_lp      = self.ss_logit_proj(ss_logits)                             # (B, L, ss_proj_dim)

        # ── Multi-view partner context ─────────────────────────────────────────
        # Mean: soft expectation over all partners
        partner_ctx_mean = torch.bmm(pair_attn, H_geom)                       # (B, L, geom_dim)

        # Top-k: concentrate on strongest predicted partners
        k = min(self.topk_partners, L)
        topk_vals, topk_idx = torch.topk(clean, k=k, dim=-1)                  # (B, L, k)
        # Zero out top-k entries that correspond to masked positions (tied at -1e4)
        topk_valid = (topk_vals > -9000.0).float()                             # (B, L, k)
        topk_w = torch.softmax(
            topk_vals.masked_fill(topk_valid == 0, -1e4) / self.tau_attn, dim=-1
        ) * topk_valid                                                          # (B, L, k)
        topk_w = topk_w / topk_w.sum(-1, keepdim=True).clamp(min=1e-8)        # renormalise
        topk_w = torch.nan_to_num(topk_w, nan=0.0)
        b_idx  = torch.arange(B, device=H_geom.device).view(B, 1, 1).expand(B, L, k)
        H_topk = H_geom[b_idx, topk_idx]                                      # (B, L, k, geom_dim)
        partner_ctx_topk = (topk_w.unsqueeze(-1) * H_topk).sum(dim=2)         # (B, L, geom_dim)

        # ── Local vs distal pair mass (normalised) ─────────────────────────────
        # dist_mat: row = position i, col = position j  →  correct i/j semantics
        w   = self.local_distal_window
        row = torch.arange(L, device=pair_prob.device).view(1, L, 1)
        col = torch.arange(L, device=pair_prob.device).view(1, 1, L)
        dist_mat = (row - col).abs().float()                                   # (1, L, L)
        local_mask  = (dist_mat <= w).float()
        distal_mask = (dist_mat >  w).float()
        # Count valid local/distal partners per token for normalisation
        n_local  = (valid_pair_mask.float() * local_mask).sum(-1, keepdim=True).clamp(min=1)
        n_distal = (valid_pair_mask.float() * distal_mask).sum(-1, keepdim=True).clamp(min=1)
        local_pair_mass  = (pair_prob * local_mask).sum(-1, keepdim=True)  / n_local   # (B, L, 1)
        distal_pair_mass = (pair_prob * distal_mask).sum(-1, keepdim=True) / n_distal  # (B, L, 1)

        # ── Windowed accessibility ─────────────────────────────────────────────
        up_flat  = unpaired_score.squeeze(-1)       # (B, L)
        pen_flat = pair_entropy_norm.squeeze(-1)   # (B, L)  normalised entropy ∈ [0, 1]
        win_up_w3  = _win_mean(up_flat,  mf, 2).unsqueeze(-1)   # ±2  (B, L, 1)
        win_up_w7  = _win_mean(up_flat,  mf, 5).unsqueeze(-1)   # ±5  (B, L, 1)
        win_ent_w7 = _win_mean(pen_flat, mf, 5).unsqueeze(-1)   # ±5  (B, L, 1)

        # ── B_stats_tok: 10 interpretable scalar channels ─────────────────────
        # Channel index:  0          1             2               3 (norm entropy)
        #                 4            5                6
        #                 7          8          9
        # Channels 3 and 9 use normalised pair entropy (÷ log n_valid_j) so that
        # the uncertainty signal is length-independent.
        B_stats_tok = torch.cat([
            pair_mass, max_pair_prob, unpaired_score, pair_entropy_norm, ss_entropy,
            local_pair_mass, distal_pair_mass,
            win_up_w3, win_up_w7, win_ent_w7,
        ], dim=-1) * mf.unsqueeze(-1)                                          # (B, L, 10)

        # ── Two-branch fusion ──────────────────────────────────────────────────
        content_in = torch.cat([H_geom, partner_ctx_mean, partner_ctx_topk, curv], dim=-1)
        # Normalise scalar stats (they span very different scales) before passing to stats_mlp
        stats_in   = torch.cat([self.stats_input_ln(B_stats_tok), ss_prob, ss_lp], dim=-1)

        content_h = self.content_mlp(content_in)                              # (B, L, content_dim)
        stats_h   = self.stats_mlp(stats_in)                                  # (B, L, stats_dim)
        B_tok     = self.merge_mlp(torch.cat([content_h, stats_h], dim=-1))   # (B, L, bottleneck_dim)
        B_tok     = B_tok * mf.unsqueeze(-1)

        return B_tok, B_stats_tok


# ─── Global bottleneck ────────────────────────────────────────────────────────

class GlobalBottleneck(nn.Module):
    """
    Summarises B_tok into a fixed-size global vector B_glob ∈ R^{glob_dim}.

    When n_stats > 0 (V2 mode), also receives B_stats_tok and adds 8 global
    heterogeneity descriptors that capture distributional structure beyond the
    mean: entropy spread, accessibility variability, local burden range,
    and fractions of high-pairing / high-uncertainty positions.

    These are important for MRL because ribosome scanning depends not just on
    average structure but on whether a barrier is concentrated or distributed.
    """

    def __init__(self, tok_dim: int = 64, glob_dim: int = 128, n_stats: int = 0):
        super().__init__()
        self.n_stats   = n_stats
        self.pool_attn = nn.Linear(tok_dim, 1)
        # Base input: pooled B_tok (tok_dim) + mean_pair_degree (1) + mfe_hat (1)
        # V2 extension: + _N_GLOB_STATS global heterogeneity scalars
        glob_in = tok_dim + 2 + (_N_GLOB_STATS if n_stats > 0 else 0)
        self.glob_mlp = nn.Sequential(
            nn.Linear(glob_in, glob_dim * 2),
            nn.GELU(),
            nn.Linear(glob_dim * 2, glob_dim),
        )
        _init_weights(self)

    def forward(
        self,
        B_tok:       torch.Tensor,             # (B, L, tok_dim)
        pair_logits: torch.Tensor,             # (B, L, L)
        mfe_hat:     torch.Tensor,             # (B,)
        seq_mask:    torch.Tensor,             # (B, L) bool
        B_stats_tok: Optional[torch.Tensor] = None,  # (B, L, N_STATS_V2) or None
    ) -> torch.Tensor:                         # (B, glob_dim)
        mf      = seq_mask.float()             # (B, L)
        n_valid = mf.sum(-1).clamp(min=1)     # (B,)

        # Attention pool over B_tok
        pooled = _pool(B_tok, seq_mask, self.pool_attn)                      # (B, tok_dim)

        # Mean normalised pair mass.
        # In V2 (n_stats > 0) we reuse the tau-consistent per-token pair_mass already
        # computed by StructureBottleneck (B_stats_tok channel 0), avoiding a second
        # recomputation with a different calibration.
        # In V1 (n_stats == 0) we fall back to recomputing from pair_logits.
        if self.n_stats > 0 and B_stats_tok is not None:
            mean_pair_degree = (B_stats_tok[..., 0] * mf).sum(-1) / n_valid  # (B,)
        else:
            L     = pair_logits.shape[-1]
            diag  = torch.eye(L, device=pair_logits.device, dtype=torch.bool).unsqueeze(0)
            clean = pair_logits.masked_fill(diag | ~seq_mask.unsqueeze(1), -1e4)
            pair_probs       = torch.sigmoid(clean)
            mean_pair_degree = (pair_probs * mf.unsqueeze(1)).sum(-1)        # (B, L)
            mean_pair_degree = (mean_pair_degree * mf).sum(-1) / n_valid     # (B,)

        parts = [pooled, mean_pair_degree.unsqueeze(-1), mfe_hat.unsqueeze(-1)]

        if self.n_stats > 0 and B_stats_tok is not None:
            # B_stats_tok channel layout:
            #   0: pair_mass   1: max_pair_prob   2: unpaired_score   3: pair_entropy
            #   4: ss_entropy  5: local_pair_mass  6: distal_pair_mass
            #   7: win_up_w3   8: win_up_w7        9: win_ent_w7

            def _mean(ch: int) -> torch.Tensor:
                return (B_stats_tok[..., ch] * mf).sum(-1) / n_valid

            pe = B_stats_tok[..., 3]   # pair_entropy  (B, L)
            up = B_stats_tok[..., 2]   # unpaired_score (B, L)
            lb = B_stats_tok[..., 5]   # local_pair_mass (B, L)

            pe_mean = _mean(3)
            pe_var  = ((pe - pe_mean.unsqueeze(-1)).pow(2) * mf).sum(-1) / n_valid
            pe_std  = pe_var.clamp(min=0).sqrt()

            up_mean = _mean(2)
            # Min unpairedness over valid positions (most constrained token).
            # Use inf sentinel so padding can never win the min.
            up_min  = up.masked_fill(~seq_mask, float('inf')).min(-1).values  # (B,)

            lb_mean = _mean(5)
            lb_max  = (lb * mf).max(-1).values                              # (B,)

            # Fraction with a dominant pairing partner (max_pair_prob > 0.5).
            # Uses channel 1 (max_pair_prob ∈ [0,1]), which is scale-independent
            # unlike raw pair_mass which grows with sequence length.
            frac_high_pair  = ((B_stats_tok[..., 1] > 0.5).float() * mf).sum(-1) / n_valid
            # Channel 3 is normalised entropy ∈ [0, 1]; > 0.5 means > half max entropy.
            frac_uncertain  = ((pe > 0.5).float() * mf).sum(-1) / n_valid

            glob_stats = torch.stack(
                [pe_mean, pe_std, up_mean, up_min, lb_mean, lb_max,
                 frac_high_pair, frac_uncertain], dim=-1
            )                                                                # (B, 8)
            parts.append(glob_stats)

        return self.glob_mlp(torch.cat(parts, dim=-1))                      # (B, glob_dim)


# ─── Cross-attention bridge (Stage B structure injection) ─────────────────────

class CrossAttentionBridge(nn.Module):
    """
    Injects structure content (B_tok) into sequence representations (H_seq)
    via multi-head cross-attention.

        Q from H_seq  (seq_dim)
        K, V from B_tok  (tok_dim → projected internally by nn.MHA)

    Pre-LN residual:
        H_tilde = H_seq + out_proj(drop(MHA(LN(H_seq), B_tok, B_tok)))
    """

    def __init__(
        self,
        seq_dim:   int,
        tok_dim:   int,
        num_heads: int   = 4,
        dropout:   float = 0.1,
    ):
        super().__init__()
        self.ln      = nn.LayerNorm(seq_dim)
        self.mha     = nn.MultiheadAttention(
            embed_dim   = seq_dim,
            num_heads   = num_heads,
            kdim        = tok_dim,
            vdim        = tok_dim,
            batch_first = True,
            dropout     = dropout,
        )
        self.out_proj = nn.Linear(seq_dim, seq_dim)
        self.drop     = nn.Dropout(dropout)
        _init_weights(self)

    def forward(
        self,
        H_seq:    torch.Tensor,   # (B, L, seq_dim)
        B_tok:    torch.Tensor,   # (B, L, tok_dim)
        seq_mask: torch.Tensor,   # (B, L) bool  True=valid
    ) -> torch.Tensor:            # (B, L, seq_dim)
        H_norm  = self.ln(H_seq)
        # key_padding_mask: True = IGNORE (i.e. padding positions)
        H_cross, _ = self.mha(H_norm, B_tok, B_tok, key_padding_mask=~seq_mask)
        return H_seq + self.drop(self.out_proj(H_cross))


# ─── Full hybrid model ─────────────────────────────────────────────────────────

class RNAHybridModel(nn.Module):
    """
    Two-stage RNA model for MRL prediction.

    Stage A: geometry encoder + interpretable structure bottleneck (V2 default)
    Stage B: sequence encoder + stats injection + cross-attention from structure

    Supports:
      - pretrained geometry encoder via load_pretrained_geom()
      - freeze_encoder_and_heads() for Phase 2 (safe; bottleneck stays trainable)
      - freeze_stage_a() for full Stage A freeze (ONLY if bottleneck was pretrained too)
      - three-group LR via get_optimizer_groups() (pretrained/bottleneck/stage_b)
      - edge name aliases: edge_idx/edge_feat (folding) or edge_index/edge_attrs (UTR)
    """

    def __init__(
        self,
        vocab_size:           int           = VOCAB_SIZE,
        max_len:              int           = 256,
        # Stage A — geometry encoder
        geom_dim:             int           = 128,
        geom_num_layers:      int           = 4,
        geom_reduced_dim:     int           = 32,
        geom_ff_dim:          Optional[int] = None,
        geom_max_len:         Optional[int] = None,   # override when ckpt max_len differs
        # Stage A — structure bottleneck
        struct_bottleneck_dim: int          = 64,
        glob_bottleneck_dim:   int          = 128,
        curv_out:              int          = 16,
        bottleneck_mode:       str          = 'v2',   # 'v2' | 'v1'/'full' | 'simple'
        # Stage B — sequence encoder
        seq_dim:              int           = 128,
        seq_num_layers:       int           = 2,
        seq_num_heads:        int           = 8,
        seq_ff_dim:           Optional[int] = None,
        cross_attn_heads:     int           = 4,
        # Shared
        dropout:              float         = 0.1,
        pooling:              str           = 'attention',
        num_libraries:        int           = 0,
        # Stage A auxiliary loss weights
        lambda_pair:          float         = 0.1,
        lambda_ss:            float         = 0.1,
        lambda_mfe:           float         = 0.01,
        lambda_curv:          float         = 0.01,
        lambda_cons:          float         = 0.0,
    ):
        super().__init__()
        self.lambda_pair = lambda_pair
        self.lambda_ss   = lambda_ss
        self.lambda_mfe  = lambda_mfe
        self.lambda_curv = lambda_curv
        self.lambda_cons = lambda_cons
        self._bottleneck_mode = bottleneck_mode

        _geom_max_len = geom_max_len or max_len

        # ── Stage A — geometry encoder ─────────────────────────────────────────
        self.geom_encoder = RNABenderEncoder(
            vocab_size  = vocab_size,
            max_len     = _geom_max_len,
            model_dim   = geom_dim,
            num_layers  = geom_num_layers,
            reduced_dim = geom_reduced_dim,
            ff_dim      = geom_ff_dim,
            dropout     = dropout,
            pooling     = pooling,
        )
        self.pair_head = PairMapHead(geom_dim)
        self.ss_head   = nn.Linear(geom_dim, 3)
        self.mfe_head  = nn.Linear(geom_dim, 1)
        self.mfe_pool  = nn.Linear(geom_dim, 1)   # attention weights for MFE pooling

        # ── Stage A — structure bottleneck ─────────────────────────────────────
        self.struct_bottleneck = StructureBottleneck(
            geom_dim       = geom_dim,
            reduced_dim    = geom_reduced_dim,
            bottleneck_dim = struct_bottleneck_dim,
            curv_out       = curv_out,
            mode           = bottleneck_mode,
        )
        _n_stats = N_STATS_V2 if bottleneck_mode == 'v2' else 0
        self.global_bottleneck = GlobalBottleneck(
            tok_dim  = struct_bottleneck_dim,
            glob_dim = glob_bottleneck_dim,
            n_stats  = _n_stats,
        )

        # ── Stage B — sequence encoder ─────────────────────────────────────────
        self.seq_encoder = RNASequenceEncoder(
            vocab_size = vocab_size,
            max_len    = max_len,
            model_dim  = seq_dim,
            num_layers = seq_num_layers,
            num_heads  = seq_num_heads,
            ff_dim     = seq_ff_dim,
            dropout    = dropout,
            pooling    = pooling,
        )

        # Stats injection: project B_stats_tok into seq_dim and add to H_seq before
        # cross-attention.  LayerNorm stabilises the scale; a learned scalar gate
        # (initialised to 0) lets the model open the injection channel gradually.
        if _n_stats > 0:
            self.stats_proj: Optional[nn.Linear]    = nn.Linear(_n_stats, seq_dim)
            self.stats_ln:   Optional[nn.LayerNorm] = nn.LayerNorm(seq_dim)
            self.stats_gate: Optional[nn.Parameter] = nn.Parameter(torch.zeros(1))
        else:
            self.stats_proj = None
            self.stats_ln   = None
            self.stats_gate = None

        # Cross-attention bridge: Q=H_seq (enriched), K=V=B_tok
        self.cross_attn    = CrossAttentionBridge(
            seq_dim   = seq_dim,
            tok_dim   = struct_bottleneck_dim,
            num_heads = cross_attn_heads,
            dropout   = dropout,
        )
        self.seq_pool_attn = nn.Linear(seq_dim, 1)

        # ── MRL prediction head ─────────────────────────────────────────────────
        head_in = seq_dim + glob_bottleneck_dim
        self.mrl_head = nn.Sequential(
            nn.Linear(head_in, head_in // 2),
            nn.GELU(),
            nn.Linear(head_in // 2, 1),
        )

        # ── Library conditioning (MRL cross-library) ────────────────────────────
        if num_libraries > 0:
            self.lib_emb: Optional[nn.Embedding] = nn.Embedding(num_libraries, seq_dim)
        else:
            self.lib_emb = None

        self.drop = nn.Dropout(dropout)
        self._init_new_params()

    def _init_new_params(self):
        """Initialise Stage B modules and heads (encoders handle their own init)."""
        for m in [self.mrl_head, self.cross_attn, self.struct_bottleneck,
                  self.global_bottleneck, self.pair_head, self.ss_head,
                  self.mfe_head, self.mfe_pool, self.seq_pool_attn]:
            _init_weights(m)
        if self.stats_proj is not None:
            _init_weights(self.stats_proj)
            # stats_ln is LayerNorm — _init_weights skips it; default weight=1/bias=0 is correct
        if self.lib_emb is not None:
            nn.init.normal_(self.lib_emb.weight, std=0.02)
        # Set True by load_pretrained_geom() when structure heads are loaded.
        # Used by train_utr.py to choose the right freeze method.
        self._heads_loaded: bool = False

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Stage A module groups ──────────────────────────────────────────────────

    def _stage_a_modules(self) -> List[nn.Module]:
        """All Stage A modules: used for LR grouping (get_optimizer_groups)."""
        return [self.geom_encoder, self.pair_head, self.ss_head, self.mfe_head,
                self.mfe_pool, self.struct_bottleneck, self.global_bottleneck]

    def _pretrained_modules(self) -> List[nn.Module]:
        """Modules actually saved in the pretrain_bender.py checkpoint.

        Excludes struct_bottleneck and global_bottleneck — they are hybrid-specific
        and always start from random initialisation.
        """
        return [self.geom_encoder, self.pair_head, self.ss_head,
                self.mfe_head, self.mfe_pool]

    # ── Freeze helpers ─────────────────────────────────────────────────────────

    def freeze_geom_encoder(self):
        """Freeze only the geometry encoder backbone.

        Safe when structure heads are NOT pretrained — only the backbone carries
        transferred knowledge; heads remain trainable for aux supervision.
        """
        for p in self.geom_encoder.parameters():
            p.requires_grad = False

    def unfreeze_geom_encoder(self):
        """Unfreeze geometry encoder backbone."""
        for p in self.geom_encoder.parameters():
            p.requires_grad = True

    def freeze_encoder_and_heads(self):
        """Freeze geom_encoder + pair_head + ss_head + mfe_head + mfe_pool.

        This is the CORRECT freeze to use after load_pretrained_geom() loads all
        head state dicts.  struct_bottleneck and global_bottleneck remain trainable
        — they are NOT in the pretrain checkpoint and must learn from MRL data.

        Used by train_utr.py when _heads_loaded=True and freeze_geom_epochs > 0.
        """
        for mod in self._pretrained_modules():
            for p in mod.parameters():
                p.requires_grad = False

    def unfreeze_encoder_and_heads(self):
        """Unfreeze geom_encoder + structure heads (reverse of freeze_encoder_and_heads)."""
        for mod in self._pretrained_modules():
            for p in mod.parameters():
                p.requires_grad = True

    def freeze_stage_a(self):
        """Freeze ALL of Stage A, including struct_bottleneck and global_bottleneck.

        WARNING: struct_bottleneck and global_bottleneck are NOT in the pretrain
        checkpoint — they always start random.  Calling this at epoch 0 freezes
        randomly-initialised modules that can then never learn.

        Only use if you have separately pretrained the bottleneck modules (e.g.
        via a full hybrid folding pretraining run — not currently implemented).

        For the standard workflow, use freeze_encoder_and_heads() instead.
        """
        for mod in self._stage_a_modules():
            for p in mod.parameters():
                p.requires_grad = False

    def unfreeze_stage_a(self):
        """Unfreeze ALL of Stage A (reverse of freeze_stage_a)."""
        for mod in self._stage_a_modules():
            for p in mod.parameters():
                p.requires_grad = True

    def get_optimizer_groups(self, base_lr: float, geom_lr_scale: float = 0.1) -> List[Dict]:
        """
        Three-group LR split matching pretraining status of each module:

            pretrained  (geom_encoder + pair/ss/mfe heads) → base_lr × geom_lr_scale
            bottleneck  (struct_bottleneck + global_bottleneck) → base_lr
            stage_b     (seq_encoder + cross_attn + mrl_head + stats_proj + …) → base_lr

        struct_bottleneck and global_bottleneck are always randomly initialised
        (not in the pretrain checkpoint) so they train at full base_lr alongside
        Stage B — not at the reduced pretrained LR.
        """
        pretrained_ids  = {id(p) for mod in self._pretrained_modules()
                           for p in mod.parameters()}
        bottleneck_ids  = {id(p) for mod in (self.struct_bottleneck, self.global_bottleneck)
                           for p in mod.parameters()}

        pretrained_params = [p for p in self.parameters() if id(p) in pretrained_ids]
        bottleneck_params = [p for p in self.parameters() if id(p) in bottleneck_ids]
        stage_b_params    = [p for p in self.parameters()
                             if id(p) not in pretrained_ids and id(p) not in bottleneck_ids]
        return [
            {'params': stage_b_params,    'lr': base_lr,                 'name': 'stage_b'},
            {'params': bottleneck_params, 'lr': base_lr,                 'name': 'bottleneck'},
            {'params': pretrained_params, 'lr': base_lr * geom_lr_scale, 'name': 'pretrained'},
        ]

    # ── Checkpoint loading ─────────────────────────────────────────────────────

    def load_pretrained_geom(self, path: str, strict: bool = False) -> Tuple:
        """
        Load pretrained geometry encoder weights from a pretrain_bender.py checkpoint.

        Also loads pair_head, ss_head, mfe_head, mfe_pool when present.
        Updated pretrain_bender.py saves all four alongside the encoder.

        After loading:
          - If heads were loaded → use freeze_encoder_and_heads() (protects encoder + heads;
            struct_bottleneck and global_bottleneck stay trainable at full LR).
          - If only encoder loaded → use freeze_geom_encoder() (backbone only).
          - Do NOT use freeze_stage_a(): struct_bottleneck/global_bottleneck are not in
            this checkpoint and must remain trainable.
        """
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        sd   = (ckpt.get('geom_encoder_state_dict')
                or ckpt.get('encoder_state_dict')
                or ckpt)
        missing, unexpected = self.geom_encoder.load_state_dict(sd, strict=strict)
        n = self.geom_encoder.get_num_params()
        print(f'  Loaded pretrained geom encoder ({n:,} params) from {path}')
        if missing:
            print(f'    missing  : {missing[:4]}{"..." if len(missing) > 4 else ""}')
        if unexpected:
            print(f'    unexpected: {unexpected[:4]}{"..." if len(unexpected) > 4 else ""}')

        heads_loaded = []
        for attr, key in [('pair_head', 'pair_head_state_dict'),
                          ('ss_head',   'ss_head_state_dict'),
                          ('mfe_head',  'mfe_head_state_dict'),
                          ('mfe_pool',  'mfe_pool_state_dict')]:
            if key in ckpt:
                getattr(self, attr).load_state_dict(ckpt[key], strict=False)
                heads_loaded.append(attr)

        if 'pair_head' in heads_loaded and 'ss_head' in heads_loaded:
            self._heads_loaded = True
            print(f'    Also loaded structure heads: {heads_loaded}')
            print(f'    → use freeze_encoder_and_heads() to protect pretrained weights.')
        else:
            self._heads_loaded = False
            print(f'    Note: structure heads not in checkpoint — heads start fresh.')
            print(f'    → use freeze_geom_encoder() (backbone only).')

        return missing, unexpected

    # ── Loss computation ───────────────────────────────────────────────────────

    def _compute_loss(
        self,
        out:          Dict,
        labels:       torch.Tensor,
        seq_mask:     torch.Tensor,
        pair_targets: Optional[torch.Tensor] = None,
        ss_labels:    Optional[torch.Tensor] = None,
        mfe_labels:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full hybrid loss:
            L = L_MRL + λ_pair·L_pair + λ_ss·L_ss + λ_mfe·L_mfe
                      + λ_curv·L_curv + λ_cons·L_cons

        pair/SS/MFE supervision is applied only when the corresponding targets are
        provided.  For UTR datasets this means:
          - pair_targets: only available if explicitly passed (not in collate_utr by default)
          - ss_labels: available when aux_struct=True  (batch key 'ss_ids')
          - mfe_labels: available when aux_struct=True (batch key 'mfe')
        """
        loss = F.mse_loss(out['task_logits'], labels.float())

        # Pair-map BCE
        if self.lambda_pair > 0 and pair_targets is not None:
            pair_logits = out['pair_logits']
            mf          = seq_mask.float()
            pair_mask   = mf.unsqueeze(1) * mf.unsqueeze(2)
            n_valid     = pair_mask.sum().clamp(min=1)
            loss = loss + self.lambda_pair * (
                F.binary_cross_entropy_with_logits(
                    pair_logits, pair_targets.float(), reduction='none'
                ) * pair_mask
            ).sum() / n_valid

        # Per-token SS cross-entropy
        if self.lambda_ss > 0 and ss_labels is not None:
            ss_logits = out['ss_logits']
            valid     = ss_labels != SS_IGNORE_IDX
            if valid.any():
                loss = loss + self.lambda_ss * F.cross_entropy(
                    ss_logits[valid], ss_labels[valid]
                )

        # MFE regression
        if self.lambda_mfe > 0 and mfe_labels is not None:
            loss = loss + self.lambda_mfe * F.mse_loss(
                out['mfe_pred'], mfe_labels.float()
            )

        # Curvature regularisation
        if self.lambda_curv > 0:
            kappa_list = out.get('kappa_list', [])
            if kappa_list:
                mf      = seq_mask.float()
                n_valid = mf.sum().clamp(min=1)
                loss_curv = sum(
                    (k.pow(2).sum(-1) * mf).sum() / n_valid for k in kappa_list
                ) / len(kappa_list)
                loss = loss + self.lambda_curv * loss_curv

        # Backbone–pairing consistency
        if self.lambda_cons > 0:
            p_bb1_list    = out.get('p_bb1_list', [])
            p_struct_list = out.get('p_struct_list', [])
            edge_feat     = out.get('edge_feat')
            if p_bb1_list and p_struct_list and edge_feat is not None:
                loss = loss + self.lambda_cons * _consistency_loss(
                    p_bb1_list[-1], p_struct_list[-1], edge_feat
                )

        return loss

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:    torch.Tensor,
        seq_mask:     torch.Tensor,
        # Folding collate: edge_idx / edge_feat
        edge_idx:     Optional[torch.Tensor] = None,
        edge_feat:    Optional[torch.Tensor] = None,
        # UTR collate: edge_index / edge_attrs / edge_mask (aliases; edge_mask unused)
        edge_index:   Optional[torch.Tensor] = None,
        edge_attrs:   Optional[torch.Tensor] = None,
        edge_mask:    Optional[torch.Tensor] = None,
        labels:       Optional[torch.Tensor] = None,
        library_ids:  Optional[torch.Tensor] = None,
        # Auxiliary supervision targets
        pair_targets: Optional[torch.Tensor] = None,  # (B, L, L) — folding or BPP
        ss_labels:    Optional[torch.Tensor] = None,  # (B, L)     — aux_struct mode
        mfe_labels:   Optional[torch.Tensor] = None,  # (B,)       — aux_struct mode
    ) -> Dict:
        """
        Returns a dict with:
            task_logits  : (B,)          predicted MRL
            pair_logits  : (B, L, L)     Stage A pair map
            ss_logits    : (B, L, 3)     Stage A SS logits
            mfe_pred     : (B,)          Stage A MFE prediction
            B_tok        : (B, L, d_btok) per-token structure bottleneck
            B_stats_tok  : (B, L, 10) or None — interpretable scalar channels (v2)
            B_glob       : (B, d_bglob)  global structure bottleneck
            kappa_list, p_bb1_list, p_struct_list, edge_feat
            loss         : scalar (only when labels provided)
        """
        eidx  = edge_idx  if edge_idx  is not None else edge_index
        efeat = edge_feat if edge_feat is not None else edge_attrs

        # ── Stage A: geometry encoder ──────────────────────────────────────────
        H_geom, _, geom_aux = self.geom_encoder.encode(
            input_ids, eidx, efeat, seq_mask
        )
        kappa_list    = geom_aux['kappa_list']
        p_bb1_list    = geom_aux['p_bb1_list']
        p_struct_list = geom_aux['p_struct_list']
        kappa_last    = (kappa_list[-1] if kappa_list
                         else H_geom.new_zeros(H_geom.shape[0], H_geom.shape[1], 0))

        pair_logits, _ = self.pair_head(H_geom, seq_mask)       # (B, L, L)
        ss_logits      = self.ss_head(H_geom)                    # (B, L, 3)
        mfe_hat        = self.mfe_head(
            _pool(H_geom, seq_mask, self.mfe_pool)
        ).squeeze(-1)                                            # (B,)

        # ── Stage A: structure bottleneck ──────────────────────────────────────
        B_tok, B_stats_tok = self.struct_bottleneck(
            H_geom, pair_logits, ss_logits, kappa_last, seq_mask
        )                                                        # (B, L, d_btok), (B, L, 10)|None
        B_glob = self.global_bottleneck(
            B_tok, pair_logits, mfe_hat, seq_mask,
            B_stats_tok=B_stats_tok,
        )                                                        # (B, d_bglob)

        # ── Stage B: sequence encoder ──────────────────────────────────────────
        H_seq, _, _ = self.seq_encoder.encode(input_ids, seq_mask)   # (B, L, seq_dim)

        # Inject structural statistics directly into sequence representations.
        # LayerNorm stabilises the scale mismatch between encoder hidden states
        # and raw scalar structural features.  stats_gate starts at 0 so the
        # path opens gradually as the model learns to use it.
        if self.stats_proj is not None and B_stats_tok is not None:
            H_seq = H_seq + self.stats_gate * self.stats_ln(self.stats_proj(B_stats_tok))

        # Cross-attention: inject structure content (B_tok) into sequence space
        H_tilde = self.cross_attn(H_seq, B_tok, seq_mask)       # (B, L, seq_dim)

        h_pool = _pool(H_tilde, seq_mask, self.seq_pool_attn)   # (B, seq_dim)
        if self.lib_emb is not None and library_ids is not None:
            h_pool = h_pool + self.lib_emb(library_ids)

        y = self.mrl_head(
            self.drop(torch.cat([h_pool, B_glob], dim=-1))
        ).squeeze(-1)                                            # (B,)

        out: Dict = {
            'task_logits':   y,
            'pair_logits':   pair_logits,
            'ss_logits':     ss_logits,
            'mfe_pred':      mfe_hat,
            'B_tok':         B_tok,
            'B_stats_tok':   B_stats_tok,
            'B_glob':        B_glob,
            'kappa_list':    kappa_list,
            'p_bb1_list':    p_bb1_list,
            'p_struct_list': p_struct_list,
            'edge_feat':     efeat,
        }

        if labels is not None:
            out['loss'] = self._compute_loss(
                out, labels, seq_mask,
                pair_targets=pair_targets,
                ss_labels=ss_labels,
                mfe_labels=mfe_labels,
            )

        return out


__all__ = [
    'RNAHybridModel',
    'StructureBottleneck',
    'GlobalBottleneck',
    'CrossAttentionBridge',
    'N_STATS_V2',
]
