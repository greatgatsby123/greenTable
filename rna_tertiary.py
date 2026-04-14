"""
RNA Tertiary Structure Prediction  (rna_tertiary.py)

Hypothesis
    Grassmann-derived local backbone geometry  →  3D nucleotide frames  →  tertiary fold.

Representation: 3 beads / nucleotide
    P      phosphate backbone atom
    C4'    sugar carbon  (local frame origin)
    N_gly  glycosidic nitrogen  (N9 for A/G purines, N1 for C/U/T pyrimidines)

Local rigid frame F_i = (R_i, t_i):
    t_i  = C4'_i
    R_i  built from non-collinear template (C4'→N_gly, C4'→P) via Gram-Schmidt.
    All inter-nucleotide features derived from relative frames → E(3)-invariant.

Architecture
    RNATertiaryEncoder    – sequence transformer with Grassmann backbone features
    RNAFrameHead          – predicts per-nucleotide frame update (δr, δt) per refine step
    RNADistanceHead       – sequence-only distogram  p(d_ij | sequence)
    RNAOrientationHead    – sequence-only orientation prior  p(ori_ij | sequence)
    RNASeparation         – learnable pairing propensity w_θ(X, Y | sequence)
    RNAEnergy             – E = E_chem + E_dist(X|prior) + E_ori(X|prior) + E_pair
    RNARefiner            – T-step iterative coordinate / frame refinement
    RNATertiaryModel      – top-level module

Energy  (structure X scored against sequence-only priors)
    E_dist  = -log p_dist( bin(C4'_i–C4'_j distance) | sequence )
    E_ori   = -log p_ori( bin(orientation | sequence ) )
    E_chem  = -logsigmoid( separation logits )
    E_pair  = alias for E_chem

Losses
    L_FAPE   Frame-Aligned Point Error on all T refinement steps (ramped weight)
    L_disto  Cross-entropy on distogram prior vs true distance bins
    L_ori    Cross-entropy on orientation prior vs true orientation bins
    L_bond   Bond-length / consecutive-residue distance penalty (physics prior)
    L_rank   E(true) < E(noised decoy) with margin
    L_curv   Curvature smoothness
    Total    w_fape·L_FAPE + w_disto·L_disto + w_ori·L_ori + w_bond·L_bond
           + w_rank·L_rank + w_curv·L_curv

Reference
    Grassmann Flow — arXiv:2512.19428
    AlphaFold2 FAPE — Nature 596 (2021)
    DRfold  —  Nat. Commun. 14, 5330 (2023)
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Vocabulary ────────────────────────────────────────────────────────────────

NUC_VOCAB: Dict[str, int] = {
    'A': 0, 'C': 1, 'G': 2, 'U': 3,
    'T': 3, 'N': 4, '<PAD>': 5,
}
VOCAB_SIZE = 6
PAD_ID     = NUC_VOCAB['<PAD>']
BACKBONE_OFFSETS = (1, 2, 4)

# Distogram settings
DIST_MIN    = 2.0
DIST_MAX    = 40.0
N_DIST_BINS = 40

# Orientation bins
N_ORI_BINS = 24

# Ideal C4'–C4' bond distance between consecutive residues (~5.9 Å in A-form)
C4P_BOND_IDEAL = 5.9
C4P_BOND_TOL   = 1.5   # tolerance before penalty kicks in


# ─── Positional encoding ───────────────────────────────────────────────────────

def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe  = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
    return pe


# ─── Plücker wedge product ─────────────────────────────────────────────────────

def plucker_coords(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    r     = u.shape[-1]
    outer = u.unsqueeze(-1) * v.unsqueeze(-2)
    anti  = outer - outer.transpose(-1, -2)
    idx   = torch.triu_indices(r, r, offset=1, device=u.device)
    p     = anti[..., idx[0], idx[1]]
    return p / p.norm(dim=-1, keepdim=True).clamp_min(1e-8)


# ─── Local rigid frame construction ───────────────────────────────────────────

def build_frames(
    coords: torch.Tensor,          # (B, L, 3, 3)  [P, C4', N_gly]
    mask:   Optional[torch.Tensor] = None,  # (B, L) bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build local orthonormal frames from 3-bead coords.

    Gram-Schmidt basis:
        e1 = normalize(C4' → N_gly)            (toward base)
        e2 = normalize(C4' → P ⊥ e1)           (in-plane backbone)
        e3 = e1 × e2                            (out-of-plane)

    The three atoms are NON-COLLINEAR by construction (P is off the C4'→N axis
    in any real nucleotide), so the frame is always well-defined.

    Padded positions (mask=False) receive the identity frame.

    Returns:
        R: (B, L, 3, 3)   columns = [e1, e2, e3]
        t: (B, L, 3)      C4' position
    """
    P   = coords[..., 0, :]   # (B, L, 3)
    C4p = coords[..., 1, :]
    N   = coords[..., 2, :]

    # Primary axis e1: C4' → N_gly
    e1_raw = N - C4p
    e1 = F.normalize(e1_raw, dim=-1)

    # e2: component of (C4'→P) orthogonal to e1 — guaranteed nonzero in real RNA
    u  = P - C4p
    e2 = u - (u * e1).sum(-1, keepdim=True) * e1
    e2 = F.normalize(e2, dim=-1)

    # e3: right-hand cross product
    e3 = torch.cross(e1, e2, dim=-1)

    R = torch.stack([e1, e2, e3], dim=-1)   # (B, L, 3, 3)

    # Set padded positions to identity frame
    if mask is not None:
        pad = ~mask  # (B, L)
        I   = torch.eye(3, device=R.device, dtype=R.dtype)
        R   = R.masked_fill(pad.unsqueeze(-1).unsqueeze(-1), 0.0)
        R   = R + (pad.unsqueeze(-1).unsqueeze(-1).float() * I)

    return R, C4p


def _skew(v: torch.Tensor) -> torch.Tensor:
    """Anti-symmetric 3×3 skew matrix for each vector (...,3) → (...,3,3)."""
    z   = torch.zeros_like(v[..., 0])
    x, y, zz = v[..., 0], v[..., 1], v[..., 2]
    row0 = torch.stack([ z,  -zz,  y], dim=-1)
    row1 = torch.stack([ zz,   z, -x], dim=-1)
    row2 = torch.stack([-y,    x,  z], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def apply_frame_update(
    R:       torch.Tensor,   # (B, L, 3, 3)
    t:       torch.Tensor,   # (B, L, 3)
    delta_r: torch.Tensor,   # (B, L, 3)  axis-angle
    delta_t: torch.Tensor,   # (B, L, 3)  local translation
    mask:    Optional[torch.Tensor] = None,   # (B, L)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply small axis-angle rotation + local-frame translation."""
    # Zero updates for padding
    if mask is not None:
        mf      = mask.float().unsqueeze(-1)
        delta_r = delta_r * mf
        delta_t = delta_t * mf

    angle   = delta_r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    axis    = delta_r / angle                                    # (B,L,3)
    cos_a   = torch.cos(angle).unsqueeze(-1)                    # (B,L,1,1)
    sin_a   = torch.sin(angle).unsqueeze(-1)
    K       = _skew(axis)                                       # (B,L,3,3)
    I       = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(K)
    R_delta = (cos_a * I
               + sin_a * K
               + (1 - cos_a) * axis.unsqueeze(-1) * axis.unsqueeze(-2))
    R_new   = R_delta @ R

    # Translate in local frame
    t_new   = t + (R @ delta_t.unsqueeze(-1)).squeeze(-1)
    return R_new, t_new


def frames_to_invariants(
    R: torch.Tensor,   # (B, L, 3, 3)
    t: torch.Tensor,   # (B, L, 3)
) -> torch.Tensor:
    """
    Pairwise E(3)-invariant features for all pairs (i,j).  Shape (B,L,L,7):
        [ dist, Δt_local_x, Δt_local_y, Δt_local_z, trace(R_rel), cos_angle, log_dist ]
    """
    B, L = t.shape[:2]
    ti   = t.unsqueeze(2).expand(B, L, L, 3)
    tj   = t.unsqueeze(1).expand(B, L, L, 3)
    dt   = tj - ti

    Ri   = R.unsqueeze(2).expand(B, L, L, 3, 3)
    dt_l = (Ri.transpose(-1, -2) @ dt.unsqueeze(-1)).squeeze(-1)   # local coords

    dist    = dt.norm(dim=-1, keepdim=True).clamp_min(1e-4)
    log_d   = torch.log(dist)

    Rj      = R.unsqueeze(1).expand(B, L, L, 3, 3)
    R_rel   = Ri.transpose(-1, -2) @ Rj
    trace   = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
    cos_a   = ((trace - 1) / 2).clamp(-1, 1)

    return torch.cat([dist, dt_l, trace, cos_a, log_d], dim=-1)   # (B,L,L,7)


# ─── Distance/orientation bin utilities ───────────────────────────────────────

def coords_to_dist_bins(
    coords: torch.Tensor,
    n_bins: int   = N_DIST_BINS,
    d_min:  float = DIST_MIN,
    d_max:  float = DIST_MAX,
) -> torch.Tensor:
    """(B,L,3,3) → (B,L,L) integer distance bins on C4'–C4'."""
    C4p  = coords[:, :, 1, :]
    d    = torch.cdist(C4p.float(), C4p.float())
    bins = ((d - d_min) / (d_max - d_min) * n_bins).long()
    return bins.clamp(0, n_bins - 1)


def frames_to_ori_bins(
    R:      torch.Tensor,   # (B, L, 3, 3)
    n_bins: int = N_ORI_BINS,
) -> torch.Tensor:
    """(B,L,3,3) → (B,L,L) integer orientation bins via rotation angle."""
    B, L  = R.shape[:2]
    Ri    = R.unsqueeze(2).expand(B, L, L, 3, 3)
    Rj    = R.unsqueeze(1).expand(B, L, L, 3, 3)
    trace = (Ri.transpose(-1, -2) @ Rj).diagonal(dim1=-2, dim2=-1).sum(-1)
    cos_a = ((trace - 1) / 2).clamp(-1, 1)
    angle = torch.acos(cos_a)          # [0, π]
    bins  = (angle / math.pi * n_bins).long().clamp(0, n_bins - 1)
    return bins


# ─── Encoder ──────────────────────────────────────────────────────────────────

class _BackboneMixer(nn.Module):
    """Backbone Plücker + discrete curvature features (sequence-geometry encoder)."""

    def __init__(self, model_dim: int, reduced_dim: int,
                 offsets: Tuple[int, ...] = BACKBONE_OFFSETS):
        super().__init__()
        self.offsets = offsets
        plu_dim      = reduced_dim * (reduced_dim - 1) // 2
        hidden       = model_dim // 2
        self.W_red   = nn.Linear(model_dim, reduced_dim)
        self.phi_bb  = nn.Sequential(
            nn.Linear(len(offsets) * plu_dim, hidden), nn.GELU(),
            nn.Linear(hidden, model_dim),
        )
        self.phi_curv = nn.Sequential(
            nn.Linear(plu_dim, hidden), nn.GELU(),
            nn.Linear(hidden, model_dim),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, _ = h.shape
        z  = self.W_red(h)
        mf = mask.float().unsqueeze(-1)

        p_list = []
        for delta in self.offsets:
            zj = torch.zeros_like(z)
            zj[:, :L - delta] = z[:, delta:]
            mj = torch.zeros_like(mf)
            mj[:, :L - delta] = mask[:, delta:].float().unsqueeze(-1)
            p_list.append(plucker_coords(z, zj) * (mf * mj))

        p_bb1 = p_list[0]
        p_fwd = torch.zeros_like(p_bb1); p_bwd = torch.zeros_like(p_bb1)
        p_fwd[:, :L - 1] = p_bb1[:, 1:]
        p_bwd[:, 1:]     = p_bb1[:, :L - 1]
        kappa = (p_fwd - 2 * p_bb1 + p_bwd) * mf

        g_bb   = self.phi_bb(torch.cat(p_list, dim=-1))
        g_curv = self.phi_curv(kappa)
        return g_bb, g_curv, kappa


class _TertiaryEncoderLayer(nn.Module):
    def __init__(self, model_dim: int, reduced_dim: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        d = model_dim
        self.ln1    = nn.LayerNorm(d)
        self.ln2    = nn.LayerNorm(d)
        self.bb     = _BackboneMixer(d, reduced_dim)
        self.agg    = nn.Sequential(nn.Linear(d * 2, d), nn.GELU())
        self.gate_w = nn.Linear(d * 2, d)
        self.gate_a = nn.Linear(d * 2, d)
        self.ff     = nn.Sequential(
            nn.Linear(d, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, d), nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = h
        h  = self.ln1(h)
        g_bb, g_curv, kappa = self.bb(h, mask)
        g   = self.agg(torch.cat([g_bb, g_curv], dim=-1))
        cat = torch.cat([h, g], dim=-1)
        u   = self.gate_w(cat)
        a   = torch.sigmoid(self.gate_a(cat))
        h   = h0 + self.drop(a * u)
        h   = h + self.drop(self.ff(self.ln2(h)))
        return h, kappa


class RNATertiaryEncoder(nn.Module):
    """Sequence encoder with Grassmann backbone features."""

    def __init__(self, model_dim=128, num_layers=6, reduced_dim=16,
                 ff_dim=512, dropout=0.1, max_len=4096):
        super().__init__()
        self.embed  = nn.Embedding(VOCAB_SIZE, model_dim, padding_idx=PAD_ID)
        self.register_buffer('pe', _sinusoidal_pe(max_len, model_dim))
        self.layers = nn.ModuleList([
            _TertiaryEncoderLayer(model_dim, reduced_dim, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.ln_out = nn.LayerNorm(model_dim)

    def forward(self, tokens: torch.Tensor
                ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        mask = (tokens != PAD_ID)
        B, L = tokens.shape
        h    = self.embed(tokens) + self.pe[:L].unsqueeze(0)
        kappas = []
        for layer in self.layers:
            h, kappa = layer(h, mask)
            kappas.append(kappa)
        return self.ln_out(h), kappas, mask


# ─── Frame update head ─────────────────────────────────────────────────────────

class RNAFrameHead(nn.Module):
    """Per-nucleotide axis-angle δr + local translation δt (zero-initialized)."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 6),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.mlp(h)
        return out[..., :3], out[..., 3:]


# ─── Sequence-only geometry prior heads ───────────────────────────────────────
#
# These heads predict p(d_ij | sequence) and p(ori_ij | sequence) from the
# ENCODER HIDDEN STATE ONLY — no current 3D coordinates.  The energy function
# then evaluates -log p at the CURRENT geometry of the structure.
# This prevents the trivial "look at the answer" shortcut.

class RNADistanceHead(nn.Module):
    """
    Sequence-only distogram: p(d_ij | tokens).
    Input: h_i ‖ h_j ‖ relative-position-encoding  →  n_bins logits.
    """

    def __init__(self, model_dim: int, n_bins: int = N_DIST_BINS, hidden: int = 64):
        super().__init__()
        self.n_bins = n_bins
        # Relative position encoding: 2 features  [sin(π·rel), cos(π·rel)]
        pair_in = model_dim * 2 + 2
        self.mlp = nn.Sequential(
            nn.LayerNorm(pair_in),
            nn.Linear(pair_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_bins),
        )

    def forward(
        self,
        h:    torch.Tensor,   # (B, L, d)
        mask: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:        # (B, L, L, n_bins)
        B, L, d = h.shape
        hi = h.unsqueeze(2).expand(B, L, L, d)
        hj = h.unsqueeze(1).expand(B, L, L, d)

        # Relative position in [0,1]
        pos  = torch.arange(L, device=h.device, dtype=h.dtype)
        rel  = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs() / max(L - 1, 1)  # (L,L)
        rpe  = torch.stack([torch.sin(math.pi * rel),
                            torch.cos(math.pi * rel)], dim=-1)               # (L,L,2)
        rpe  = rpe.unsqueeze(0).expand(B, -1, -1, -1)

        logits = self.mlp(torch.cat([hi, hj, rpe], dim=-1))   # (B,L,L,n_bins)
        logits = (logits + logits.transpose(1, 2)) * 0.5       # symmetrize
        pm     = (mask.unsqueeze(1) & mask.unsqueeze(2))
        logits = logits.masked_fill(~pm.unsqueeze(-1), 0.0)
        return logits


class RNAOrientationHead(nn.Module):
    """Sequence-only orientation prior: p(ori_ij | tokens)."""

    def __init__(self, model_dim: int, n_bins: int = N_ORI_BINS, hidden: int = 64):
        super().__init__()
        self.n_bins = n_bins
        pair_in = model_dim * 2 + 2
        self.mlp = nn.Sequential(
            nn.LayerNorm(pair_in),
            nn.Linear(pair_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_bins),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L, d = h.shape
        hi  = h.unsqueeze(2).expand(B, L, L, d)
        hj  = h.unsqueeze(1).expand(B, L, L, d)
        pos = torch.arange(L, device=h.device, dtype=h.dtype)
        rel = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs() / max(L - 1, 1)
        rpe = torch.stack([torch.sin(math.pi * rel),
                           torch.cos(math.pi * rel)], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        logits = self.mlp(torch.cat([hi, hj, rpe], dim=-1))
        pm     = (mask.unsqueeze(1) & mask.unsqueeze(2))
        logits = logits.masked_fill(~pm.unsqueeze(-1), 0.0)
        return logits


# ─── Separation / pairing propensity ──────────────────────────────────────────

class RNASeparation(nn.Module):
    """Learnable contact propensity w_θ(i, j | sequence)."""

    def __init__(self, model_dim: int, hidden: int = 64):
        super().__init__()
        self.query = nn.Linear(model_dim, hidden)
        self.key   = nn.Linear(model_dim, hidden)
        self.bias  = nn.Linear(model_dim, 1)
        self.scale = math.sqrt(hidden)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        q = self.query(h)
        k = self.key(h)
        w = (q @ k.transpose(-1, -2)) / self.scale
        b = self.bias(h).squeeze(-1)
        w = w + b.unsqueeze(2) + b.unsqueeze(1)
        pm = (mask.unsqueeze(1) & mask.unsqueeze(2))
        return w.masked_fill(~pm, -3e4)


# ─── Energy function ───────────────────────────────────────────────────────────

class RNAEnergy(nn.Module):
    """
    Full 3D folding energy — sequence priors evaluated at current geometry X.

    Heads predict priors from sequence only.
    Energy evaluates how well structure X fits those priors.

    E = w[0]·E_chem + w[1]·E_dist + w[2]·E_ori + w[3]·E_pair
    """

    def __init__(self, model_dim: int, n_dist_bins=N_DIST_BINS,
                 n_ori_bins=N_ORI_BINS, hidden=64):
        super().__init__()
        self.dist_head = RNADistanceHead(model_dim, n_dist_bins, hidden)
        self.ori_head  = RNAOrientationHead(model_dim, n_ori_bins, hidden)
        self.sep       = RNASeparation(model_dim, hidden)
        self.log_w     = nn.Parameter(torch.zeros(4))   # softmax term weights

    def compute_priors(self, h: torch.Tensor, mask: torch.Tensor) -> Tuple:
        """Compute sequence-only priors (detached from geometry)."""
        dist_logits = self.dist_head(h, mask)   # (B,L,L,nd)
        ori_logits  = self.ori_head(h, mask)    # (B,L,L,no)
        sep_logits  = self.sep(h, mask)         # (B,L,L)
        return dist_logits, ori_logits, sep_logits

    def score_structure(
        self,
        h:           torch.Tensor,   # (B, L, d)
        coords:      torch.Tensor,   # (B, L, 3, 3)
        mask:        torch.Tensor,   # (B, L)
        dist_logits: Optional[torch.Tensor] = None,
        ori_logits:  Optional[torch.Tensor] = None,
        sep_logits:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Score a structure X by evaluating priors at the actual geometry.
        Returns (B,) total energy.
        """
        if dist_logits is None or ori_logits is None or sep_logits is None:
            dist_logits, ori_logits, sep_logits = self.compute_priors(h, mask)

        w = torch.softmax(self.log_w, dim=0)

        # Distance energy: -log p(true bin | seq)
        R, t        = build_frames(coords, mask)
        true_dbins  = coords_to_dist_bins(coords, dist_logits.shape[-1])   # (B,L,L)
        dist_lp     = F.log_softmax(dist_logits, dim=-1)                   # (B,L,L,nd)
        E_dist      = -dist_lp.gather(-1, true_dbins.unsqueeze(-1)).squeeze(-1)  # (B,L,L)

        # Orientation energy
        true_obins  = frames_to_ori_bins(R, ori_logits.shape[-1])          # (B,L,L)
        ori_lp      = F.log_softmax(ori_logits, dim=-1)
        E_ori       = -ori_lp.gather(-1, true_obins.unsqueeze(-1)).squeeze(-1)   # (B,L,L)

        # Chemical / contact energy
        E_chem = -F.logsigmoid(sep_logits)   # (B,L,L)
        E_pair = E_chem

        pm   = (mask.unsqueeze(1) & mask.unsqueeze(2)).float()
        # Exclude self-pairs and very short range (i==j or |i-j|<=2)
        L    = mask.shape[1]
        rng  = torch.ones(L, L, device=mask.device).triu(3)   # |i-j| >= 3
        pm   = pm * rng

        def _sum(E):
            return (E * pm).sum(dim=(-1, -2))   # (B,)

        return (w[0] * _sum(E_chem)
              + w[1] * _sum(E_dist)
              + w[2] * _sum(E_ori)
              + w[3] * _sum(E_pair))

    def forward(self, h, coords, mask, dist_logits=None, ori_logits=None,
                sep_logits=None) -> Dict[str, torch.Tensor]:
        if dist_logits is None:
            dist_logits, ori_logits, sep_logits = self.compute_priors(h, mask)
        total = self.score_structure(h, coords, mask, dist_logits, ori_logits, sep_logits)
        return {
            'total':       total,
            'dist_logits': dist_logits,
            'ori_logits':  ori_logits,
            'sep_logits':  sep_logits,
        }


# ─── Local atom template (non-collinear 3-bead offsets) ───────────────────────

# Approximate ideal geometry in local frame (Å).
# P is off-axis, ensuring e2 is never degenerate.
# Values are typical A-form RNA local geometry.
_LOCAL_TEMPLATE = torch.tensor([
    [-0.9,  1.2,  0.0],    # P   (behind and above C4' in local frame)
    [ 0.0,  0.0,  0.0],    # C4' (origin)
    [ 2.5,  0.0,  0.0],    # N_gly (along e1 = toward base)
], dtype=torch.float32)    # (3, 3)  — rows = atoms, cols = xyz


def _reconstruct_atoms(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct 3 atom positions from frame using the local template.

    coords[b,i,atom,xyz] = t[b,i] + R[b,i] @ template[atom]

    Args:
        R: (B, L, 3, 3)
        t: (B, L, 3)
    Returns:
        coords: (B, L, 3, 3)  [P, C4', N_gly]
    """
    tmpl = _LOCAL_TEMPLATE.to(R.device, R.dtype)   # (3,3)
    # t: (B,L,3) → (B,L,1,3); R: (B,L,3,3); tmpl: (3,3)=(atoms, xyz)
    # R @ tmpl[a]  for each atom a:  R @ (3,)  → (B,L,3)
    # broadcast: (B,L,3,3) @ (3,1) → (B,L,3,1) for each atom
    atoms = []
    for a in range(3):
        d = (R @ tmpl[a].unsqueeze(-1)).squeeze(-1)  # (B,L,3)
        atoms.append(t + d)
    return torch.stack(atoms, dim=2)   # (B, L, 3, 3)


# ─── Iterative refiner ─────────────────────────────────────────────────────────

class RNARefiner(nn.Module):
    """
    T-step iterative refinement.

    Each step:
      1. Build frames from current coords.
      2. Compute pairwise E(3)-invariant features.
      3. Mean-pool pairwise info per nucleotide.
      4. Gate into sequence hidden state.
      5. Predict frame update (δr, δt); apply.
      6. Reconstruct 3-bead coords from updated frame via non-collinear template.
    """

    def __init__(self, model_dim: int, n_steps: int = 4, hidden: int = 64):
        super().__init__()
        self.n_steps = n_steps
        inv_dim = 7   # from frames_to_invariants
        self.pair_proj = nn.ModuleList([
            nn.Sequential(nn.Linear(inv_dim, hidden), nn.GELU(),
                          nn.Linear(hidden, model_dim))
            for _ in range(n_steps)
        ])
        self.gate_w = nn.ModuleList([
            nn.Linear(model_dim * 2, model_dim) for _ in range(n_steps)
        ])
        self.gate_a = nn.ModuleList([
            nn.Linear(model_dim * 2, model_dim) for _ in range(n_steps)
        ])
        self.ln     = nn.ModuleList([nn.LayerNorm(model_dim) for _ in range(n_steps)])
        self.frame_head = RNAFrameHead(model_dim)

    def forward(
        self,
        h:      torch.Tensor,   # (B, L, d)
        coords: torch.Tensor,   # (B, L, 3, 3)
        mask:   torch.Tensor,   # (B, L)
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        R, t       = build_frames(coords, mask)
        coords_list = []

        for step in range(self.n_steps):
            inv  = frames_to_invariants(R, t)      # (B,L,L,7)
            pm   = (mask.unsqueeze(1) & mask.unsqueeze(2)).float()   # (B,L,L)
            msg  = self.pair_proj[step](inv)        # (B,L,L,d)
            n_v  = pm.sum(dim=2, keepdim=True).clamp_min(1)
            agg  = (msg * pm.unsqueeze(-1)).sum(dim=2) / n_v          # (B,L,d)

            h_s  = self.ln[step](h)
            cat  = torch.cat([h_s, agg], dim=-1)
            g    = self.gate_w[step](cat)
            a    = torch.sigmoid(self.gate_a[step](cat))
            h_aug = h + a * g

            delta_r, delta_t = self.frame_head(h_aug)
            R, t = apply_frame_update(R, t, delta_r, delta_t, mask)

            # Reconstruct non-collinear 3-bead coords from updated frame
            coords_new  = _reconstruct_atoms(R, t)
            coords_list.append(coords_new)

        return coords_list[-1], coords_list


# ─── Loss functions ────────────────────────────────────────────────────────────

def fape_loss(
    R_pred:      torch.Tensor,   # (B, L, 3, 3)
    t_pred:      torch.Tensor,   # (B, L, 3)
    R_true:      torch.Tensor,
    t_true:      torch.Tensor,
    coords_pred: torch.Tensor,   # (B, L, 3, 3)
    coords_true: torch.Tensor,
    mask:        torch.Tensor,   # (B, L)
    clamp:       float = 10.0,
) -> torch.Tensor:
    """Frame-Aligned Point Error (AF2 eq. 28, adapted for 3-bead RNA)."""
    B, L, A, _ = coords_pred.shape

    def _to_local(R, t, xyz):
        t_i  = t.unsqueeze(2).unsqueeze(3)                  # (B,Li,1,1,3)
        R_i  = R.unsqueeze(2).unsqueeze(3)                  # (B,Li,1,1,3,3)
        dx   = xyz.unsqueeze(1) - t_i                       # (B,Li,Lp,A,3)
        return (R_i.transpose(-1, -2) @ dx.unsqueeze(-1)).squeeze(-1)

    lp = _to_local(R_pred, t_pred, coords_pred)   # (B,L,L,A,3)
    lt = _to_local(R_true, t_true, coords_true)

    err   = (lp - lt).norm(dim=-1).clamp_max(clamp)         # (B,L,L,A)
    pm    = (mask.unsqueeze(1) & mask.unsqueeze(2)).float()
    count = pm.sum(dim=(-1, -2)).clamp_min(1) * A
    loss  = (err * pm.unsqueeze(-1)).sum(dim=(1, 2, 3)) / count
    return loss.mean()


def distogram_loss(
    dist_logits: torch.Tensor,   # (B, L, L, n_bins)
    coords_true: torch.Tensor,   # (B, L, 3, 3)
    mask:        torch.Tensor,
) -> torch.Tensor:
    B, L   = mask.shape
    n_bins = dist_logits.shape[-1]
    bins   = coords_to_dist_bins(coords_true, n_bins)       # (B,L,L)
    pm     = (mask.unsqueeze(1) & mask.unsqueeze(2))
    # Exclude diagonal (trivial self-distance)
    eye    = torch.eye(L, dtype=torch.bool, device=mask.device).unsqueeze(0)
    pm     = pm & ~eye
    ce = F.cross_entropy(
        dist_logits[pm],         # (N_valid, n_bins)
        bins[pm],                # (N_valid,)
    )
    return ce


def orientation_loss(
    ori_logits: torch.Tensor,   # (B, L, L, n_bins)
    R_true:     torch.Tensor,   # (B, L, 3, 3)
    mask:       torch.Tensor,
) -> torch.Tensor:
    B, L = mask.shape
    n_bins = ori_logits.shape[-1]
    bins = frames_to_ori_bins(R_true, n_bins)               # (B,L,L)
    pm   = (mask.unsqueeze(1) & mask.unsqueeze(2))
    eye  = torch.eye(L, dtype=torch.bool, device=mask.device).unsqueeze(0)
    pm   = pm & ~eye
    return F.cross_entropy(ori_logits[pm], bins[pm])


def bond_length_loss(
    coords: torch.Tensor,   # (B, L, 3, 3)
    mask:   torch.Tensor,   # (B, L)
    ideal:  float = C4P_BOND_IDEAL,
    tol:    float = C4P_BOND_TOL,
) -> torch.Tensor:
    """
    Penalise consecutive C4'–C4' distances that deviate more than `tol` Å from
    the ideal bond length.  Only penalises where both positions are valid.
    """
    C4p  = coords[:, :, 1, :]             # (B, L, 3)
    diff = (C4p[:, 1:] - C4p[:, :-1]).norm(dim=-1)   # (B, L-1)
    dev  = F.relu((diff - ideal).abs() - tol)          # hinge on deviation
    m    = (mask[:, 1:] & mask[:, :-1]).float()
    n    = m.sum().clamp_min(1)
    return (dev * m).sum() / n


def ranking_loss(
    energy_true:  torch.Tensor,   # (B,)
    energy_decoy: torch.Tensor,   # (B,)
    margin:       float = 1.0,
) -> torch.Tensor:
    return F.relu(energy_true - energy_decoy + margin).mean()


# ─── Top-level model ───────────────────────────────────────────────────────────

class RNATertiaryModel(nn.Module):
    """
    Full RNA 3D tertiary structure prediction model.

    Forward:
        tokens       : (B, L)            nucleotide token IDs
        coords_true  : (B, L, 3, 3) opt  ground-truth 3-bead coords
        coords_init  : (B, L, 3, 3) opt  initial coords; helix if None
        **kwargs     : silently absorbed (train_utr.py compatibility)

    Returns dict:
        'coords'       : (B, L, 3, 3)   final refined coords
        'coords_list'  : list of T intermediate coords
        'dist_logits'  : (B, L, L, n_dist_bins)
        'ori_logits'   : (B, L, L, n_ori_bins)
        'energy'       : (B,) total energy of refined structure
        'loss'         : scalar (only when coords_true is given)
    """

    def __init__(
        self,
        model_dim:    int   = 128,
        num_layers:   int   = 6,
        reduced_dim:  int   = 16,
        ff_dim:       int   = 512,
        dropout:      float = 0.1,
        n_refine:     int   = 4,
        n_dist_bins:  int   = N_DIST_BINS,
        n_ori_bins:   int   = N_ORI_BINS,
        energy_hidden: int  = 64,
        max_len:      int   = 4096,
        # loss weights
        w_fape:  float = 1.0,
        w_disto: float = 0.5,
        w_ori:   float = 0.3,
        w_bond:  float = 0.1,
        w_rank:  float = 0.2,
        w_curv:  float = 0.01,
        fape_clamp: float = 10.0,
    ):
        super().__init__()
        self.w_fape    = w_fape
        self.w_disto   = w_disto
        self.w_ori     = w_ori
        self.w_bond    = w_bond
        self.w_rank    = w_rank
        self.w_curv    = w_curv
        self.fape_clamp = fape_clamp
        self.n_dist_bins = n_dist_bins
        self.n_ori_bins  = n_ori_bins

        self.encoder = RNATertiaryEncoder(
            model_dim=model_dim, num_layers=num_layers, reduced_dim=reduced_dim,
            ff_dim=ff_dim, dropout=dropout, max_len=max_len,
        )
        self.refiner = RNARefiner(model_dim, n_steps=n_refine, hidden=energy_hidden)
        self.energy  = RNAEnergy(model_dim, n_dist_bins, n_ori_bins, energy_hidden)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _init_coords(self, B: int, L: int, device: torch.device,
                     dtype: torch.dtype) -> torch.Tensor:
        """Helical initialisation: A-form-like rise and twist."""
        t   = torch.arange(L, device=device, dtype=dtype)
        # A-form: rise ≈ 2.8 Å, radius ≈ 9 Å, twist ≈ 32.7°
        rise  = 2.8
        twist = math.radians(32.7)
        radius = 9.0
        x = radius * torch.cos(t * twist)
        y = radius * torch.sin(t * twist)
        z = t * rise
        C4p = torch.stack([x, y, z], dim=-1)   # (L, 3)
        # Use local template offsets to construct P and N
        tmpl = _LOCAL_TEMPLATE.to(device, dtype)   # (3,3): P, C4', N
        # Simple approximation: apply identity frame (no rotation)
        P  = C4p + tmpl[0]
        N  = C4p + tmpl[2]
        coords = torch.stack([P, C4p, N], dim=1)   # (L, 3, 3)
        return coords.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

    def forward(
        self,
        tokens:      torch.Tensor,
        coords_true: Optional[torch.Tensor] = None,
        coords_init: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        B, L   = tokens.shape
        device = tokens.device
        dtype  = self.encoder.embed.weight.dtype

        # Encode sequence
        h, kappas, mask = self.encoder(tokens)

        # Compute sequence-only geometry priors (used in both energy and loss)
        dist_logits, ori_logits, sep_logits = self.energy.compute_priors(h, mask)

        # Initial coordinates
        if coords_init is None:
            coords_init = self._init_coords(B, L, device, dtype)
        else:
            coords_init = coords_init.to(dtype)

        # Iterative refinement
        coords_final, coords_list = self.refiner(h, coords_init, mask)

        # Score final structure against sequence-only priors
        energy_val = self.energy.score_structure(
            h, coords_final, mask, dist_logits, ori_logits, sep_logits
        )

        out: Dict[str, torch.Tensor] = {
            'coords':      coords_final,
            'coords_list': coords_list,
            'dist_logits': dist_logits,
            'ori_logits':  ori_logits,
            'energy':      energy_val,
        }

        if coords_true is not None:
            coords_true = coords_true.to(dtype)
            R_true, t_true = build_frames(coords_true, mask)
            loss = coords_true.new_zeros(())

            # FAPE on every refinement step (ramped weight)
            for step_i, c_step in enumerate(coords_list):
                w_step = (step_i + 1) / len(coords_list)
                R_s, t_s = build_frames(c_step, mask)
                loss = loss + self.w_fape * w_step * fape_loss(
                    R_s, t_s, R_true, t_true, c_step, coords_true, mask, self.fape_clamp
                )

            # Distogram and orientation priors vs true geometry
            loss = loss + self.w_disto * distogram_loss(dist_logits, coords_true, mask)
            loss = loss + self.w_ori   * orientation_loss(ori_logits, R_true, mask)

            # Bond-length penalty on final coords
            if self.w_bond > 0:
                loss = loss + self.w_bond * bond_length_loss(coords_final, mask)

            # Curvature smoothness
            if self.w_curv > 0:
                kappa_loss = sum(k.pow(2).mean() for k in kappas) / len(kappas)
                loss = loss + self.w_curv * kappa_loss

            # Ranking: E(true) < E(noised decoy)
            if self.w_rank > 0:
                noise  = torch.randn_like(coords_true) * 2.0
                c_dec  = coords_true + noise
                E_true = self.energy.score_structure(h, coords_true, mask,
                             dist_logits, ori_logits, sep_logits)
                E_dec  = self.energy.score_structure(h, c_dec,       mask,
                             dist_logits, ori_logits, sep_logits)
                loss = loss + self.w_rank * ranking_loss(E_true, E_dec)

            out['loss'] = loss

        return out


# ─── Evaluation helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def compute_rmsd(
    coords_pred: torch.Tensor,   # (L, 3, 3) or (B, L, 3, 3)
    coords_true: torch.Tensor,
    mask:        Optional[torch.Tensor] = None,
    atom:        int = 1,         # 0=P, 1=C4', 2=N_gly
) -> torch.Tensor:
    """
    Per-sequence C4' (or specified atom) RMSD after Kabsch superimposition.
    Returns (B,) or scalar.
    """
    batched = coords_pred.dim() == 4
    if not batched:
        coords_pred = coords_pred.unsqueeze(0)
        coords_true = coords_true.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)

    B, L = coords_pred.shape[:2]
    rmsds = []
    for b in range(B):
        if mask is not None:
            m = mask[b]
        else:
            m = torch.ones(L, dtype=torch.bool, device=coords_pred.device)
        P = coords_pred[b, m, atom]   # (n, 3)
        Q = coords_true[b, m, atom]
        rmsd = _kabsch_rmsd(P, Q)
        rmsds.append(rmsd)

    result = torch.stack(rmsds)
    return result if batched else result[0]


def _kabsch_rmsd(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Kabsch-aligned RMSD between point sets P and Q, shape (N,3)."""
    P  = P - P.mean(0)
    Q  = Q - Q.mean(0)
    H  = P.T @ Q
    U, S, Vt = torch.linalg.svd(H)
    d  = torch.linalg.det(Vt.T @ U.T)
    D  = torch.diag(torch.stack([torch.ones_like(d), torch.ones_like(d), d]))
    R  = Vt.T @ D @ U.T
    P_rot = P @ R.T
    return ((P_rot - Q).pow(2).sum(-1).mean()).sqrt()


@torch.no_grad()
def evaluate_tertiary(
    model,
    loader,
    device: torch.device,
) -> Dict[str, float]:
    """Compute C4' RMSD, 3-bead RMSD, and mean energy."""
    model.eval()
    c4p_rmsds, bead_rmsds, energies = [], [], []

    for batch in loader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        if not batch or 'tokens' not in batch:
            continue

        coords_true = batch.get('coords_true')
        mask        = batch.get('seq_mask', (batch['tokens'] != 5))

        out = model(batch['tokens'])
        c   = out['coords']

        if coords_true is not None:
            r4  = compute_rmsd(c, coords_true, mask, atom=1)
            rb  = compute_rmsd(
                c.view(c.shape[0], c.shape[1], -1),
                coords_true.view(*coords_true.shape[:2], -1)
                    .unsqueeze(-1).expand_as(c.view(c.shape[0], c.shape[1], -1))
                    .view(c.shape[0], c.shape[1], 3, 3),
                mask, atom=1,
            )
            # simpler 3-bead: average RMSD over the 3 atoms
            all_r = [compute_rmsd(c, coords_true, mask, atom=a) for a in range(3)]
            rb    = torch.stack(all_r).mean(0)
            c4p_rmsds.append(r4.cpu())
            bead_rmsds.append(rb.cpu())

        energies.append(out['energy'].cpu())

    metrics: Dict[str, float] = {}
    if c4p_rmsds:
        metrics['c4p_rmsd']  = torch.cat(c4p_rmsds).mean().item()
        metrics['bead_rmsd'] = torch.cat(bead_rmsds).mean().item()
    if energies:
        metrics['energy']    = torch.cat(energies).mean().item()

    return metrics


# ─── Convenience factory ──────────────────────────────────────────────────────

def build_tertiary_model(**kwargs) -> RNATertiaryModel:
    return RNATertiaryModel(**kwargs)
