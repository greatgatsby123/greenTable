"""
RNA Bender – Energy Ablation  (rna_bender_energy.py)

Hypothesis under test
    Grassmann local geometry  →  local mechanical energy  →  global fold inference

Key differences from rna_bender.py
──────────────────────────────────
INPUT
    Only sequence tokens + sinusoidal positional encoding.
    No BPP values, no structural-edge flags, no precomputed graph.
    Canonical pair mask and |i-j|/L distance are derived on the fly.

ENCODER
    Same BackboneCurvatureMixer (W_red, Plücker, curvature).
    RNABenderEnergyLayer = simplified layer without StructuralEdgeMixer.

ENERGY HEADS  (new)
    LocalEnergyHead   : e_local[i]   = f(h_i, z_i, p_i, κ_i)
    UnpairedEnergyHead: e_unp[i]     = u(h_i, z_i, p_i)
    PairEnergyHead    : e_pair[i,j]  = g(h_i,h_j, z_i,z_j, p_i,p_j, p_ij, dist, canon)

PHYSICAL PRIOR
    Hard mask: forbids i==j, |i-j| < min_hairpin+1, non-canonical pairs, padding.
    Forbidden entries are set to +1e9 before decoding.

DECODER  (new)
    FoldingDecoder: pseudoknot-free Nussinov DP with learned energies.
    Runs on CPU (detached numpy), returns non-differentiable structure.
    Energy of the decoded structure is then re-evaluated with live tensors.

LOSS  (new)
    Structured hinge:
        L_hinge = max(0,  E(x, S_true) − E(x, S_pred) + margin)
    Curvature smoothness (optional):
        L_smooth = mean_i ||κ_i||²
    Total:
        L = L_hinge + λ_smooth · L_smooth

Total energy of a structure S:
    E(x,S) = Σ_i e_local_i  +  Σ_{i ∈ unpaired(S)} e_unp_i  +  Σ_{(i,j)∈S} e_pair_ij

Ablation control
    Train two variants:
        A) without_grassmann=True  → PairEnergyHead uses only (h_i, h_j, dist, canon)
        B) without_grassmann=False → full Grassmann features in all heads
    Compare on family-split validation and TS0.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

# ─── Vocabulary ────────────────────────────────────────────────────────────────

NUC_VOCAB: Dict[str, int] = {
    'A': 0, 'C': 1, 'G': 2, 'U': 3,
    'T': 3,
    'N': 4,
    '<PAD>': 5,
}
VOCAB_SIZE = 6
PAD_ID     = NUC_VOCAB['<PAD>']

BACKBONE_OFFSETS = (1, 2, 4)

# Canonical RNA base pairs: A-U, U-A, G-C, C-G, G-U, U-G
CANONICAL_PAIRS = {(0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2)}

MIN_HAIRPIN = 4   # minimum number of unpaired nucleotides in a hairpin loop


# ─── Positional encoding ───────────────────────────────────────────────────────

def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    pe  = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
    return pe


# ─── Plücker / wedge-product coordinate ───────────────────────────────────────

def plucker_coords(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    L2-normalised Plücker coordinates of span{u, v}.
    Works for any leading batch shape; last dim is the vector dimension r.
    Returns (..., r*(r-1)//2).
    """
    r     = u.shape[-1]
    outer = u.unsqueeze(-1) * v.unsqueeze(-2)           # (..., r, r)
    anti  = outer - outer.transpose(-1, -2)
    idx   = torch.triu_indices(r, r, offset=1, device=u.device)
    p     = anti[..., idx[0], idx[1]]                   # (..., C(r,2))
    return p / p.norm(dim=-1, keepdim=True).clamp_min(1e-8)


# ─── Canonical pair mask ───────────────────────────────────────────────────────

def _compute_canon_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """
    Returns (B, L, L) bool: True where (i,j) is a canonical Watson-Crick / wobble pair.
    Derived purely from the sequence — no external structural information.
    """
    B, L   = input_ids.shape
    ids_i  = input_ids.unsqueeze(2).expand(B, L, L)   # (B, L, L)
    ids_j  = input_ids.unsqueeze(1).expand(B, L, L)   # (B, L, L)

    canon = torch.zeros(B, L, L, dtype=torch.bool, device=input_ids.device)
    for (a, b) in CANONICAL_PAIRS:
        canon = canon | ((ids_i == a) & (ids_j == b))
    return canon


# ─── Physical pair mask ────────────────────────────────────────────────────────

def _physical_pair_mask(
    seq_mask:       torch.Tensor,   # (B, L)
    canon_mask:     torch.Tensor,   # (B, L, L)
    min_hairpin:    int  = MIN_HAIRPIN,
    canonical_only: bool = True,
) -> torch.Tensor:
    """
    (B, L, L) bool: True where pairing (i, j) is physically permitted.
    Forbids: i==j, |i-j| <= min_hairpin, non-canonical (if canonical_only), padding.
    """
    B, L= seq_mask.shape
    device = seq_mask.device

    pos = torch.arange(L, device=device)
    dist_mat = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs()     # (L, L)

    valid = seq_mask.unsqueeze(2) & seq_mask.unsqueeze(1)         # (B, L, L)
    dist_ok  = (dist_mat > min_hairpin).unsqueeze(0).expand(B, L, L)

    if canonical_only:
        mask = valid & dist_ok & canon_mask
    else:
        mask = valid & dist_ok

    # No self-pairing
    eye = torch.eye(L, dtype=torch.bool, device=device).unsqueeze(0)
    mask = mask & ~eye

    return mask


# ─── Backbone curvature mixer (unchanged from rna_bender.py) ──────────────────

class BackboneCurvatureMixer(nn.Module):
    """
    Steps B & C (backbone) of one encoder layer.

      z_i     = W_red(h_i)
      p_{i,Δ} = normalize(z_i ∧ z_{i+Δ})  for Δ ∈ offsets
      κ_i     = p_{i+1,1} − 2·p_{i,1} + p_{i−1,1}
    """

    def __init__(
        self,
        reduced_dim: int,
        model_dim:   int,
        offsets:     Tuple[int, ...] = BACKBONE_OFFSETS,
    ):
        super().__init__()
        self.offsets = offsets
        self.r  = reduced_dim
        plu_dim = reduced_dim * (reduced_dim - 1) // 2
        self.plu_dim = plu_dim
        hidden = model_dim // 2

        self.W_red = nn.Linear(model_dim, reduced_dim, bias=True)
        self.phi_bb = nn.Sequential(
            nn.Linear(len(offsets) * plu_dim, hidden), nn.GELU(),
            nn.Linear(hidden, model_dim),
        )
        self.phi_curv = nn.Sequential(
            nn.Linear(plu_dim, hidden), nn.GELU(),
            nn.Linear(hidden, model_dim),
        )

    def forward(
        self,
        h:torch.Tensor,   # (B, L, d)
        seq_mask: torch.Tensor,   # (B, L)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: z, g_bb, g_curv, p_bb1, kappa
          z      : (B, L, r)
          g_bb   : (B, L, d)
          g_curv : (B, L, d)
          p_bb1  : (B, L, plu_dim)   offset-1 Plücker (raw, for energy heads)
          kappa  : (B, L, plu_dim)   discrete curvature (raw, for energy heads)
        """
        B, L, _ = h.shape
        z = self.W_red(h)                                # (B, L, r)
        mask_f = seq_mask.float().unsqueeze(-1)               # (B, L, 1)

        p_list: List[torch.Tensor] = []
        for delta in self.offsets:
            zj = torch.zeros_like(z)
            zj[:, :L - delta, :] = z[:, delta:, :]
            mj = torch.zeros_like(seq_mask, dtype=torch.float)
            mj[:, :L - delta] = seq_mask[:, delta:].float()
            m_ij = (seq_mask.float() * mj).unsqueeze(-1)
            p_list.append(plucker_coords(z, zj) * m_ij)

        p_bb1 = p_list[0]

        p_fwd = torch.zeros_like(p_bb1)
        p_bwd = torch.zeros_like(p_bb1)
        p_fwd[:, :L - 1, :] = p_bb1[:, 1:, :]
        p_bwd[:, 1:,     :] = p_bb1[:, :L - 1, :]
        kappa = (p_fwd - 2 * p_bb1 + p_bwd) * mask_f

        g_bb = self.phi_bb(torch.cat(p_list, dim=-1))
        g_curv = self.phi_curv(kappa)

        return z, g_bb, g_curv, p_bb1, kappa


# ─── Simplified encoder layer (no structural edges) ───────────────────────────

class RNABenderEnergyLayer(nn.Module):
    """
    RNA Bender layer without StructuralEdgeMixer.
    Aggregates only backbone + curvature geometry channels.
    Returns raw z, p_bb1, kappa for the energy heads.
    """

    def __init__(
        self,
        model_dim: int,
        reduced_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
        offsets: Tuple[int, ...] = BACKBONE_OFFSETS,
    ):
        super().__init__()
        d = model_dim

        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)

        self.bb_mixer = BackboneCurvatureMixer(reduced_dim, d, offsets)

        # Aggregate bb + curv (2 channels)
        self.agg = nn.Sequential(nn.Linear(d * 2, d), nn.GELU())

        self.W_g = nn.Linear(d * 2, d)
        self.W_alpha = nn.Linear(d * 2, d)

        self.ffn = nn.Sequential(
            nn.Linear(d, ff_dim), nn.GELU(), nn.Linear(ff_dim, d),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,   # (B, L, d)
        seq_mask: torch.Tensor,   # (B, L)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: h_new, z, p_bb1, kappa
        """
        h_in = self.ln1(h)

        z, g_bb, g_curv, p_bb1, kappa = self.bb_mixer(h_in, seq_mask)

        g  = self.agg(torch.cat([g_bb, g_curv], dim=-1))

        hg    = torch.cat([h_in, g], dim=-1)
        u     = self.W_g(hg)
        alpha = torch.sigmoid(self.W_alpha(hg))
        h_new = h + self.drop(alpha * u)
        h_new = h_new + self.drop(self.ffn(self.ln2(h_new)))

        return h_new, z, p_bb1, kappa


# ─── Energy heads ──────────────────────────────────────────────────────────────

class LocalEnergyHead(nn.Module):
    """
    Per-nucleotide local mechanical energy.
        e_local[i] = f(h_i, z_i, p_i, κ_i)

    Without Grassmann terms (ablation A): only h_i is used.
    """

    def __init__(
        self,
        model_dim:       int,
        z_dim:           int,
        plu_dim:         int,
        hidden:          int,
        without_grassmann: bool = False,
    ):
        super().__init__()
        self.without_grassmann = without_grassmann
        in_dim = model_dim if without_grassmann else model_dim + z_dim + 2 * plu_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        h: torch.Tensor,   # (B, L, d)
        z:     torch.Tensor,   # (B, L, r)
        p:     torch.Tensor,   # (B, L, plu_dim)
        kappa: torch.Tensor,   # (B, L, plu_dim)
    ) -> torch.Tensor:         # (B, L)
        if self.without_grassmann:
            x = h
        else:
            x = torch.cat([h, z, p, kappa], dim=-1)
        return self.net(x).squeeze(-1)


class UnpairedEnergyHead(nn.Module):
    """
    Cost for leaving a nucleotide unpaired.
        e_unp[i] = u(h_i, z_i, p_i)

    Without Grassmann terms (ablation A): only h_i is used.
    """

    def __init__(
        self,
        model_dim:         int,
        z_dim:             int,
        plu_dim:           int,
        hidden:            int,
        without_grassmann: bool = False,
    ):
        super().__init__()
        self.without_grassmann = without_grassmann
        in_dim = model_dim if without_grassmann else model_dim + z_dim + plu_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        h: torch.Tensor,   # (B, L, d)
        z: torch.Tensor,   # (B, L, r)
        p: torch.Tensor,   # (B, L, plu_dim)
    ) -> torch.Tensor:     # (B, L)
        if self.without_grassmann:
            x = h
        else:
            x = torch.cat([h, z, p], dim=-1)
        return self.net(x).squeeze(-1)


class PairEnergyHead(nn.Module):
    """
    Pairwise interaction energy.
        e_pair[i,j] = g(h_i, h_j, z_i, z_j, p_i, p_j, p_ij, |i-j|/L, canon_ij)

    Without Grassmann terms (ablation A): only (h_i, h_j, dist, canon) are used.

    Memory: O(B · L² · in_dim).  For L=200, d=128, r=16 this is ~800 MB/batch.
    Use smaller batch sizes or reduce reduced_dim if needed.
    """

    def __init__(
        self,
        model_dim:         int,
        z_dim:             int,
        plu_dim:           int,
        hidden:            int,
        without_grassmann: bool = False,
    ):
        super().__init__()
        self.without_grassmann = without_grassmann
        if without_grassmann:
            in_dim = 2 * model_dim + 2   # h_i, h_j, dist, canon
        else:
            in_dim = 2 * model_dim + 2 * z_dim + 3 * plu_dim + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        h: torch.Tensor,   # (B, L, d)
        z: torch.Tensor,   # (B, L, r)
        p: torch.Tensor,   # (B, L, plu_dim)
        seq_mask: torch.Tensor,   # (B, L)
        canon_mask: torch.Tensor,   # (B, L, L)
    ) -> torch.Tensor:              # (B, L, L)
        B, L, _ = h.shape
        device   = h.device

        # Distance feature |i - j| / L
        pos = torch.arange(L, device=device).float()
        dist = (pos.unsqueeze(1) - pos.unsqueeze(0)).abs() / max(L, 1)  # (L, L)
        dist = dist.unsqueeze(0).unsqueeze(-1).expand(B, L, L, 1)

        # Canonical pair indicator
        canon_f  = canon_mask.float().unsqueeze(-1)   # (B, L, L, 1)

        if self.without_grassmann:
            h_i = h.unsqueeze(2).expand(B, L, L, -1)
            h_j = h.unsqueeze(1).expand(B, L, L, -1)
            feat = torch.cat([h_i, h_j, dist, canon_f], dim=-1)
        else:
            h_i = h.unsqueeze(2).expand(B, L, L, -1)
            h_j = h.unsqueeze(1).expand(B, L, L, -1)
            z_i = z.unsqueeze(2).expand(B, L, L, -1)
            z_j = z.unsqueeze(1).expand(B, L, L, -1)
            p_i = p.unsqueeze(2).expand(B, L, L, -1)
            p_j = p.unsqueeze(1).expand(B, L, L, -1)
            p_ij = plucker_coords(z_i, z_j)           # (B, L, L, plu_dim)
            feat = torch.cat([h_i, h_j, z_i, z_j, p_i, p_j, p_ij, dist, canon_f], dim=-1)

        e_pair = self.net(feat).squeeze(-1)   # (B, L, L)

        # Symmetrise: e_ij = (e_ij + e_ji) / 2
        # Without this the fold energy depends on index-ordering artifacts.
        e_pair = (e_pair + e_pair.transpose(1, 2)) * 0.5

        # Mask out padded positions
        valid  = seq_mask.unsqueeze(2) & seq_mask.unsqueeze(1)
        e_pair = e_pair.masked_fill(~valid, 3e4)   # fp16-safe forbidden sentinel
        return e_pair


# ─── Nussinov DP (CPU / numpy) ────────────────────────────────────────────────

def _nussinov_decode_numpy(
    e_pair_np:  np.ndarray,   # (L, L) float64
    e_unp_np:   np.ndarray,   # (L,)   float64
    allowed_np: np.ndarray,   # (L, L) bool
    min_loop:   int = MIN_HAIRPIN,
) -> np.ndarray:              # (L,) int32, -1 = unpaired
    """
    Nussinov-like DP minimising total energy.  O(L³) time, O(L²) space.

    Recurrence for span [i, j]:
        E[i,j] = min(
            E[i+1, j]   + e_unp[i],          # i unpaired
            E[i,   j-1] + e_unp[j],          # j unpaired
            E[i+1, j-1] + e_pair[i,j],       # (i,j) paired  (if allowed)
            min_k E[i,k] + E[k+1,j]          # bifurcation
        )
    """
    L   = len(e_unp_np)
    INF = 1e18
    dp  = np.full((L, L), INF, dtype=np.float64)
    # ptr[i,j] encodes the winning case:
    #   0   = i unpaired
    #   1   = j unpaired
    #   2   = (i,j) paired
    #   k+3 = bifurcation at k  (k = ptr - 3)
    ptr = np.full((L, L), -1, dtype=np.int32)

    # Base: single nucleotides
    for i in range(L):
        dp[i, i]  = float(e_unp_np[i])
        ptr[i, i] = 0

    # Fill by increasing span length
    for span in range(2, L + 1):
        for i in range(L - span + 1):
            j    = i + span - 1
            best = INF
            bp   = -1

            # Case 0: i unpaired
            inner_ij = dp[i + 1, j] if i + 1 <= j else 0.0
            val = inner_ij + e_unp_np[i]
            if val < best:
                best = val; bp = 0

            # Case 1: j unpaired
            inner_ij2 = dp[i, j - 1] if i <= j - 1 else 0.0
            val = inner_ij2 + e_unp_np[j]
            if val < best:
                best = val; bp = 1

            # Case 2: (i, j) paired — requires at least min_loop unpaired inside
            if allowed_np[i, j] and (j - i) > min_loop:
                inner = dp[i + 1, j - 1] if (i + 1 <= j - 1) else 0.0
                val   = inner + e_pair_np[i, j]
                if val < best:
                    best = val; bp = 2

            # Case 3+: bifurcation
            for k in range(i, j):
                left  = dp[i, k]
                right = dp[k + 1, j] if (k + 1 <= j) else 0.0
                val   = left + right
                if val < best:
                    best = val; bp = k + 3

            dp[i, j]  = best
            ptr[i, j] = bp

    # Traceback (iterative stack)
    pairs: Dict[int, int] = {}
    stack = [(0, L - 1)]
    while stack:
        i, j = stack.pop()
        if i > j:
            continue
        if i == j:
            continue   # single nucleotide → unpaired (no partner to record)

        p = ptr[i, j]
        if p == 0:
            stack.append((i + 1, j))
        elif p == 1:
            stack.append((i, j - 1))
        elif p == 2:
            pairs[i] = j
            pairs[j] = i
            if i + 1 <= j - 1:
                stack.append((i + 1, j - 1))
        else:
            k = p - 3
            stack.append((i, k))
            if k + 1 <= j:
                stack.append((k + 1, j))

    result = np.full(L, -1, dtype=np.int32)
    for i, jj in pairs.items():
        result[i] = jj
    return result


class FoldingDecoder(nn.Module):
    """
    Pseudoknot-free DP folding decoder.

    Runs _nussinov_decode_numpy on CPU (detached tensors) for each sample in
    the batch, then returns a (B, L) int64 tensor of predicted partners.
    The DP is not differentiable; gradients flow only through the energy VALUES
    evaluated at the decoded structure positions.
    """

    def __init__(self, min_hairpin: int = MIN_HAIRPIN):
        super().__init__()
        self.min_hairpin = min_hairpin

    @torch.no_grad()
    def forward(
        self,
        e_pair:torch.Tensor,   # (B, L, L)  +1e9 for forbidden
        e_unp: torch.Tensor,   # (B, L)
        pair_mask:torch.Tensor,   # (B, L, L) bool
    ) -> torch.Tensor:             # (B, L) int64, -1 = unpaired
        return self._run_dp(e_pair, e_unp, pair_mask)

    @torch.no_grad()
    def forward_augmented(
        self,
        e_pair:torch.Tensor,   # (B, L, L)  masked
        e_unp: torch.Tensor,   # (B, L)
        pair_mask:torch.Tensor,   # (B, L, L) bool
        pair_labels: torch.Tensor,   # (B, L) int64  — true structure
    ) -> torch.Tensor:               # (B, L) int64  — loss-augmented decode
        """
        Loss-augmented Viterbi for SSVM training.

        Subtracts a per-position Hamming cost from the energy before decoding,
        so the decoder is biased toward structures that are simultaneously
        low-energy AND different from the true structure (the 'hardest negative').

        Augmentation rules (δ = 1 per wrong position):
          – pair (i,j) ∉ S*  →  ẽ_pair[i,j] -= 1
          – truly-paired i left unpaired  →  ẽ_unp[i] -= 1
        """
        B, L   = e_unp.shape
        device = e_pair.device

        # Build (B, L, L) mask of true pairs
        true_i = pair_labels.clamp(min=0)                              # (B, L)
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
        i_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        true_pair_mat = torch.zeros(B, L, L, dtype=torch.bool, device=device)
        is_paired = pair_labels >= 0                                   # (B, L)
        true_pair_mat[b_idx[is_paired], i_idx[is_paired], true_i[is_paired]] = True

        # Subtract 1 for pairs not in S* (allowed pairs only, to avoid inf - 1 issues)
        non_true = pair_mask & ~true_pair_mat
        aug_pair = e_pair - non_true.float()

        # Subtract 1 for unpaired energy at truly-paired positions
        aug_unp = e_unp - is_paired.float()

        return self._run_dp(aug_pair, aug_unp, pair_mask)

    def _run_dp(
        self,
        e_pair:torch.Tensor,
        e_unp:torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L   = e_unp.shape
        device = e_unp.device

        ep_cpu = e_pair.detach().cpu().numpy().astype(np.float64)
        eu_cpu = e_unp.detach().cpu().numpy().astype(np.float64)
        pm_cpu = pair_mask.cpu().numpy()

        results = [
            _nussinov_decode_numpy(ep_cpu[b], eu_cpu[b], pm_cpu[b], self.min_hairpin)
            for b in range(B)
        ]
        return torch.tensor(np.stack(results), dtype=torch.long, device=device)


# ─── Hamming loss helper ──────────────────────────────────────────────────────

def _hamming_loss(
    pred_pairs: torch.Tensor,   # (B, L) int64
    true_pairs: torch.Tensor,   # (B, L) int64
    seq_mask:   torch.Tensor,   # (B, L) bool
) -> torch.Tensor:              # (B,) float — per-sample normalised Hamming
    """
    Fraction of positions where pairing status differs between pred and true.
    Counts both false positives (wrong partner) and false negatives (missed pair).
    """
    mf    = seq_mask.float()
    n     = mf.sum(dim=1).clamp(min=1)
    wrong = (pred_pairs != true_pairs).float() * mf
    return wrong.sum(dim=1) / n


# ─── Energy-of-structure helper ───────────────────────────────────────────────

def _energy_of_structure(
    e_local:  torch.Tensor,   # (B, L)  — pairing propensity per nucleotide
    e_unp:    torch.Tensor,   # (B, L)
    e_pair:   torch.Tensor,   # (B, L, L)
    pairs:    torch.Tensor,   # (B, L) int64, -1 = unpaired
    seq_mask: torch.Tensor,   # (B, L) bool
) -> torch.Tensor:            # (B,)
    """
    Vectorised:
        E(x, S) = Σ_{i unpaired in S} e_unp_i
                + Σ_{(i,j) ∈ S, i < j} (e_pair[i,j] + e_local[i] + e_local[j])

    e_local acts as a per-nucleotide pairing propensity: negative → prefers pairing,
    positive → prefers staying unpaired.  It is added to the energy of each *pair*
    that involves position i, so it is structure-dependent and affects both DP
    decoding and the training loss.  (A constant sum over all i would cancel in
    E_true − E_pred and disappear from every gradient.)
    """
    B, L   = seq_mask.shape
    device = e_local.device

    # Indices for vectorised gather
    b_idx   = torch.arange(B, device=device).unsqueeze(1).expand(B, L)   # (B, L)
    i_idx   = torch.arange(L, device=device).unsqueeze(0).expand(B, L)   # (B, L)
    clamped = pairs.clamp(min=0)                                          # (B, L)

    # ── Unpaired energy ────────────────────────────────────────────────────────
    is_unp    = ((pairs == -1) & seq_mask).float()
    e_unp_sum = (e_unp * is_unp).sum(dim=1)                               # (B,)

    # ── Pair energy + local propensity, counted once per pair (i < j) ─────────
    is_pair = (pairs >= 0) & (i_idx < clamped) & seq_mask                 # (B, L)

    pair_e   = e_pair[b_idx, i_idx, clamped]                              # (B, L) — e_pair[i,j]
    local_i  = e_local                                                    # (B, L) — e_local[i]
    local_j  = e_local[b_idx, clamped]                                    # (B, L) — e_local[j]

    e_pair_sum = ((pair_e + local_i + local_j) * is_pair.float()).sum(dim=1)   # (B,)

    return e_unp_sum + e_pair_sum


# ─── Full RNA Bender Energy model ─────────────────────────────────────────────

class RNABenderEnergyModel(nn.Module):
    """
    RNA Bender Energy Ablation.

    Usage (training)::

        model = RNABenderEnergyModel(...)
        out   = model(input_ids, seq_mask, pair_labels=pair_labels)
        loss  = out['loss']
        loss.backward()

    pair_labels: (B, L) int64 tensor.
        pair_labels[b, i] = j  if position i is paired with j in the true structure
        pair_labels[b, i] = -1 if position i is unpaired
        Use dot_bracket_to_pair_labels() to build this from dot-bracket strings.

    Ablation control:
        without_grassmann=True  → energy heads use only hidden states (no z, p, κ)
        without_grassmann=False → energy heads use full Grassmann geometry (default)
    """

    def __init__(
        self,
        vocab_size:        int            = VOCAB_SIZE,
        max_len:           int            = 256,
        model_dim:         int            = 128,
        num_layers:        int            = 4,
        reduced_dim:       int            = 16,
        ff_dim:            Optional[int]  = None,
        dropout:           float          = 0.1,
        offsets:           Tuple[int,...] = BACKBONE_OFFSETS,
        energy_hidden:     int            = 64,
        loss_type:         str            = 'perceptron',  # 'perceptron' | 'ssvm'
        lambda_smooth:     float          = 0.01,
        min_hairpin:       int            = MIN_HAIRPIN,
        canonical_only:    bool           = True,
        without_grassmann: bool           = False,   # ablation A / B switch
    ):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * model_dim

        self.model_dim = model_dim
        self.reduced_dim = reduced_dim
        plu_dim = reduced_dim * (reduced_dim - 1) // 2
        self.plu_dim = plu_dim
        self.loss_type = loss_type
        self.lambda_smooth = lambda_smooth
        self.min_hairpin = min_hairpin
        self.canonical_only = canonical_only
        self.without_grassmann = without_grassmann

        # ── Embeddings ────────────────────────────────────────────────────────
        self.token_emb = nn.Embedding(vocab_size, model_dim, padding_idx=PAD_ID)
        self.register_buffer(
            'pos_enc', _sinusoidal_pe(max_len, model_dim), persistent=False
        )

        # ── Encoder ───────────────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            RNABenderEnergyLayer(model_dim, reduced_dim, ff_dim, dropout, offsets)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(model_dim)

        # ── Energy heads ──────────────────────────────────────────────────────
        self.local_head = LocalEnergyHead(
            model_dim, reduced_dim, plu_dim, energy_hidden, without_grassmann
        )
        self.unpaired_head = UnpairedEnergyHead(
            model_dim, reduced_dim, plu_dim, energy_hidden, without_grassmann
        )
        self.pair_head = PairEnergyHead(
            model_dim, reduced_dim, plu_dim, energy_hidden, without_grassmann
        )

        # ── Folding decoder ───────────────────────────────────────────────────
        self.decoder = FoldingDecoder(min_hairpin)

        self.drop = nn.Dropout(dropout)
        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].zero_()

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(
        self,
        input_ids: torch.Tensor,   # (B, L)
        seq_mask:  torch.Tensor,   # (B, L)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List]:
        """
        Returns: h, z, p_bb1, kappa  (from last layer), kappa_list (all layers)
        """
        B, L = input_ids.shape
        h    = self.drop(self.token_emb(input_ids) + self.pos_enc[:L])

        z = p_bb1 = kappa = None
        kappa_list: List[torch.Tensor] = []
        for block in self.blocks:
            h, z, p_bb1, kappa = block(h, seq_mask)
            kappa_list.append(kappa)

        h = self.ln_f(h)
        return h, z, p_bb1, kappa, kappa_list

    # ── Loss ─────────────────────────────────────────────────────────────────

    def _compute_loss(
        self,
        e_local: torch.Tensor,   # (B, L)
        e_unp: torch.Tensor,   # (B, L)
        e_pair: torch.Tensor,   # (B, L, L)  masked
        pair_labels:torch.Tensor,   # (B, L) int64
        pred_pairs: torch.Tensor,   # (B, L) int64  — argmin decoder output
        phys_mask: torch.Tensor,   # (B, L, L) bool
        kappa_list: List[torch.Tensor],
        seq_mask: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:
        e_true = _energy_of_structure(e_local, e_unp, e_pair, pair_labels, seq_mask)

        if self.loss_type == 'perceptron':
            # ── Perceptron loss ───────────────────────────────────────────────
            # L = E_true − E_pred.
            # pred_pairs is the argmin of E, so E_pred ≤ E_true always → L ≥ 0.
            # Goes to zero when the true structure is a global minimum.
            e_pred = _energy_of_structure(e_local, e_unp, e_pair, pred_pairs, seq_mask)
            loss_main = (e_true - e_pred).clamp(min=0.0).mean()

        else:
            # ── SSVM with loss-augmented decoding ─────────────────────────────
            # S^ = argmin_S (E(S) − Δ(S, S*))   ← hardest wrong competitor
            # L  = max(0,  E(S*) − E(S^) + Δ(S^, S*))
            # Can reach zero when S* is the global minimum with sufficient margin.
            aug_pairs = self.decoder.forward_augmented(
                e_pair, e_unp, phys_mask, pair_labels
            )
            e_aug = _energy_of_structure(e_local, e_unp, e_pair, aug_pairs, seq_mask)
            delta = _hamming_loss(aug_pairs, pair_labels, seq_mask)
            loss_main = torch.clamp(e_true - e_aug + delta, min=0.0).mean()

        # ── Curvature smoothness prior ────────────────────────────────────────
        if self.lambda_smooth > 0 and kappa_list:
            mf = seq_mask.float()
            n_valid = mf.sum().clamp(min=1)
            loss_sm = sum(
                (kappa.pow(2).sum(-1) * mf).sum() / n_valid
                for kappa in kappa_list
            ) / len(kappa_list)
            return loss_main + self.lambda_smooth * loss_sm

        return loss_main

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,                   # (B, L)
        seq_mask: torch.Tensor,                   # (B, L) bool
        pair_targets: Optional[torch.Tensor] = None,  # (B, L, L) float — from collate_rnastralign
        pair_labels: Optional[torch.Tensor] = None,  # (B, L) int64 — direct label override
        **kwargs,                                     # absorb edge_idx / edge_feat / ss_labels
    ) -> Dict[str, torch.Tensor]:
        """
        Compatible with the GeoFoldNet / RNABenderModel interface used by train_utr.py.

        Accepts pair_targets (B, L, L) from collate_rnastralign and converts them
        to pair_labels (B, L) internally.  Also accepts pair_labels directly.
        Extra kwargs (edge_idx, edge_feat, ss_labels, families, …) are silently ignored
        — this model derives all structure from sequence alone.

        Returns a dict with:
            pair_logits     : (B, L, L)  pseudo-logits from pred_pairs (+10/-10)
                              compatible with folding_loss / _evaluate_structure
            kappa_list      : list[(B, L, plu_dim)]  for curvature regularisation
            local_energy    : (B, L)
            unpaired_energy : (B, L)
            pair_energy     : (B, L, L)  with +1e9 for forbidden pairs
            pred_pairs      : (B, L) int64  decoded structure
            total_energy    : (B,)  E(x, S_pred)
            loss            : scalar, present when pair_targets or pair_labels given
        """
        # Derive pair_labels from pair_targets if not given directly
        if pair_labels is None and pair_targets is not None:
            partners = pair_targets.argmax(dim=-1).long()
            is_paired = (pair_targets.max(dim=-1).values > 0.5)
            pair_labels = torch.where(
                is_paired, partners, torch.full_like(partners, -1)
            )

        h, z, p_bb1, kappa, kappa_list = self.encode(input_ids, seq_mask)

        # Geometric context from sequence only
        canon_mask = _compute_canon_mask(input_ids)
        phys_mask = _physical_pair_mask(
            seq_mask, canon_mask, self.min_hairpin, self.canonical_only
        )

        # Energy tables
        e_local = self.local_head(h, z, p_bb1, kappa)                  # (B, L)
        e_unp = self.unpaired_head(h, z, p_bb1)                      # (B, L)
        e_pair = self.pair_head(h, z, p_bb1, seq_mask, canon_mask)    # (B, L, L)
        e_pair = e_pair.masked_fill(~phys_mask, 3e4)  # fp16-safe forbidden sentinel

        # Decode optimal structure (non-differentiable argmin)
        pred_pairs = self.decoder(e_pair, e_unp, phys_mask)            # (B, L)

        # Energy of decoded structure (differentiable)
        total_energy = _energy_of_structure(
            e_local, e_unp, e_pair, pred_pairs, seq_mask
        )

        # Build pair_logits from pred_pairs: +10 where paired, -10 elsewhere.
        # This makes the output compatible with folding_loss (BCE) and
        # _evaluate_structure (sigmoid threshold) without any changes there.
        B, L = pred_pairs.shape
        pair_logits = torch.full((B, L, L), -10.0,
                                 device=input_ids.device, dtype=e_pair.dtype)
        b_idx = torch.arange(B, device=input_ids.device).unsqueeze(1).expand(B, L)
        i_idx = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        clamped = pred_pairs.clamp(min=0)
        is_pair = (pred_pairs >= 0)
        pair_logits[b_idx[is_pair], i_idx[is_pair], clamped[is_pair]] = 10.0
        pair_logits[b_idx[is_pair], clamped[is_pair], i_idx[is_pair]] = 10.0

        out: Dict[str, object] = {
            'pair_logits': pair_logits,
            'kappa_list': kappa_list,
            'local_energy': e_local,
            'unpaired_energy': e_unp,
            'pair_energy': e_pair,
            'pred_pairs': pred_pairs,
            'total_energy': total_energy,
        }

        if pair_labels is not None:
            out['loss'] = self._compute_loss(
                e_local, e_unp, e_pair, pair_labels, pred_pairs,
                phys_mask, kappa_list, seq_mask,
            )

        return out


# ─── Utility: dot-bracket → pair labels ───────────────────────────────────────

def dot_bracket_to_pair_labels(ss: str, dtype=np.int32) -> np.ndarray:
    """
    Convert a dot-bracket secondary structure string to a (L,) int32 array.
    result[i] = j if position i is paired with j, else -1.
    Handles standard '()', '[]', '{}' bracket pairs.
    """
    L      = len(ss)
    result = np.full(L, -1, dtype=np.int32)
    stacks: Dict[str, List[int]] = {'(': [], '[': [], '{': []}
    close_to_open = {')': '(', ']': '[', '}': '{'}

    for i, c in enumerate(ss):
        if c in stacks:
            stacks[c].append(i)
        elif c in close_to_open:
            opener = close_to_open[c]
            if stacks[opener]:
                j = stacks[opener].pop()
                result[i] = j
                result[j] = i
        # '.' and unknown characters → unpaired (default -1)

    return result


def pair_labels_to_dot_bracket(labels: np.ndarray) -> str:
    """Inverse of dot_bracket_to_pair_labels (single pseudoknot-free layer)."""
    L  = len(labels)
    ss = ['.'] * L
    for i in range(L):
        j = labels[i]
        if j != -1 and i < j:
            ss[i] = '('
            ss[j] = ')'
    return ''.join(ss)


# ─── Re-exports for compatibility with existing training infrastructure ────────

try:
    from rna_structure_plucker import (   # noqa: E402
        preprocess_sample,
        collate_rna,
        compute_bpp,
        compute_ss_mfe,
        encode_ss,
    )
except ImportError:
    pass   # allow standalone use without the full repo

__all__ = [
    # Model
    'RNABenderEnergyModel',
    # Building blocks
    'plucker_coords',
    'BackboneCurvatureMixer',
    'RNABenderEnergyLayer',
    'LocalEnergyHead',
    'UnpairedEnergyHead',
    'PairEnergyHead',
    'FoldingDecoder',
    # Helpers
    '_compute_canon_mask',
    '_physical_pair_mask',
    '_energy_of_structure',
    '_nussinov_decode_numpy',
    'dot_bracket_to_pair_labels',
    'pair_labels_to_dot_bracket',
    # Constants
    'BACKBONE_OFFSETS',
    'VOCAB_SIZE', 'PAD_ID', 'NUC_VOCAB',
    'CANONICAL_PAIRS', 'MIN_HAIRPIN',
]
