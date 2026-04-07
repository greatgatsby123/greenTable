"""
RNA Bender: Grassmann Flow with Curvature for RNA Structure Prediction.

Design hypothesis:
    RNA folding is better represented as a structured trajectory through latent
    geometry than as isolated local pair interactions.

Hierarchy:
    zeroth order  — nucleotide embedding
    first order   — local 2-plane / subspace (backbone Plücker)
    second order  — change of subspace = curvature surrogate
    global order  — sequence-level Grassmann flow statistics

Architecture per layer (RNABenderLayer) — 5 steps:
  A: Hidden state input  h_i ∈ R^d
  B: Low-rank projection z_i = W_red h_i,  z_i ∈ R^r  (r ≪ d)
  C: Geometric feature extraction:
       backbone Plücker : p_{i,Δ} = normalize(z_i ∧ z_{i+Δ})  for Δ ∈ {1,2,4}
       structural edges : p_{ij}  = normalize(z_i ∧ z_j)
       curvature        : κ_i = p_{i+1,1} − 2·p_{i,1} + p_{i−1,1}
  D: MLP aggregation:
       g_i = Agg(φ_bb(·), φ_bp(·), φ_curv(·))
  E: Gated residual injection:
       u_i = W_g [h_i ∥ g_i]
       α_i = σ(W_α [h_i ∥ g_i])
       h̃_i = h_i + α_i ⊙ u_i
     Then FFN + LayerNorm.

Output heads:
  Primary  : pooled task head (regression / classification)
  Pair map : P̂_{ij} = σ(u_i · v_j)  — trained against BPP labels
  Auxiliary: per-token SS (3-class) + scalar MFE  (optional)

Loss:
  L = λ_task·L_task
    + λ_pair·L_pair       (pair map vs BPP supervision)
    + λ_curv·L_curv       (curvature regularisation: smooth unless data needs bending)
    + λ_cons·L_cons       (backbone–pairing geometry consistency at structural edges)
    + λ_ss·L_ss           (optional auxiliary SS cross-entropy)
    + λ_mfe·L_mfe         (optional auxiliary MFE regression)

Complexity: O(L · K) per layer; backbone and curvature steps are O(L).

References:
  Grassmann Flow — arXiv:2512.19428
  5'UTR CFPS data — PIIS2666675824001152 (Cell Systems 2024)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

# ─── Vocabulary (same as rna_structure_plucker) ────────────────────────────────

NUC_VOCAB: Dict[str, int] = {
    'A': 0, 'C': 1, 'G': 2, 'U': 3,
    'T': 3,
    'N': 4,
    '<PAD>': 5,
}
VOCAB_SIZE  = 6
PAD_ID  = NUC_VOCAB['<PAD>']
MASK_ID   = NUC_VOCAB['N']   
N_EDGE_FEATS = 3       
SS_IGNORE_IDX = -100

BACKBONE_OFFSETS = (1, 2, 4) 


def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
    """Return (max_len, d_model) sinusoidal positional encoding (no parameters)."""
    pe  = torch.zeros(max_len, d_model)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:d_model // 2])
    return pe


# ─── Plücker / wedge-product coordinate ───────────────────────────────────────

def plucker_coords(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Compute L2-normalised Plücker coordinates of span{u, v}.

    (u ∧ v)_{a<b} = u_a·v_b − u_b·v_a

    Args:
        u, v: (..., r)  —  any batch shape
    Returns:
        p: (..., r*(r-1)//2)  L2-normalised; zero vector when u ∥ v or either is zero
    """
    r = u.shape[-1]
    # Outer products
    uu = u.unsqueeze(-1)   # (..., r, 1)
    vv = v.unsqueeze(-2)   # (..., 1, r)
    outer = uu * vv    # (..., r, r)  :  outer[...,a,b] = u_a * v_b
    # Anti-symmetrise and extract upper triangle (a < b)
    anti = outer - outer.transpose(-1, -2)   # (..., r, r)
    idx  = torch.triu_indices(r, r, offset=1, device=u.device)
    p    = anti[..., idx[0], idx[1]]         # (..., C(r,2))
    # Clamp-min norm avoids NaN gradients near zero (safer than F.normalize)
    p_norm = p.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return p / p_norm


# ─── Backbone Plücker mixer (Step B + C for backbone) ─────────────────────────

class BackboneCurvatureMixer(nn.Module):
    """
    Steps B & C (backbone part) of one RNABenderLayer.

    For each position i computes:
      - z_i = W_red(h_i)                                    (Step B)
      - p_{i,Δ} = normalize(z_i ∧ z_{i+Δ})  for Δ ∈ offsets
      - κ_i = p_{i+1,1} − 2·p_{i,1} + p_{i−1,1}           (discrete curvature)

    Then projects each channel through a small MLP (Step D, backbone part).

    Args:
        reduced_dim : r
        model_dim   : d  (output feature dimension)
        offsets     : backbone offset values (default (1,2,4))
    """

    def __init__(
        self,
        reduced_dim: int,
        model_dim:   int,
        offsets:     Tuple[int, ...] = BACKBONE_OFFSETS,
    ):
        super().__init__()
        self.offsets   = offsets
        self.r         = reduced_dim
        plu_dim        = reduced_dim * (reduced_dim - 1) // 2
        self.plu_dim   = plu_dim
        hidden         = model_dim // 2

        # Step B
        self.W_red = nn.Linear(model_dim, reduced_dim, bias=True)

        # Step D backbone channel  (concat of all offsets → model_dim)
        self.phi_bb = nn.Sequential(
            nn.Linear(len(offsets) * plu_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, model_dim),
        )

        # Step D curvature channel
        self.phi_curv = nn.Sequential(
            nn.Linear(plu_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, model_dim),
        )

    def forward(
        self,
        h: torch.Tensor, 
        seq_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            z        : (B, L, r)        reduced representations
            g_bb     : (B, L, d)        backbone geometric features
            g_curv   : (B, L, d)        curvature features
            p_bb1    : (B, L, plu_dim)  offset-1 Plücker (for curvature loss)
            kappa    : (B, L, plu_dim)  curvature surrogates
        """
        B, L, _d = h.shape
        z = self.W_red(h)   # (B, L, r)   — Step B

        mask_f = seq_mask.float().unsqueeze(-1)   # (B, L, 1)

        p_list: List[torch.Tensor] = []
        for delta in self.offsets:
            if delta > 0:
                # z_j = z shifted left by delta; pad right with zeros
                zj = torch.zeros_like(z)
                zj[:, :L - delta, :] = z[:, delta:, :]
                # Both i and i+delta must be valid
                mj = torch.zeros_like(seq_mask, dtype=torch.float)
                mj[:, :L - delta] = seq_mask[:, delta:].float()
                m_ij = (seq_mask.float() * mj).unsqueeze(-1)  # (B,L,1)
            else:
                # delta == 0 would be self (unused), but guard anyway
                zj   = z
                m_ij = mask_f

            p = plucker_coords(z, zj) * m_ij   # (B, L, plu_dim)
            p_list.append(p)

        # Offset-1 Plücker used for curvature
        p_bb1 = p_list[0]   # (B, L, plu_dim)

        # Discrete curvature: κ_i = p_{i+1,1} − 2·p_{i,1} + p_{i−1,1}
        p_fwd = torch.zeros_like(p_bb1)
        p_bwd = torch.zeros_like(p_bb1)
        p_fwd[:, :L - 1, :] = p_bb1[:, 1:, :]    # p_{i+1,1}
        p_bwd[:, 1:,     :] = p_bb1[:, :L - 1, :]  # p_{i-1,1}
        kappa = p_fwd - 2 * p_bb1 + p_bwd         # (B, L, plu_dim)
        kappa = kappa * mask_f                     # zero at padding

        # Step D
        p_all  = torch.cat(p_list, dim=-1)   # (B, L, n_off * plu_dim)
        g_bb   = self.phi_bb(p_all)          # (B, L, d)
        g_curv = self.phi_curv(kappa)        # (B, L, d)

        return z, g_bb, g_curv, p_bb1, kappa


# ─── Structural edge mixer (Step C + D for graph edges) ───────────────────────

class StructuralEdgeMixer(nn.Module):
    """
    Sparse attention aggregation over structural (and local) graph edges.

    For each edge (i, j) computes p_{ij} = normalize(z_i ∧ z_j) and
    projects through a small MLP, then collects via learned attention weights.
    Finally gates the result against the hidden state (Step E partial).

    Mirrors StructureEdgePluckerLayer from rna_structure_plucker.py but
    operates on pre-computed z vectors passed in from BackboneCurvatureMixer.
    """

    def __init__(
        self,
        reduced_dim: int,
        model_dim: int,
        edge_feat_dim: int = N_EDGE_FEATS,
    ):
        super().__init__()
        plu_dim  = reduced_dim * (reduced_dim - 1) // 2
        d        = model_dim

        # Project Plücker + edge features → message
        self.phi_bp = nn.Sequential(
            nn.Linear(plu_dim + edge_feat_dim, d),
            nn.GELU(),
        )

        # Sparse edge attention scoring
        self.attn_score = nn.Linear(d + d + edge_feat_dim, 1)

    def forward(
        self,
        z: torch.Tensor,   # (B, L, r)
        h: torch.Tensor,   # (B, L, d)  — hidden state for attention query
        edge_idx: torch.Tensor,   # (B, L, K)  int64; -1 = padding
        edge_feat: torch.Tensor,   # (B, L, K, E)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            g_bp   : (B, L, d)        aggregated structural edge features
            p_struct : (B, L, K, plu) structural Plücker vectors (for consistency loss)
        """
        B, L, r = z.shape
        K        = edge_idx.shape[-1]

        valid_mask  = (edge_idx >= 0)                          # (B, L, K) bool
        clamped_idx = edge_idx.clamp(min=0).long()             # (B, L, K)

        # Gather z_j: (B, L, K, r)
        batch_idx = torch.arange(B, device=z.device).view(B, 1, 1).expand(B, L, K)
        z_j = z[batch_idx, clamped_idx]    # (B, L, K, r)
        z_i = z.unsqueeze(2).expand(B, L, K, r)

        # Plücker of each edge
        p_struct = plucker_coords(z_i, z_j)   # (B, L, K, plu_dim)
        p_struct = p_struct * valid_mask.unsqueeze(-1).float()

        # Message per edge
        feat = torch.cat([p_struct, edge_feat], dim=-1)   # (B, L, K, plu+E)
        msg  = self.phi_bp(feat)                           # (B, L, K, d)

        # Attention: score based on (msg, h_j, edge_feat)
        h_j    = h[batch_idx, clamped_idx]               # (B, L, K, d)
        scores = self.attn_score(
            torch.cat([msg, h_j, edge_feat], dim=-1)
        ).squeeze(-1)                                    # (B, L, K)
        scores = scores.masked_fill(~valid_mask, -1e4)
        attn   = torch.softmax(scores, dim=-1)           # (B, L, K)
        attn   = torch.nan_to_num(attn, nan=0.0)

        g_bp = (attn.unsqueeze(-1) * msg).sum(dim=2)    # (B, L, d)
        return g_bp, p_struct


# ─── One full RNA Bender layer ─────────────────────────────────────────────────

class RNABenderLayer(nn.Module):
    """
    One layer of RNA Bender implementing the 5-step geometric block.

    Step A: h_i ∈ R^d  (input)
    Step B: z_i = W_red h_i
    Step C: backbone Plücker {p_{i,Δ}}, structural edge Plücker {p_{ij}}, κ_i
    Step D: g_i = Agg(φ_bb, φ_bp, φ_curv)
    Step E: h̃_i = h_i + α_i ⊙ W_g[h_i ∥ g_i]  (gated residual)
    Then:   LayerNorm + FFN + residual
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

        # Steps B + C (backbone) + D (backbone)
        self.bb_mixer = BackboneCurvatureMixer(reduced_dim, d, offsets)

        # Steps C + D (structural edges)
        self.edge_mixer = StructuralEdgeMixer(reduced_dim, d)

        # Step D — aggregate three channels into one
        self.agg = nn.Sequential(
            nn.Linear(d * 3, d),
            nn.GELU(),
        )

        # Step E — gated residual
        self.W_g     = nn.Linear(d * 2, d)
        self.W_alpha = nn.Linear(d * 2, d)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d),
        )

        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,   # (B, L, d)
        edge_idx:  torch.Tensor,   # (B, L, K)
        edge_feat: torch.Tensor,   # (B, L, K, 3)
        seq_mask:  torch.Tensor,   # (B, L)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_new    : (B, L, d)
            p_bb1    : (B, L, plu_dim)  backbone offset-1 Plücker
            kappa    : (B, L, plu_dim)  curvature surrogates
            p_struct : (B, L, K, plu)   structural edge Plücker
        """
        h_in = self.ln1(h)                                         # pre-norm

        # Backbone: Steps B + C + D(bb+curv)
        z, g_bb, g_curv, p_bb1, kappa = self.bb_mixer(h_in, seq_mask)

        # Structural edges: Steps C + D(bp)
        g_bp, p_struct = self.edge_mixer(z, h_in, edge_idx, edge_feat)

        # Step D — aggregate
        g = self.agg(torch.cat([g_bb, g_bp, g_curv], dim=-1))     # (B, L, d)

        # Step E — gated residual
        hg    = torch.cat([h_in, g], dim=-1)                       # (B, L, 2d)
        u     = self.W_g(hg)
        alpha = torch.sigmoid(self.W_alpha(hg))
        h_new = h + self.drop(alpha * u)

        # FFN + residual
        h_new = h_new + self.drop(self.ffn(self.ln2(h_new)))

        return h_new, p_bb1, kappa, p_struct


# ─── Pair map head (memory-efficient bilinear form) ───────────────────────────

class PairMapHead(nn.Module):
    """
    Predict base-pair probability for all (i,j) pairs via symmetric bilinear.

    Parameterisation:  logit_{ij} = (u_i·v_j + u_j·v_i) / 2
    where u, v ∈ R^{d//2} are linearly projected from h.

    Memory: O(B·L·d) for projections + O(B·L²) for the score matrix.
    For L=100, B=64, d=128: ~25 MB — acceptable.
    """

    def __init__(self, model_dim: int):
        super().__init__()
        d2 = model_dim // 2
        self.proj = nn.Linear(model_dim, d2 * 2, bias=False)

    def forward(
        self,
        h: torch.Tensor,   # (B, L, d)
        seq_mask: torch.Tensor,   # (B, L)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits    : (B, L, L)  raw scores (unsymmetrised diagonal; apply sigmoid)
            pair_mask : (B, L, L)  True where both i and j are valid
        """
        d2   = self.proj.out_features // 2
        uv   = self.proj(h)                         # (B, L, 2*d2)
        u, v = uv[..., :d2], uv[..., d2:]           # (B, L, d2)

        logits = (torch.bmm(u, v.transpose(1, 2))
                + torch.bmm(v, u.transpose(1, 2))) * 0.5   # (B, L, L) symmetric

        pair_mask = seq_mask.unsqueeze(2) & seq_mask.unsqueeze(1)  # (B, L, L)
        return logits, pair_mask


# ─── Global Grassmann statistics (for analysis / conditioning) ────────────────

class GlobalGrassmannStats(nn.Module):
    """
    Compute sequence-level Grassmann statistics from backbone geometry.

    Stats (per layer):
        p̄     — mean backbone Plücker  (global geometric direction)
        m_κ   — mean ||κ_i||           (average bending)
        v_κ   — Var_i(||κ_i||)         (heterogeneity of bending)

    These can be used as global conditioning vectors or for analysis.
    Currently returned but not fed back into the model by default.
    """

    def compute(
        self,
        p_bb1: torch.Tensor,   # (B, L, plu_dim)
        kappa: torch.Tensor,   # (B, L, plu_dim)
        seq_mask: torch.Tensor,   # (B, L)
    ) -> torch.Tensor:
        """Returns: (B, plu_dim + 2)"""
        mf   = seq_mask.float()                          # (B, L)
        lens = mf.sum(dim=1, keepdim=True).clamp(min=1) # (B, 1)

        p_mean = (p_bb1 * mf.unsqueeze(-1)).sum(1) / lens          # (B, plu_dim)

        kappa_norm = kappa.norm(dim=-1)                             # (B, L)
        m_kappa    = (kappa_norm * mf).sum(1) / lens.squeeze(-1)   # (B,)
        v_kappa    = (
            ((kappa_norm - m_kappa.unsqueeze(1)) ** 2) * mf
        ).sum(1) / lens.squeeze(-1)                                 # (B,)

        return torch.cat([p_mean,
                          m_kappa.unsqueeze(1),
                          v_kappa.unsqueeze(1)], dim=1)             # (B, plu_dim+2)


# ─── Full RNA Bender model ─────────────────────────────────────────────────────

class RNABenderModel(nn.Module):
    """
    RNA Bender: Grassmann Flow with Curvature for RNA structure-informed
    expression / function prediction.

    Drop-in replacement for RNAStructureGrassmann.  Returns (logits, loss)
    from forward() when labels are provided.
    """

    def __init__(
        self,
        vocab_size: int   = VOCAB_SIZE,
        max_len:int   = 256,
        model_dim: int   = 128,
        num_layers: int   = 4,
        reduced_dim: int   = 16,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        pooling: str   = 'attention',   # 'attention' | 'mean'
        task:   str   = 'regression',  # 'regression' | 'classification' | 'folding'
        num_libraries: int   = 0,
        offsets: Tuple[int, ...] = BACKBONE_OFFSETS,
        # auxiliary structure heads (UTR-LM comparison)
        aux_struct:bool  = False,
        lambda_ss:  float = 0.1,
        lambda_mfe: float = 0.01,
        # geometric heads
        use_pair_head:bool  = True,
        lambda_pair:   float = 0.1,
        # curvature and consistency regularisers
        lambda_curv: float = 0.01,
        lambda_cons: float = 0.01,
        # positional encoding type
        pos_emb_type:  str   = 'sinusoidal',  # 'sinusoidal' | 'learned'
    ):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * model_dim

        self.model_dim = model_dim
        self.reduced_dim  = reduced_dim
        self.task  = task
        self.aux_struct  = aux_struct
        self.use_pair_head = use_pair_head
        self.pos_emb_type  = pos_emb_type

        # Loss weights (stored as buffers so they move with .to(device))
        self.lambda_ss = lambda_ss
        self.lambda_mfe  = lambda_mfe
        self.lambda_pair = lambda_pair
        self.lambda_curv = lambda_curv
        self.lambda_cons = lambda_cons

        # ── Embeddings ──────────────────────────────────────────────────────
        self.token_emb = nn.Embedding(vocab_size, model_dim, padding_idx=PAD_ID)
        if pos_emb_type == 'learned':
            self.pos_emb = nn.Embedding(max_len, model_dim)
        else:
            # Sinusoidal: no parameters, works at any length
            self.register_buffer('pos_enc', _sinusoidal_pe(max_len, model_dim), persistent=False)

        # ── Transformer blocks ──────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            RNABenderLayer(model_dim, reduced_dim, ff_dim, dropout, offsets)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(model_dim)

        # ── Pooling ─────────────────────────────────────────────────────────
        if pooling == 'attention':
            self.pool_attn = nn.Linear(model_dim, 1)
        else:
            self.pool_attn = None

        # ── Library conditioning (MRL cross-library) ────────────────────────
        if num_libraries > 0:
            self.lib_emb: Optional[nn.Embedding] = nn.Embedding(num_libraries, model_dim)
        else:
            self.lib_emb = None

        # ── Primary task head ───────────────────────────────────────────────
        # Not built for 'folding': loss is fully external and task_logits unused.
        if task != 'folding':
            out_dim = 1   # regression and binary classification both use scalar output
            self.task_head: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(model_dim, model_dim // 2),
                nn.GELU(),
                nn.Linear(model_dim // 2, out_dim),
            )
        else:
            self.task_head = None

        # ── Auxiliary structure heads ────────────────────────────────────────
        if aux_struct:
            self.ss_head  = nn.Linear(model_dim, 3)    # per-token: . ( )
            self.mfe_head = nn.Linear(model_dim, 1)    # scalar MFE

        # ── Pair map head ────────────────────────────────────────────────────
        if use_pair_head:
            self.pair_head = PairMapHead(model_dim)

        # ── Global Grassmann statistics (analysis / future conditioning) ────
        self.global_stats = GlobalGrassmannStats()

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

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _pool(self, h: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        """Attention-weighted or mean pooling. Returns (B, d)."""
        if self.pool_attn is not None:
            scores  = self.pool_attn(h).squeeze(-1)           # (B, L)
            scores  = scores.masked_fill(~seq_mask, -1e4)
            weights = torch.softmax(scores, dim=-1)
            weights = torch.nan_to_num(weights, nan=0.0)
            return (weights.unsqueeze(-1) * h).sum(dim=1)
        else:
            mf = seq_mask.float()
            return (h * mf.unsqueeze(-1)).sum(1) / mf.sum(1, keepdim=True).clamp(min=1)

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(
        self,
        input_ids: torch.Tensor,   # (B, L)
        edge_idx: torch.Tensor,   # (B, L, K)
        edge_feat: torch.Tensor,   # (B, L, K, 3)
        seq_mask: torch.Tensor,   # (B, L)
    ) -> Tuple[torch.Tensor, List, List, List]:
        """
        Run all transformer blocks.

        Returns:
            h          : (B, L, d)  final hidden states
            p_bb1_list : list[Tensor(B,L,plu)]  per-layer backbone Plücker
            kappa_list : list[Tensor(B,L,plu)]  per-layer curvature
            p_struct_list : list[Tensor(B,L,K,plu)]  per-layer structural Plücker
        """
        B, L = input_ids.shape
        if self.pos_emb_type == 'learned':
            pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
            h   = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))
        else:
            h   = self.drop(self.token_emb(input_ids) + self.pos_enc[:L])

        p_bb1_list, kappa_list, p_struct_list = [], [], []
        for block in self.blocks:
            h, p_bb1, kappa, p_struct = block(h, edge_idx, edge_feat, seq_mask)
            p_bb1_list.append(p_bb1)
            kappa_list.append(kappa)
            p_struct_list.append(p_struct)

        h = self.ln_f(h)
        return h, p_bb1_list, kappa_list, p_struct_list

    # ── Loss computation ──────────────────────────────────────────────────────

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        seq_mask: torch.Tensor,
        kappa_list: List[torch.Tensor],
        p_bb1_list: List[torch.Tensor],
        p_struct_list: List[torch.Tensor],
        edge_idx: torch.Tensor,
        edge_feat: torch.Tensor,
        pair_logits: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        ss_logits: Optional[torch.Tensor],
        mfe_pred: Optional[torch.Tensor],
        ss_labels: Optional[torch.Tensor],
        mfe_labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Return the total scalar loss."""

        # ── Primary task ─────────────────────────────────────────────────────
        if self.task == 'regression':
            loss = F.mse_loss(logits, labels.float())
        else:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        # ── Pair map loss (BPP supervision) ──────────────────────────────────
        # Scatter edge BPP values into a dense (B, L, L) target matrix.
        # Non-edge pairs contribute as negatives (target = 0), giving the
        # model a proper push-pull signal rather than only positive examples.
        if (self.use_pair_head
                and pair_logits is not None
                and self.lambda_pair > 0):
            bpp_dense = _edge_feat_to_dense_bpp(edge_feat, edge_idx,
                                                 L=pair_logits.shape[1])
            if pair_mask is not None:
                loss_pair = F.binary_cross_entropy_with_logits(
                    pair_logits[pair_mask], bpp_dense[pair_mask]
                )
                loss = loss + self.lambda_pair * loss_pair

        # ── Curvature regularisation ──────────────────────────────────────────
        # L_curv = mean over layers and valid positions of ||κ_i||²
        # Encourages smooth latent flows unless the data needs bending.
        if self.lambda_curv > 0 and kappa_list:
            mf        = seq_mask.float()
            n_valid   = mf.sum().clamp(min=1)
            loss_curv = sum(
                (kappa.pow(2).sum(-1) * mf).sum() / n_valid
                for kappa in kappa_list
            ) / len(kappa_list)
            loss = loss + self.lambda_curv * loss_curv

        # ── Backbone–pairing geometry consistency ─────────────────────────────
        # At structural edges (is_struct = edge_feat[...,2] == 1.0),
        # compare the backbone direction p_{i,1} with the edge Plücker p_{ij}.
        # Penalises wild geometric disagreement in regions that should be stable.
        if self.lambda_cons > 0 and p_struct_list and p_bb1_list:
            loss_cons = _consistency_loss(
                p_bb1_list[-1],   # last layer backbone Plücker
                p_struct_list[-1],  # last layer structural edge Plücker
                edge_feat,
            )
            loss = loss + self.lambda_cons * loss_cons

        # ── Auxiliary SS supervision ─────────────────────────────────────────
        if (self.aux_struct
                and ss_logits is not None
                and ss_labels is not None):
            loss_ss = F.cross_entropy(
                ss_logits.reshape(-1, 3),
                ss_labels.reshape(-1).long(),
                ignore_index=SS_IGNORE_IDX,
            )
            loss = loss + self.lambda_ss * loss_ss

        # ── Auxiliary MFE supervision ────────────────────────────────────────
        if (self.aux_struct
                and mfe_pred is not None
                and mfe_labels is not None):
            loss_mfe = F.mse_loss(mfe_pred, mfe_labels.float())
            loss = loss + self.lambda_mfe * loss_mfe

        return loss

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:torch.Tensor,
        seq_mask: torch.Tensor,
        # Primary edge inputs (rnastralign / folding collate names)
        edge_idx: Optional[torch.Tensor] = None,
        edge_feat: Optional[torch.Tensor] = None,
        # Alias names used by UTR collate (collate_rna) — mapped to edge_idx/edge_feat
        edge_index:Optional[torch.Tensor] = None,
        edge_attrs: Optional[torch.Tensor] = None,
        edge_mask: Optional[torch.Tensor] = None,   # accepted but not used
        # Task labless
        labels: Optional[torch.Tensor] = None,
        library_ids: Optional[torch.Tensor] = None,
        ss_labels:  Optional[torch.Tensor] = None,
        mfe_labels:  Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids   : (B, L) int64
            edge_idx    : (B, L, K) int64  — neighbour indices, -1 = padding
            edge_feat   : (B, L, K, 3) float  — [bp_prob, norm_dist, is_struct]
            seq_mask    : (B, L) bool  — True = valid token
            labels      : (B,) primary task labels — triggers built-in task loss
            library_ids : (B,) for MRL conditioning
            ss_labels   : (B, L) auxiliary SS targets
            mfe_labels  : (B,) auxiliary MFE targets

        Returns a dict with:
            task_logits   : (B,) pooled task output
            pair_logits   : (B, L, L)  if use_pair_head
            ss_logits     : (B, L, 3)  if aux_struct
            mfe_pred      : (B,)       if aux_struct
            kappa_list    : list of (B, L, plu) per-layer curvature
            p_bb1_list    : list of (B, L, plu) per-layer backbone Plücker
            p_struct_list : list of (B, L, K, plu) per-layer structural Plücker
            edge_feat     : the input edge_feat (forwarded for loss helpers)
            loss          : scalar total loss, present only when labels is not None
        """
        # Accept UTR-collate names (edge_index / edge_attrs) as aliases
        if edge_idx  is None: edge_idx  = edge_index
        if edge_feat is None: edge_feat = edge_attrs

        h, p_bb1_list, kappa_list, p_struct_list = self.encode(
            input_ids, edge_idx, edge_feat, seq_mask
        )

        out: Dict[str, torch.Tensor] = {
            'kappa_list':    kappa_list,
            'p_bb1_list':    p_bb1_list,
            'p_struct_list': p_struct_list,
            'edge_feat':     edge_feat,
        }

        # Auxiliary heads
        if self.aux_struct:
            out['ss_logits'] = self.ss_head(h)
            out['mfe_pred']  = self.mfe_head(self._pool(h, seq_mask)).squeeze(-1)

        if self.use_pair_head:
            pair_logits, _ = self.pair_head(h, seq_mask)
            out['pair_logits'] = pair_logits

        # Primary task (skipped for 'folding' — loss is fully external)
        if self.task_head is not None:
            pooled = self._pool(h, seq_mask)
            if self.lib_emb is not None and library_ids is not None:
                pooled = pooled + self.lib_emb(library_ids)
            out['task_logits'] = self.task_head(pooled).squeeze(-1)

        # Optional built-in loss (backward compat: when labels are passed directly)
        # Not supported for task='folding' (loss is always external there).
        if labels is not None and self.task_head is not None:
            pair_logits_  = out.get('pair_logits')
            pair_mask_    = (seq_mask.unsqueeze(2) & seq_mask.unsqueeze(1)
                             if pair_logits_ is not None else None)
            out['loss'] = self._compute_loss(
                out['task_logits'], labels, seq_mask,
                kappa_list, p_bb1_list, p_struct_list,
                edge_idx, edge_feat,
                pair_logits_, pair_mask_,
                out.get('ss_logits'), out.get('mfe_pred'),
                ss_labels, mfe_labels,
            )

        return out


# ─── Helpers for loss computation ─────────────────────────────────────────────

def _edge_feat_to_dense_bpp(
    edge_feat: torch.Tensor,   # (B, L, K, 3)  — channel 0 is bp_prob
    edge_idx: torch.Tensor,   # (B, L, K)     — neighbour indices, -1 = pad
    L:         int,
) -> torch.Tensor:
    """
    Reconstruct a dense (B, L, L) BPP target by scattering sparse edge values.

    For each valid edge (b, i, k):   bpp_dense[b, i, edge_idx[b,i,k]] = bp_prob
    All other entries remain 0 — non-edge pairs are treated as negatives.
    The matrix is symmetrised so that (i→j) and (j→i) are consistent.
    """
    B = edge_feat.shape[0]
    device = edge_feat.device

    bpp_dense  = torch.zeros(B, L, L, device=device, dtype=edge_feat.dtype)
    valid      = (edge_idx >= 0)                              # (B, L, K)
    clamped    = edge_idx.clamp(min=0).long()                 # (B, L, K)
    bp_probs   = edge_feat[..., 0] * valid.float()            # (B, L, K)

    # scatter_: bpp_dense[b, i, clamped[b,i,k]] = bp_probs[b,i,k]
    # (last write wins — fine because the graph is symmetric and values match)
    bpp_dense.scatter_(2, clamped, bp_probs)

    # Symmetrise: take the max of (i,j) and (j,i) so no edge is lost
    bpp_dense = torch.maximum(bpp_dense, bpp_dense.transpose(1, 2))
    # Zero out the diagonal (no self-pairing)
    bpp_dense.diagonal(dim1=1, dim2=2).zero_()

    return bpp_dense


def _consistency_loss(
    p_bb1:torch.Tensor,   # (B, L, plu_dim)  backbone offset-1 Plücker
    p_struct: torch.Tensor,   # (B, L, K, plu_dim)  structural edge Plücker
    edge_feat: torch.Tensor,  # (B, L, K, 3)
) -> torch.Tensor:
    """
    Consistency loss between backbone geometry at i and structural edge geometry.

    For each structural edge (i, j) [is_struct = edge_feat[...,2] > 0.5],
    penalise ||p_{i,1} − p_{ij}||².

    Interpretation: positions involved in base-pairs should have compatible
    local geometric directions in backbone and pairing channels.
    """
    is_struct = (edge_feat[..., 2] > 0.5)   # (B, L, K) bool

    if not is_struct.any():
        return torch.tensor(0.0, device=p_bb1.device, requires_grad=False)

    # Expand backbone Plücker to (B, L, K, plu_dim)
    p_bb_exp = p_bb1.unsqueeze(2).expand_as(p_struct)   # (B, L, K, plu_dim)

    # L2 distance between backbone and structural Plücker
    diff = (p_bb_exp - p_struct).pow(2).sum(-1)          # (B, L, K)

    masked_diff = diff[is_struct]
    if masked_diff.numel() == 0:
        return torch.tensor(0.0, device=p_bb1.device, requires_grad=False)

    return masked_diff.mean()


# ─── Re-exports for backward compatibility ────────────────────────────────────
# (These are the functions utr_datasets.py imports from rna_structure_plucker.
#  They are identical in semantics; this module provides its own copies so that
#  rna_bender.py can be used as a standalone drop-in.)

from rna_structure_plucker import (   # noqa: E402  (relative import at end)
    preprocess_sample,
    collate_rna,
    compute_bpp,
    compute_ss_mfe,
    encode_ss,
)

__all__ = [
    'RNABenderModel',
    'plucker_coords',
    'BackboneCurvatureMixer',
    'StructuralEdgeMixer',
    'RNABenderLayer',
    'PairMapHead',
    'GlobalGrassmannStats',
    'BACKBONE_OFFSETS',
    'VOCAB_SIZE', 'PAD_ID', 'MASK_ID', 'N_EDGE_FEATS',
    'preprocess_sample', 'collate_rna', 'compute_bpp',
    'compute_ss_mfe', 'encode_ss',
]
