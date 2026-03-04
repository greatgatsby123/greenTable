"""
Structure-Edge Plucker Mixer for RNA 5'UTR Expression Prediction.

Core idea (replacing fixed-offset Grassmann Flow with biology-aware message passing):

  Instead of pairing tokens at fixed sequence offsets Δ (as in arXiv:2512.19428),
  we pair tokens that are biologically connected:
    1. Local sequence neighbors (t ± 1, t ± 2)
    2. RNA base-pair partners from a base-pair probability (BPP) matrix

  On each edge (t, j), the Plücker / wedge-product feature
      p_{t,j} = z_t ∧ z_j  ∈ R^(r choose 2)
  encodes the geometric relationship between the two token representations.
  This is zero iff z_t ∥ z_j (redundant) and largest when the pair spans
  an independent 2-D subspace of R^r.

Architecture per layer (StructureGrassmannBlock):
  Pre-norm:  h_in = LayerNorm(h)
  Inside StructureEdgePluckerLayer (receives h_in, returns delta):
    1.  z_t      = W_red(h_in_t)                            ∈ R^r
    2.  p_{t,j}  = normalize(z_t ∧ z_j)                    ∈ R^(r*(r-1)/2)
    3.  m_{t,j}  = W_plu([p_{t,j}; edge_attrs_{t,j}])      ∈ R^d
    4.  α_{t,j}  = softmax_j( W_attn([h_in_t; h_in_j; e]) )  (sparse edge attn)
    5.  m_t      = Σ_j α_{t,j} · m_{t,j}                   ∈ R^d
    6.  β_t      = sigmoid(W_gate([h_in_t; m_t]))
        delta_t  = (1 − β_t) · m_t   ← layer returns this, NOT the full update
  Residual:  h ← h + dropout(delta)
  FFN sub-layer follows the same pre-norm + residual pattern.

References:
  - Grassmann Flow:  arXiv:2512.19428
  - 5'UTR CFPS data: PIIS2666675824001152 (Cell Systems 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any


# ─── Nucleotide vocabulary ─────────────────────────────────────────────────────

NUC_VOCAB: Dict[str, int] = {
    'A': 0, 'C': 1, 'G': 2, 'U': 3,
    'T': 3,   # treat T as U
    'N': 4,   # ambiguous
    '<PAD>': 5,
}
VOCAB_SIZE = 6
PAD_ID = NUC_VOCAB['<PAD>']

# Edge feature channels:  [bp_prob, norm_seq_dist, is_struct_edge]
N_EDGE_FEATS = 3

# Secondary-structure symbol vocabulary  (dot-bracket)
SS_VOCAB: Dict[str, int] = {'.': 0, '(': 1, ')': 2}
N_SS_CLASSES = 3
SS_IGNORE_IDX = -100   # padding target — ignored by cross-entropy


def encode_sequence(seq: str) -> List[int]:
    """Convert a nucleotide string to a list of integer token IDs."""
    return [NUC_VOCAB.get(c.upper(), NUC_VOCAB['N']) for c in seq]


# ─── Base-pair probability (BPP) computation ──────────────────────────────────

def compute_bpp(seq: str) -> np.ndarray:
    """
    Compute the L×L base-pair probability matrix using ViennaRNA.

    Requires the ViennaRNA Python bindings (``pip install ViennaRNA``).
    Falls back to a zero matrix if ViennaRNA is not installed, which gives
    a local-only edge graph (still functional, just no long-range structure).

    Args:
        seq: Nucleotide string (A/C/G/U/T).

    Returns:
        (L, L) float32 array, symmetric, diagonal zero, values in [0, 1].
    """
    L = len(seq)
    seq_rna = seq.upper().replace('T', 'U')
    try:
        import RNA  # ViennaRNA Python bindings
        fc = RNA.fold_compound(seq_rna)
        fc.pf()  # partition function → fills bpp table
        raw = np.array(fc.bpp())  # (L+1, L+1), 1-indexed
        bpp = raw[1:L + 1, 1:L + 1].astype(np.float32)
        bpp = bpp + bpp.T          # ViennaRNA fills upper triangle only
        np.fill_diagonal(bpp, 0.0)
        return bpp
    except ImportError:
        return np.zeros((L, L), dtype=np.float32)


# ─── Secondary-structure / MFE computation ────────────────────────────────────

def compute_ss_mfe(seq: str) -> Tuple[str, float]:
    """
    Compute the MFE secondary structure (dot-bracket string) and minimum free
    energy using ViennaRNA's fast MFE fold (RNA.fold).

    This is the same quantity UTR-LM uses as an auxiliary training target:
    the dot-bracket string becomes the per-token SS label and the scalar MFE
    becomes the sequence-level regression target.

    Falls back to an all-unpaired structure and 0.0 kcal/mol if ViennaRNA
    is not installed.

    Args:
        seq: Nucleotide string (A/C/G/U/T).

    Returns:
        (ss, mfe): dot-bracket string of length L and free energy (kcal/mol).
    """
    seq_rna = seq.upper().replace('T', 'U')
    try:
        import RNA  # ViennaRNA Python bindings
        structure, mfe = RNA.fold(seq_rna)
        return structure, float(mfe)
    except ImportError:
        return '.' * len(seq), 0.0


def encode_ss(ss: str) -> np.ndarray:
    """Convert a dot-bracket string to integer class IDs (int8 array)."""
    return np.array([SS_VOCAB.get(c, 0) for c in ss], dtype=np.int8)


# ─── Sparse edge builder ───────────────────────────────────────────────────────

def build_padded_edges(
    L: int,
    bpp: Optional[np.ndarray] = None,
    local_offsets: Tuple[int, ...] = (-2, -1, 1, 2),
    top_k_struct: int = 4,
    bp_threshold: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the padded edge tensors for a single RNA sequence.

    Structure edges are made symmetric: if t chooses j as a top-K partner,
    j is also given t as a neighbor (even if t is not in j's own top-K).
    K_max = len(local_offsets) + 2*top_k_struct to accommodate the extra
    reverse edges that symmetry can add.

    Edge attribute channels (3):
        0 → base-pair probability P_{t,j}  (real BPP for both local and
            structure edges when bpp is provided; 0 otherwise)
        1 → normalised sequence distance |t−j|/(L−1)
        2 → 1.0 if structure edge, 0.0 if local sequence edge

    Args:
        L:              sequence length
        bpp:            (L, L) BPP matrix (symmetric, diagonal=0);
                        None → local-only edges
        local_offsets:  sequence offsets for local edges
        top_k_struct:   max structure edges each node may *initiate*;
                        reverse edges from partners may add up to K more
        bp_threshold:   minimum BPP for a structure edge to be included

    Returns:
        edge_index: (L, K_max) int32, -1 = padding
        edge_mask:  (L, K_max) bool,  True = valid edge
        edge_attrs: (L, K_max, 3) float32
    """
    # Worst-case: each node has len(local_offsets) local + 2*top_k_struct struct
    K_max = len(local_offsets) + 2 * top_k_struct
    norm_denom = max(L - 1, 1)

    # Build adjacency as per-node dicts  {j: (bp_prob, norm_dist, is_struct)}
    # Using dict preserves insertion order (Python 3.7+) and deduplicates.
    adj: List[Dict[int, Tuple[float, float, float]]] = [{} for _ in range(L)]

    # Phase 1: local sequence edges (already bidirectional via ± offsets)
    for t in range(L):
        for delta in local_offsets:
            j = t + delta
            if 0 <= j < L and j not in adj[t]:
                bp_prob = float(bpp[t, j]) if bpp is not None else 0.0
                adj[t][j] = (bp_prob, abs(delta) / norm_denom, 0.0)

    # Phase 2: structure edges — collect and symmetrize
    if bpp is not None and top_k_struct > 0:
        for t in range(L):
            row = bpp[t].copy()
            row[t] = 0.0
            # Exclude positions already in adj[t] to avoid upgrading local→struct
            for j in adj[t]:
                row[j] = 0.0
            sorted_idx = np.argsort(row)[::-1]
            n_added = 0
            for j in sorted_idx:
                if n_added >= top_k_struct or row[j] < bp_threshold:
                    break
                bp_prob = float(row[j])
                norm_dist = abs(t - j) / norm_denom
                # Forward edge t → j
                if j not in adj[t]:
                    adj[t][j] = (bp_prob, norm_dist, 1.0)
                # Reverse edge j → t  (symmetry: t becomes j's partner too)
                if t not in adj[j]:
                    adj[j][t] = (bp_prob, norm_dist, 1.0)
                n_added += 1

    # Phase 3: pack into padded arrays
    edge_index = np.full((L, K_max), -1, dtype=np.int32)
    edge_attrs = np.zeros((L, K_max, N_EDGE_FEATS), dtype=np.float32)

    for t in range(L):
        for slot, (j, (bp_prob, norm_dist, is_struct)) in enumerate(adj[t].items()):
            if slot >= K_max:
                break  # safety cap (shouldn't trigger with K_max as above)
            edge_index[t, slot] = j
            edge_attrs[t, slot] = [bp_prob, norm_dist, is_struct]

    edge_mask = edge_index >= 0
    return edge_index, edge_mask, edge_attrs


def preprocess_sample(
    seq: str,
    bpp: Optional[np.ndarray] = None,
    local_offsets: Tuple[int, ...] = (-2, -1, 1, 2),
    top_k_struct: int = 4,
    bp_threshold: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Preprocess one sequence into numpy arrays ready for collation.

    If bpp is None *and* ViennaRNA is available, BPP is computed automatically.
    Pass ``bpp=np.zeros((L, L))`` explicitly to disable structure edges.

    Returns:
        dict with keys 'input_ids', 'edge_index', 'edge_mask', 'edge_attrs'
        (optionally 'label' if the caller adds it).
    """
    if bpp is None:
        bpp = compute_bpp(seq)

    input_ids = np.array(encode_sequence(seq), dtype=np.int32)
    edge_index, edge_mask, edge_attrs = build_padded_edges(
        len(seq), bpp, local_offsets, top_k_struct, bp_threshold
    )
    return {
        'input_ids': input_ids,
        'edge_index': edge_index,
        'edge_mask': edge_mask,
        'edge_attrs': edge_attrs,
    }


# ─── Batch collation ──────────────────────────────────────────────────────────

def collate_rna(
    samples: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """
    Collate variable-length samples into a padded batch.

    Each sample is the dict returned by ``preprocess_sample``, optionally
    augmented with a float ``'label'`` key.

    Returns a dict of tensors:
        input_ids  (B, L_max)             long
        edge_index (B, L_max, K_max)      long   (-1 = padding slot)
        edge_mask  (B, L_max, K_max)      bool
        edge_attrs (B, L_max, K_max, 3)   float
        seq_mask   (B, L_max)             bool   (True = real token)
        labels     (B,)                   float  (only if present in samples)
    """
    B = len(samples)
    L_max = max(s['input_ids'].shape[0] for s in samples)
    K_max = samples[0]['edge_index'].shape[1]

    input_ids  = torch.full((B, L_max), PAD_ID, dtype=torch.long)
    edge_index = torch.full((B, L_max, K_max), -1, dtype=torch.long)
    edge_mask  = torch.zeros(B, L_max, K_max, dtype=torch.bool)
    edge_attrs = torch.zeros(B, L_max, K_max, N_EDGE_FEATS)
    seq_mask   = torch.zeros(B, L_max, dtype=torch.bool)

    for i, s in enumerate(samples):
        L = s['input_ids'].shape[0]
        input_ids[i, :L]        = torch.from_numpy(s['input_ids'].astype(np.int64))
        edge_index[i, :L, :]    = torch.from_numpy(s['edge_index'].astype(np.int64))
        edge_mask[i, :L, :]     = torch.from_numpy(s['edge_mask'])
        edge_attrs[i, :L, :, :] = torch.from_numpy(s['edge_attrs'])
        seq_mask[i, :L]         = True

    batch: Dict[str, torch.Tensor] = {
        'input_ids':  input_ids,
        'edge_index': edge_index,
        'edge_mask':  edge_mask,
        'edge_attrs': edge_attrs,
        'seq_mask':   seq_mask,
    }
    if 'label' in samples[0]:
        batch['labels'] = torch.tensor(
            [s['label'] for s in samples], dtype=torch.float
        )
    # Auxiliary SS labels: (B, L_max), padded with SS_IGNORE_IDX (-100)
    if 'ss_ids' in samples[0]:
        ss_ids = torch.full((B, L_max), SS_IGNORE_IDX, dtype=torch.long)
        for i, s in enumerate(samples):
            L = s['input_ids'].shape[0]
            ss_ids[i, :L] = torch.from_numpy(s['ss_ids'].astype(np.int64))
        batch['ss_ids'] = ss_ids
    # Auxiliary MFE labels: (B,) float
    if 'mfe' in samples[0]:
        batch['mfe'] = torch.tensor([float(s['mfe']) for s in samples], dtype=torch.float)
    return batch


# ─── Plücker encoder (shared with grassmann_v4, generalised to edges) ─────────

class PluckerEncoder(nn.Module):
    """
    Compute L2-normalised Plücker (wedge-product) coordinates for a batch
    of (source, target) vector pairs in R^r.

    For vectors u, v ∈ R^r the wedge product u ∧ v has r*(r-1)/2 components:
        (u ∧ v)_{a,b} = u_a * v_b − u_b * v_a    for a < b
    """

    def __init__(self, reduced_dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        idx_i, idx_j = zip(
            *[(i, j) for i in range(reduced_dim) for j in range(i + 1, reduced_dim)]
        )
        self.register_buffer('idx_i', torch.tensor(idx_i, dtype=torch.long))
        self.register_buffer('idx_j', torch.tensor(idx_j, dtype=torch.long))

    def forward(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u, v: (..., r)   — arbitrary leading batch dimensions

        Returns:
            p_hat: (..., r*(r-1)/2)  L2-normalised wedge product
        """
        p = u[..., self.idx_i] * v[..., self.idx_j] \
          - u[..., self.idx_j] * v[..., self.idx_i]
        return p / p.norm(dim=-1, keepdim=True).clamp(min=self.eps)


# ─── Core message-passing layer ────────────────────────────────────────────────

class StructureEdgePluckerLayer(nn.Module):
    """
    One structure-edge Plücker message-passing layer.

    Replaces the fixed-Δ sliding-window loop in CausalGrassmannMixing with
    a sparse graph where the neighborhood of each token t is:
        E(t) = {local sequence neighbors} ∪ {top-K base-pair partners}

    Returns a *delta* tensor (B, L, d) intended to be added to h by the
    enclosing block via a standard pre-norm residual:
        h = h + dropout(layer(ln(h), ...))

    The gate controls how much of the structural message is admitted:
        beta  = sigmoid(W_gate([h_in; m]))   # h_in is the pre-normed input
        delta = (1 − beta) * m               # beta≈1 → small update (safe init)

    No LayerNorm or dropout live inside this module; both belong in the block.

    Memory note: with r=32, plucker_dim=496, K=12 neighbors, L=100, B=32
    the wedge-product tensor is ~76 MB per layer.  Use reduced_dim=16
    (plucker_dim=120, ~19 MB) if memory is tight.
    """

    def __init__(
        self,
        model_dim: int,
        reduced_dim: int = 32,
        n_edge_feats: int = N_EDGE_FEATS,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.reduced_dim = reduced_dim
        plucker_dim = reduced_dim * (reduced_dim - 1) // 2
        self.plucker_dim = plucker_dim

        self.plucker = PluckerEncoder(reduced_dim, eps)

        # Step 1: R^d → R^r
        self.W_red = nn.Linear(model_dim, reduced_dim)

        # Step 3: [p_hat ∈ R^P ; edge_attrs ∈ R^e] → R^d
        self.W_plu = nn.Linear(plucker_dim + n_edge_feats, model_dim)

        # Step 4: [h_src; h_nbr; edge_attrs] → scalar edge-attention logit
        # Initialised to zero so all edges start with equal weight.
        self.W_attn = nn.Linear(2 * model_dim + n_edge_feats, 1)

        # Step 6: gate — how much of the message to let through
        #   beta  = sigmoid(W_gate([h_in; m]))
        #   delta = (1 − beta) * m
        self.W_gate = nn.Linear(2 * model_dim, model_dim)

        # Weight init is handled by RNAStructureGrassmann._init_weights via
        # self.apply(...), which runs after construction.  W_attn is explicitly
        # re-zeroed there after the global apply.

    def forward(
        self,
        h: torch.Tensor,            # (B, L, d)  — pre-normed by caller
        edge_index: torch.Tensor,   # (B, L, K) long, −1 = padding
        edge_mask: torch.Tensor,    # (B, L, K) bool
        edge_attrs: torch.Tensor,   # (B, L, K, n_edge_feats)
    ) -> torch.Tensor:              # (B, L, d)  — delta, NOT the full update
        B, L, d = h.shape
        K = edge_index.shape[2]
        device = h.device

        # ── Step 1: reduce to r-dim ──────────────────────────────────────────
        z = self.W_red(h)                                          # (B, L, r)

        # ── Gather neighbour vectors (safe clamp for padding slots) ──────────
        safe_idx = edge_index.clamp(min=0)                         # (B, L, K)
        b_idx = (torch.arange(B, device=device)[:, None, None]
                 .expand(B, L, K))                                 # (B, L, K)

        z_src = z.unsqueeze(2).expand(B, L, K, self.reduced_dim)  # (B, L, K, r)
        z_nbr = z[b_idx, safe_idx]                                 # (B, L, K, r)

        # Zero padding slots before any computation so token-0's embedding never
        # enters the wedge product or attention logit.  edge_attrs is already 0
        # at padding slots from build_padded_edges, so no separate masking needed.
        mask_f = edge_mask.float().unsqueeze(-1)                   # (B, L, K, 1)
        z_nbr  = z_nbr * mask_f

        # ── Step 2: Plücker (wedge) features ────────────────────────────────
        p_hat = self.plucker(z_src, z_nbr)                         # (B, L, K, P)

        # ── Step 3: map to model dim, conditioned on edge attributes ─────────
        # Multiply by mask_f to zero the W_plu bias at padding slots.
        m_per_edge = self.W_plu(
            torch.cat([p_hat, edge_attrs], dim=-1)
        ) * mask_f                                                  # (B, L, K, d)

        # ── Step 4: sparse edge attention ────────────────────────────────────
        h_src = h.unsqueeze(2).expand(B, L, K, d)                 # (B, L, K, d)
        h_nbr = h[b_idx, safe_idx] * mask_f                        # (B, L, K, d)

        attn_logits = self.W_attn(
            torch.cat([h_src, h_nbr, edge_attrs], dim=-1)
        ).squeeze(-1)                                              # (B, L, K)
        # Mask padding slots to -inf; softmax then assigns them ~0 weight.
        # Use dtype.min/2 so the fill value is safe under fp16 AMP (−1e9 overflows).
        neg_inf = torch.finfo(attn_logits.dtype).min / 2
        attn_logits = attn_logits.masked_fill(~edge_mask, neg_inf)
        attn_weights = torch.softmax(attn_logits, dim=-1)          # (B, L, K)

        # ── Step 5: aggregate ────────────────────────────────────────────────
        m = (attn_weights.unsqueeze(-1) * m_per_edge).sum(dim=2)  # (B, L, d)

        # Nodes with no valid neighbours get zero (prevents NaN when all -inf)
        has_nbr = edge_mask.any(dim=-1, keepdim=True).float()
        m = m * has_nbr

        # ── Step 6: gated delta ──────────────────────────────────────────────
        # W_gate starts with std=0.02 weights and zero bias (from global init),
        # so at init beta = sigmoid(~0) ≈ 0.5 → delta ≈ 0.5·m.
        # (Only W_attn is explicitly zeroed. W_gate is intentionally left at
        #  the halfway point so the gate is neither open nor shut at epoch 0.)
        beta = torch.sigmoid(
            self.W_gate(torch.cat([h, m], dim=-1))
        )                                                          # (B, L, d)
        return (1.0 - beta) * m                                    # delta only


# ─── Transformer block ────────────────────────────────────────────────────────

class StructureGrassmannBlock(nn.Module):
    """
    Pre-norm transformer block:
        [LN → StructureEdgePluckerLayer → dropout → residual]
        [LN → FFN                       → residual            ]

    The layer returns only a delta (gated message); the residual add and
    the single LayerNorm (pre-norm on h) both live here, not inside the layer.
    """

    def __init__(
        self,
        model_dim: int,
        reduced_dim: int = 32,
        ff_dim: Optional[int] = None,
        n_edge_feats: int = N_EDGE_FEATS,
        dropout: float = 0.1,
    ):
        super().__init__()
        ff_dim = ff_dim or 4 * model_dim

        self.ln1 = nn.LayerNorm(model_dim)
        self.plucker_mix = StructureEdgePluckerLayer(
            model_dim=model_dim,
            reduced_dim=reduced_dim,
            n_edge_feats=n_edge_feats,
        )
        self.drop = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, model_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor,
        edge_attrs: torch.Tensor,
    ) -> torch.Tensor:
        # Pre-norm → layer returns delta → dropout → residual add
        delta = self.plucker_mix(self.ln1(h), edge_index, edge_mask, edge_attrs)
        h = h + self.drop(delta)
        h = h + self.ffn(self.ln2(h))
        return h


# ─── Full model ───────────────────────────────────────────────────────────────

class RNAStructureGrassmann(nn.Module):
    """
    Structure-edge Plücker model for 5'UTR expression prediction.

    Supports two tasks:
        'regression':     scalar output, MSE loss  (MRL / TE / EL / RLU)
        'classification': scalar logit, BCE loss   (IRES detection)

    Optional library ID conditioning (for MRL cross-library training):
        Set num_libraries > 0 to add a learned per-library bias at the
        pooled representation level before the head.

    Architecture:
        token emb + pos emb [+ lib emb] → dropout
        → N × StructureGrassmannBlock
        → final LayerNorm
        → attention pooling (or mean pooling) over L positions
        [+ library embedding added to pooled vector]
        → 2-layer MLP → scalar logit
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_seq_len: int = 256,
        model_dim: int = 128,
        num_layers: int = 4,
        reduced_dim: int = 32,
        ff_dim: Optional[int] = None,
        n_edge_feats: int = N_EDGE_FEATS,
        dropout: float = 0.1,
        pooling: str = 'attention',  # 'attention' | 'mean'
        task: str = 'regression',   # 'regression' | 'classification'
        num_libraries: int = 0,     # >0 → add learned per-library bias
        # ── Auxiliary structure supervision (UTR-LM style) ──────────────────
        aux_struct: bool = False,   # add SS + MFE prediction heads
        lambda_ss: float = 0.1,    # weight for per-token SS cross-entropy
        lambda_mfe: float = 0.01,  # weight for MFE scalar regression
    ):
        super().__init__()
        self.model_dim = model_dim
        self.pooling = pooling
        self.task = task
        self.aux_struct = aux_struct
        self.lambda_ss  = lambda_ss
        self.lambda_mfe = lambda_mfe

        # ── Embeddings ────────────────────────────────────────────────────────
        self.token_emb = nn.Embedding(vocab_size, model_dim, padding_idx=PAD_ID)
        self.pos_emb   = nn.Embedding(max_seq_len, model_dim)
        self.emb_drop  = nn.Dropout(dropout)

        # ── Structure-aware Plücker mixing blocks ─────────────────────────────
        self.blocks = nn.ModuleList([
            StructureGrassmannBlock(
                model_dim=model_dim,
                reduced_dim=reduced_dim,
                ff_dim=ff_dim,
                n_edge_feats=n_edge_feats,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(model_dim)

        # ── Pooling ───────────────────────────────────────────────────────────
        if pooling == 'attention':
            self.pool_attn = nn.Linear(model_dim, 1)

        # ── Optional library ID conditioning ──────────────────────────────────
        # Added to pooled vector before the head; shifts predictions per library
        # without touching the sequence encoder.
        self.lib_emb: Optional[nn.Embedding] = (
            nn.Embedding(num_libraries, model_dim) if num_libraries > 0 else None
        )

        # ── Prediction head (shared for regression and classification) ─────────
        # For classification, the output logit is passed to BCE loss externally;
        # sigmoid is NOT applied here so the model works with BCEWithLogitsLoss.
        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
        )

        # ── Auxiliary structure heads (UTR-LM style) ───────────────────────────
        # Both are None when aux_struct=False so they add zero parameters by
        # default and do not affect existing checkpoints.
        #
        # ss_head:  per-token 3-class classifier (.=0, (=1, )=2) applied to the
        #           full sequence of hidden states h from encode().  This mirrors
        #           UTR-LM's masked-token SS prediction, except here we predict
        #           every position rather than only masked ones.
        #
        # mfe_head: scalar regression head applied to the pooled representation
        #           *before* the library-bias is added, because MFE is a
        #           sequence-intrinsic quantity (library-independent).
        if aux_struct:
            self.ss_head  = nn.Linear(model_dim, N_SS_CLASSES)
            self.mfe_head = nn.Linear(model_dim, 1)
        else:
            self.ss_head  = None
            self.mfe_head = None

        # Global init first (touches every nn.Linear, Embedding, LayerNorm)
        self.apply(self._init_weights)
        # Then re-zero W_attn in every layer so all edges start with equal
        # weight (uniform attention).  Must run *after* apply, otherwise apply
        # would override it.
        for block in self.blocks:
            nn.init.zeros_(block.plucker_mix.W_attn.weight)
            nn.init.zeros_(block.plucker_mix.W_attn.bias)

    # ── Weight initialisation (same policy as grassmann_v4) ───────────────────
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    # ── Encoding pass ─────────────────────────────────────────────────────────
    def encode(
        self,
        input_ids:  torch.Tensor,   # (B, L)
        edge_index: torch.Tensor,   # (B, L, K)
        edge_mask:  torch.Tensor,   # (B, L, K)
        edge_attrs: torch.Tensor,   # (B, L, K, n_edge_feats)
    ) -> torch.Tensor:              # (B, L, d)
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        h = self.emb_drop(self.token_emb(input_ids) + self.pos_emb(pos))
        for block in self.blocks:
            h = block(h, edge_index, edge_mask, edge_attrs)
        return self.ln_f(h)

    # ── Sequence → single vector ───────────────────────────────────────────────
    def pool(
        self,
        h: torch.Tensor,         # (B, L, d)
        seq_mask: torch.Tensor,  # (B, L) bool
    ) -> torch.Tensor:           # (B, d)
        if self.pooling == 'attention':
            logits = self.pool_attn(h).squeeze(-1)           # (B, L)
            logits = logits.masked_fill(~seq_mask, torch.finfo(logits.dtype).min / 2)
            weights = torch.softmax(logits, dim=-1)          # (B, L)
            return (weights.unsqueeze(-1) * h).sum(dim=1)   # (B, d)
        else:  # mean pooling over real tokens
            h = h.masked_fill(~seq_mask.unsqueeze(-1), 0.0)
            return h.sum(dim=1) / seq_mask.float().sum(dim=1, keepdim=True)

    # ── Full forward pass ─────────────────────────────────────────────────────
    def forward(
        self,
        input_ids:   torch.Tensor,                    # (B, L)
        edge_index:  torch.Tensor,                    # (B, L, K)
        edge_mask:   torch.Tensor,                    # (B, L, K)
        edge_attrs:  torch.Tensor,                    # (B, L, K, n_edge_feats)
        seq_mask:    torch.Tensor,                    # (B, L) bool
        labels:      Optional[torch.Tensor] = None,   # (B,) float
        library_ids: Optional[torch.Tensor] = None,   # (B,) long
        ss_labels:   Optional[torch.Tensor] = None,   # (B, L) long — SS class IDs
        mfe_labels:  Optional[torch.Tensor] = None,   # (B,) float — MFE kcal/mol
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids:    token IDs
            edge_index:   neighbour indices (-1 = padding)
            edge_mask:    valid-edge mask
            edge_attrs:   per-edge features [bp_prob, norm_dist, is_struct]
            seq_mask:     real-token mask for pooling
            labels:       regression targets or binary class labels (0/1)
            library_ids:  integer library condition indices for lib_emb
            ss_labels:    per-token SS class IDs (0=., 1=(, 2=)); pad with -100
            mfe_labels:   MFE targets (kcal/mol) — sequence-level float

        Returns:
            logits: (B,) raw scalar predictions (no sigmoid for classification)
            loss:   combined multi-task loss (primary + λ_ss·SS + λ_mfe·MFE),
                    or None if no labels are provided
        """
        h = self.encode(input_ids, edge_index, edge_mask, edge_attrs)  # (B, L, d)
        pooled = self.pool(h, seq_mask)                                 # (B, d)

        loss = None

        # ── Auxiliary MFE head (sequence-level, before library bias) ──────────
        # MFE is a property of the sequence alone, so we apply it before the
        # library-specific bias is added to the pooled vector.
        if self.aux_struct and mfe_labels is not None:
            mfe_pred = self.mfe_head(pooled).squeeze(-1)               # (B,)
            loss = self.lambda_mfe * F.mse_loss(mfe_pred, mfe_labels.float())

        # Add per-library bias at the representation level (primary task only)
        if self.lib_emb is not None and library_ids is not None:
            pooled = pooled + self.lib_emb(library_ids)                # (B, d)

        logits = self.head(pooled).squeeze(-1)                         # (B,)

        # ── Primary task loss ─────────────────────────────────────────────────
        if labels is not None:
            if self.task == 'classification':
                primary_loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            else:
                primary_loss = F.mse_loss(logits, labels.float())
            loss = primary_loss if loss is None else loss + primary_loss

        # ── Auxiliary SS head (per-token cross-entropy) ───────────────────────
        if self.aux_struct and ss_labels is not None:
            ss_logits = self.ss_head(h)                                # (B, L, 3)
            ss_loss = F.cross_entropy(
                ss_logits.reshape(-1, N_SS_CLASSES),
                ss_labels.reshape(-1).long(),
                ignore_index=SS_IGNORE_IDX,
            )
            aux = self.lambda_ss * ss_loss
            loss = aux if loss is None else loss + aux

        return logits, loss

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Dataset ──────────────────────────────────────────────────────────────────

class UTRDataset(torch.utils.data.Dataset):
    """
    Dataset for 5'UTR expression prediction.

    Args:
        sequences:       list of nucleotide strings
        labels:          list/array of float targets (e.g. log-CFPS yield),
                         or None for inference
        bpps:            list of pre-computed (L, L) BPP matrices;
                         if None, BPP is computed via compute_bpp() per sequence
        local_offsets:   sequence offsets for local edges
        top_k_struct:    max structure edges per node
        bp_threshold:    minimum BPP to include a structure edge
    """

    def __init__(
        self,
        sequences: List[str],
        labels: Optional[List[float]] = None,
        bpps: Optional[List[np.ndarray]] = None,
        local_offsets: Tuple[int, ...] = (-2, -1, 1, 2),
        top_k_struct: int = 4,
        bp_threshold: float = 0.05,
    ):
        self.samples: List[Dict[str, Any]] = []
        for i, seq in enumerate(sequences):
            bpp = bpps[i] if bpps is not None else compute_bpp(seq)
            sample = preprocess_sample(
                seq, bpp, local_offsets, top_k_struct, bp_threshold
            )
            if labels is not None:
                sample['label'] = float(labels[i])
            self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


# ─── Training utilities ───────────────────────────────────────────────────────

def train_epoch(
    model: RNAStructureGrassmann,
    loader: torch.utils.data.DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    clip_grad: float = 1.0,
) -> float:
    """Run one training epoch; return mean MSE loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimiser.zero_grad()
        _, loss = model(
            input_ids  = batch['input_ids'],
            edge_index = batch['edge_index'],
            edge_mask  = batch['edge_mask'],
            edge_attrs = batch['edge_attrs'],
            seq_mask   = batch['seq_mask'],
            labels     = batch['labels'],
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: RNAStructureGrassmann,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate; return MSE and Pearson r."""
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        preds, _ = model(
            input_ids  = batch['input_ids'],
            edge_index = batch['edge_index'],
            edge_mask  = batch['edge_mask'],
            edge_attrs = batch['edge_attrs'],
            seq_mask   = batch['seq_mask'],
        )
        all_preds.append(preds.cpu())
        all_labels.append(batch['labels'].cpu())

    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    mse = F.mse_loss(preds, labels).item()

    # Pearson correlation
    vp = preds  - preds.mean()
    vl = labels - labels.mean()
    pearson_r = (vp * vl).sum() / (
        vp.norm() * vl.norm() + 1e-8
    )
    return {'mse': mse, 'pearson_r': pearson_r.item()}


# ─── Minimal smoke-test / usage example ──────────────────────────────────────

if __name__ == '__main__':
    import functools

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Synthetic data ────────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    seqs = [
        ''.join(rng.choice(['A', 'C', 'G', 'U'], size=rng.integers(30, 80)))
        for _ in range(128)
    ]
    labels = rng.standard_normal(128).tolist()

    # BPP: pass zeros to skip ViennaRNA (structure edges disabled for this test)
    bpps = [np.zeros((len(s), len(s)), dtype=np.float32) for s in seqs]

    dataset = UTRDataset(seqs, labels=labels, bpps=bpps)
    loader  = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_rna,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = RNAStructureGrassmann(
        model_dim=64,
        num_layers=2,
        reduced_dim=16,   # r=16 → plucker_dim=120 (memory-light for demo)
        dropout=0.1,
        pooling='attention',
    ).to(device)

    print(f'Parameters: {model.get_num_params():,}')

    # ── Training ──────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    for epoch in range(1, 4):
        loss = train_epoch(model, loader, opt, device)
        metrics = evaluate(model, loader, device)
        print(
            f'Epoch {epoch} | train_mse={loss:.4f} '
            f'| val_mse={metrics["mse"]:.4f} '
            f'| pearson_r={metrics["pearson_r"]:.4f}'
        )

    # ── Single-sample inference (no labels, no BPP) ───────────────────────────
    test_seq = 'AUGCAUGCAUGCAUGCAUGCAUGCAUGC'
    test_bpp = compute_bpp(test_seq)        # uses ViennaRNA or zero fallback
    sample   = preprocess_sample(test_seq, test_bpp)
    batch    = collate_rna([sample])
    batch    = {k: v.to(device) for k, v in batch.items()}

    model.eval()
    with torch.no_grad():
        pred, _ = model(**batch)
    print(f'\nPredicted expression for test sequence: {pred.item():.4f}')
