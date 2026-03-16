"""
Standard Transformer Encoder Baseline for RNA Structure Prediction.

Provides a non-geometric baseline to compare against RNABenderModel.
Same forward-dict API, same output heads (pair map, SS), but replaces
Grassmann / Plücker geometry with standard multi-head self-attention.

Edge features are ignored — this is a pure sequence model, making it the
correct control for the sequence-only experimental setting.

Usage (same as RNABenderModel):
    model = RNATransformerBaseline(model_dim=128, num_layers=4, num_heads=8)
    out   = model(input_ids, edge_idx, edge_feat, seq_mask)
    loss  = folding_loss(out, pair_targets, ss_labels, seq_mask)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from rna_bender import (
    NUC_VOCAB, VOCAB_SIZE, PAD_ID, SS_IGNORE_IDX,
    PairMapHead,
)


class RNATransformerBaseline(nn.Module):
    """
    Standard Transformer encoder baseline for RNA secondary structure prediction.

    Architecture:
        token embedding + positional embedding
        → N × TransformerEncoderLayer  (pre-LN, batch_first)
        → LayerNorm
        → pair map head  (same bilinear as RNABenderModel)
        → SS head        (per-token 3-class)
        → optional pooled task head (skipped for task='folding')

    Edge inputs (edge_idx, edge_feat) are accepted for API compatibility
    but are not used — this is intentionally a sequence-only model.

    The output dict uses the same keys as RNABenderModel:
        pair_logits   : (B, L, L)  if use_pair_head
        ss_logits     : (B, L, 3)  if aux_struct
        task_logits   : (B,)       if task != 'folding'
        kappa_list    : []         (always empty — no geometry)
        p_bb1_list    : []         (always empty)
        p_struct_list : []         (always empty)
        edge_feat     : forwarded unchanged

    This makes folding_loss() in rna_fold.py work without modification:
    curvature and consistency terms evaluate to 0.0 because the lists are empty.
    """

    def __init__(
        self,
        vocab_size:    int            = VOCAB_SIZE,
        max_len:       int            = 256,
        model_dim:     int            = 128,
        num_layers:    int            = 4,
        num_heads:     int            = 8,
        ff_dim:        Optional[int]  = None,
        dropout:       float          = 0.1,
        pooling:       str            = 'attention',   # 'attention' | 'mean'
        task:          str            = 'folding',     # 'folding' | 'regression' | 'classification'
        num_libraries: int            = 0,
        aux_struct:    bool           = True,          # build SS head (primary for folding)
        use_pair_head: bool           = True,
    ):
        super().__init__()
        if ff_dim is None:
            ff_dim = 4 * model_dim

        self.model_dim     = model_dim
        self.task          = task
        self.aux_struct    = aux_struct
        self.use_pair_head = use_pair_head

        # ── Embeddings ──────────────────────────────────────────────────────
        self.token_emb = nn.Embedding(vocab_size, model_dim, padding_idx=PAD_ID)
        self.pos_emb   = nn.Embedding(max_len, model_dim)

        # ── Transformer encoder ─────────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = model_dim,
            nhead           = num_heads,
            dim_feedforward = ff_dim,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,   # pre-LN, matches RNABenderLayer style
        )
        # enable_nested_tensor requires norm_first=False; disable to silence warning
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.ln_f    = nn.LayerNorm(model_dim)

        # ── Pooling ─────────────────────────────────────────────────────────
        if pooling == 'attention':
            self.pool_attn: Optional[nn.Linear] = nn.Linear(model_dim, 1)
        else:
            self.pool_attn = None

        # ── Library conditioning ─────────────────────────────────────────────
        if num_libraries > 0:
            self.lib_emb: Optional[nn.Embedding] = nn.Embedding(num_libraries, model_dim)
        else:
            self.lib_emb = None

        # ── Primary task head (not built for folding) ─────────────────────
        if task != 'folding':
            self.task_head: Optional[nn.Sequential] = nn.Sequential(
                nn.Linear(model_dim, model_dim // 2),
                nn.GELU(),
                nn.Linear(model_dim // 2, 1),
            )
        else:
            self.task_head = None

        # ── Auxiliary SS head ────────────────────────────────────────────────
        if aux_struct:
            self.ss_head: Optional[nn.Linear] = nn.Linear(model_dim, 3)
        else:
            self.ss_head = None

        # ── Pair map head ────────────────────────────────────────────────────
        if use_pair_head:
            self.pair_head: Optional[PairMapHead] = PairMapHead(model_dim)
        else:
            self.pair_head = None

        self.drop = nn.Dropout(dropout)
        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

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

    # ── Pooling ────────────────────────────────────────────────────────────────

    def _pool(self, h: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        """Attention-weighted or mean pooling → (B, d)."""
        if self.pool_attn is not None:
            scores  = self.pool_attn(h).squeeze(-1)       # (B, L)
            scores  = scores.masked_fill(~seq_mask, -1e4)
            weights = torch.softmax(scores, dim=-1)
            weights = torch.nan_to_num(weights, nan=0.0)
            return (weights.unsqueeze(-1) * h).sum(dim=1)
        else:
            mf = seq_mask.float()
            return (h * mf.unsqueeze(-1)).sum(1) / mf.sum(1, keepdim=True).clamp(min=1)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:   torch.Tensor,                    # (B, L) int64
        seq_mask:    torch.Tensor,                    # (B, L) bool  True = valid
        # Folding-collate names (rnastralign)
        edge_idx:    Optional[torch.Tensor] = None,   # ignored — sequence-only model
        edge_feat:   Optional[torch.Tensor] = None,   # ignored
        # UTR-collate aliases (collate_rna) — accepted so both datasets work
        edge_index:  Optional[torch.Tensor] = None,   # ignored
        edge_attrs:  Optional[torch.Tensor] = None,   # ignored
        edge_mask:   Optional[torch.Tensor] = None,   # ignored
        labels:      Optional[torch.Tensor] = None,   # unused (folding loss is external)
        library_ids: Optional[torch.Tensor] = None,
        ss_labels:   Optional[torch.Tensor] = None,
        mfe_labels:  Optional[torch.Tensor] = None,
    ) -> Dict[str, object]:
        """
        Returns the same dict schema as RNABenderModel.forward():
            pair_logits   : (B, L, L)  if use_pair_head
            ss_logits     : (B, L, 3)  if aux_struct
            task_logits   : (B,)       if task != 'folding'
            kappa_list    : []
            p_bb1_list    : []
            p_struct_list : []
            edge_feat     : None (no edge inputs used)

        All edge arguments (edge_idx/edge_feat/edge_index/edge_attrs/edge_mask)
        are accepted for API compatibility but ignored — pure sequence model.
        """
        B, L = input_ids.shape
        pos  = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
        h    = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))  # (B, L, d)

        # TransformerEncoder: src_key_padding_mask is True where token is PADDING
        pad_mask = ~seq_mask   # (B, L)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.ln_f(h)

        out: Dict[str, object] = {
            # Empty geometry lists → curvature / consistency losses = 0 automatically
            'kappa_list':    [],
            'p_bb1_list':    [],
            'p_struct_list': [],
            'edge_feat':     edge_feat if edge_feat is not None else edge_attrs,
        }

        if self.aux_struct and self.ss_head is not None:
            out['ss_logits'] = self.ss_head(h)               # (B, L, 3)

        if self.use_pair_head and self.pair_head is not None:
            pair_logits, _ = self.pair_head(h, seq_mask)
            out['pair_logits'] = pair_logits                 # (B, L, L)

        if self.task_head is not None:
            pooled = self._pool(h, seq_mask)
            if self.lib_emb is not None and library_ids is not None:
                pooled = pooled + self.lib_emb(library_ids)
            out['task_logits'] = self.task_head(pooled).squeeze(-1)   # (B,)

            # Built-in primary loss — mirrors RNABenderModel behaviour so that
            # train_epoch's old-API path (no compute_loss_fn) works unchanged.
            if labels is not None:
                if self.task == 'classification':
                    out['loss'] = F.binary_cross_entropy_with_logits(
                        out['task_logits'], labels.float()
                    )
                else:
                    out['loss'] = F.mse_loss(out['task_logits'], labels.float())

        return out


__all__ = ['RNATransformerBaseline']