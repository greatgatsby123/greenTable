"""
RNAMoEMRLModel — Mixture-of-Experts for UTR expression prediction.

Two-branch architecture:
    Branch A (seq)  : standard transformer encoder
    Branch B (geom) : Grassmann / Bender encoder  (optionally pretrained)

A learned gate decides per-example mixture:
    α = sigmoid( MLP( [s ∥ g] ) )      scalar (B,1) or vector (B,d)
    h = α · s + (1 − α) · g

This directly tests whether a pretrained geometry expert helps MRL when
the model is allowed to use it selectively.

Experiment matrix:
    seq-only              → RNATransformerBaseline (single branch, no MoE)
    seq + scratch-geom    → RNAMoEMRLModel(pretrained_geom_encoder=None)
    seq + pretrained-geom → RNAMoEMRLModel + load_pretrained_geom(path)
    + freeze/thaw         → freeze_geom_epochs > 0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from rna_encoders import RNASequenceEncoder, RNABenderEncoder
from rna_bender import VOCAB_SIZE, BACKBONE_OFFSETS


class RNAMoEMRLModel(nn.Module):
    """
    Mixture-of-experts UTR → MRL regression model.

    Gate modes
    ----------
    'scalar' : α ∈ (0, 1)  — shared scalar per sequence; easier to interpret
    'vector' : α ∈ (0, 1)^d — per-dimension soft-selection; more expressive

    Initialisation
    --------------
    gate_bias = 0.0  → mixture starts at α ≈ 0.5 (balanced)
    gate_bias > 0    → starts trusting sequence branch more  (α → 1)
    gate_bias < 0    → starts trusting geometry branch more  (α → 0)

    Freezing schedule
    -----------------
    Call model.freeze_geom_encoder() before training starts, then
    model.unfreeze_geom_encoder() at the desired epoch.  The training loop
    in train_utr.py handles this automatically when freeze_geom_epochs > 0.

    Differential LR
    ---------------
    model.get_optimizer_groups(base_lr, geom_lr_scale)  returns two groups:
        non-geometry params  →  base_lr
        geometry params      →  base_lr × geom_lr_scale
    Use this to prevent full overwriting of the pretrained signal.
    """

    def __init__(
        self,
        model_dim:        int             = 128,
        # Sequence branch
        seq_num_layers:   int             = 4,
        seq_num_heads:    int             = 8,
        seq_ff_dim:       Optional[int]   = None,
        # Geometry branch
        geom_num_layers:  int             = 4,
        geom_reduced_dim: int             = 32,
        geom_ff_dim:      Optional[int]   = None,
        geom_max_len:     Optional[int]   = None,   # if None, uses shared max_len
        # Shared
        vocab_size:       int             = VOCAB_SIZE,
        max_len:          int             = 256,
        dropout:          float           = 0.1,
        pooling:          str             = 'attention',
        num_libraries:    int             = 0,
        # Gate
        gate_type:        str             = 'scalar',
        gate_bias:        float           = 0.0,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.gate_type = gate_type

        # ── Branch A: sequence ────────────────────────────────────────────────
        self.seq_encoder = RNASequenceEncoder(
            vocab_size = vocab_size,
            max_len    = max_len,
            model_dim  = model_dim,
            num_layers = seq_num_layers,
            num_heads  = seq_num_heads,
            ff_dim     = seq_ff_dim,
            dropout    = dropout,
            pooling    = pooling,
        )

        # ── Branch B: geometry ────────────────────────────────────────────────
        # geom_max_len lets the geom branch use a different positional range
        # than the seq branch — needed when loading a checkpoint pretrained on
        # longer sequences (e.g. RNAstralign max_len=256) for a short-seq task
        # (e.g. UTR max_len=50).
        _geom_max_len = geom_max_len if geom_max_len is not None else max_len
        self.geom_encoder = RNABenderEncoder(
            vocab_size  = vocab_size,
            max_len     = _geom_max_len,
            model_dim   = model_dim,
            num_layers  = geom_num_layers,
            reduced_dim = geom_reduced_dim,
            ff_dim      = geom_ff_dim,
            dropout     = dropout,
            pooling     = pooling,
        )

        # ── Fusion gate ────────────────────────────────────────────────────────
        gate_out = 1 if gate_type == 'scalar' else model_dim
        self.gate = nn.Sequential(
            nn.Linear(model_dim * 2, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, gate_out),
        )
        if gate_bias != 0.0:
            with torch.no_grad():
                self.gate[-1].bias.fill_(gate_bias)

        # ── Library conditioning (MRL cross-library) ─────────────────────────
        self.lib_emb: Optional[nn.Embedding] = (
            nn.Embedding(num_libraries, model_dim) if num_libraries > 0 else None
        )

        # ── MRL regression head ───────────────────────────────────────────────
        self.mrl_head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, 1),
        )

        self.drop = nn.Dropout(dropout)
        self._init_new_params()

    def _init_new_params(self):
        """Initialise only gate and head (encoders handle their own init)."""
        for m in list(self.gate) + list(self.mrl_head):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.lib_emb is not None:
            nn.init.normal_(self.lib_emb.weight, std=0.02)

    # ── Checkpoint utilities ──────────────────────────────────────────────────

    def load_pretrained_geom(self, path: str, strict: bool = False) -> Tuple:
        """
        Load pretrained geometry encoder weights.

        Accepted checkpoint formats:
            'geom_encoder_state_dict'  — pretrain_bender.py output
            'encoder_state_dict'       — pretrain_utr.py legacy output
            plain state dict           — direct state_dict file

        Returns (missing_keys, unexpected_keys).
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
        return missing, unexpected

    def freeze_geom_encoder(self):
        for p in self.geom_encoder.parameters():
            p.requires_grad = False

    def unfreeze_geom_encoder(self):
        for p in self.geom_encoder.parameters():
            p.requires_grad = True

    def get_optimizer_groups(self, base_lr: float, geom_lr_scale: float = 0.1) -> List[Dict]:
        """
        Two-group parameter split:
            non-geometry  →  base_lr
            geom_encoder  →  base_lr × geom_lr_scale

        Use for fine-tuning a pretrained geometry branch to slow forgetting.
        """
        geom_ids     = {id(p) for p in self.geom_encoder.parameters()}
        other_params = [p for p in self.parameters() if id(p) not in geom_ids]
        geom_params  = list(self.geom_encoder.parameters())
        return [
            {'params': other_params, 'lr': base_lr},
            {'params': geom_params,  'lr': base_lr * geom_lr_scale,
             'name': 'geom_encoder'},
        ]

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Forward ────────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:   torch.Tensor,                    # (B, L)
        seq_mask:    torch.Tensor,                    # (B, L) bool
        # Folding-collate names (rnastralign)
        edge_idx:    Optional[torch.Tensor] = None,   # (B, L, K)
        edge_feat:   Optional[torch.Tensor] = None,   # (B, L, K, E)
        # UTR-collate aliases (collate_rna)
        edge_index:  Optional[torch.Tensor] = None,
        edge_attrs:  Optional[torch.Tensor] = None,
        labels:      Optional[torch.Tensor] = None,   # (B,) for built-in MSE loss
        library_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict:
        """
        Returns a dict with:
            task_logits  : (B,)        predicted expression (before any activation)
            gate         : (B, 1) or (B, d)  gate activations (for analysis)
            loss         : scalar MSE (only when labels is not None)
            kappa_list, p_bb1_list, p_struct_list, edge_feat   (geometry internals)
        """
        # Normalise edge tensor names: accept both collate_rna and collate_rnastralign keys
        edge_idx  = edge_idx  if edge_idx  is not None else edge_index
        edge_feat = edge_feat if edge_feat is not None else edge_attrs

        # Branch A
        _, seq_pool, _        = self.seq_encoder.encode(input_ids, seq_mask)

        # Branch B
        _, geom_pool, geom_aux = self.geom_encoder.encode(
            input_ids, edge_idx, edge_feat, seq_mask
        )

        # Gate — alpha is the SEQUENCE weight: alpha→1 means "use seq", alpha→0 means "use geom"
        # Logged as gate_mean/gate_std in evaluate(); watch for collapse toward 0 or 1.
        alpha  = torch.sigmoid(
            self.gate(torch.cat([seq_pool, geom_pool], dim=-1))
        )                                            # (B, 1) or (B, d)
        fused  = alpha * seq_pool + (1.0 - alpha) * geom_pool
        fused  = self.drop(fused)

        # Optional library shift
        if self.lib_emb is not None and library_ids is not None:
            fused = fused + self.lib_emb(library_ids)

        pred = self.mrl_head(fused).squeeze(-1)     # (B,)

        out: Dict = {
            'task_logits':   pred,
            'gate':          alpha,
            'kappa_list':    geom_aux.get('kappa_list',    []),
            'p_bb1_list':    geom_aux.get('p_bb1_list',    []),
            'p_struct_list': geom_aux.get('p_struct_list', []),
            'edge_feat':     edge_feat,
        }

        if labels is not None:
            out['loss'] = F.mse_loss(pred, labels.float())

        return out


__all__ = ['RNAMoEMRLModel']