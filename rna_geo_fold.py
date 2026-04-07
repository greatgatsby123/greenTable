"""
rna_geo_fold.py — Hybrid RNA Secondary Structure Prediction Model

Architecture overview
---------------------
GeoFoldNet fuses two complementary representations at every scale of a
U-Net-style encoder–decoder:

  1. 2-D CNN branch (UFold-inspired)
     Takes an L×L pair-feature map (9 channels: row/col one-hot + distance)
     and processes it through a multi-scale encoder, a bottleneck, and a
     symmetric decoder with skip connections.

  2. Geometry branch (RNA-Bender-inspired)
     Maintains a sequence-level node tensor H (B, L, d) and a pairwise
     graph tensor G (B, p, L, L).  H is updated with depth-wise 1-D
     convolutions + FFN; G is updated with 2-D convolutions conditioned on
     outer-product projections of H.

The two branches are coupled at every encoder stage through GeoFusionBlock
(bidirectional, gated with learned scalars initialised to 0) and at every
decoder stage through GeoCNNFuse (unidirectional G→F, also zero-gated).
Zero-initialisation of all cross-branch projections ensures the model starts
as a pure CNN and opens the geometry path gradually during training.

Memory: O(L²) due to the pair tensor G.  Practical limit ~1024 nt on a
typical GPU with pair_dim=16 and cnn_channels=[32,64,128,256].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from rna_bender import VOCAB_SIZE, PAD_ID, _sinusoidal_pe

N_NUC = 4        # A, C, G, U/T
N_PAIR_FEAT = 9  # CNN input channels


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def build_pair_features(input_ids: torch.Tensor) -> torch.Tensor:
    """Return (B, 9, L, L) pair-feature tensor from integer token ids (B, L)."""
    B, L = input_ids.shape
    ids_clamped = input_ids.clamp(0, N_NUC - 1)
    one_hot = F.one_hot(ids_clamped, num_classes=N_NUC).float()  # (B, L, 4)
    pad_mask = (input_ids >= N_NUC).unsqueeze(-1).float()        # (B, L, 1)
    one_hot = one_hot * (1.0 - pad_mask)

    oh = one_hot.permute(0, 2, 1)                                # (B, 4, L)
    oh_i = oh.unsqueeze(-1).expand(B, N_NUC, L, L)              # row nuc
    oh_j = oh.unsqueeze(-2).expand(B, N_NUC, L, L)              # col nuc

    idx = torch.arange(L, device=input_ids.device).float()
    denom = max(L - 1, 1)
    dist = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() / denom   # (L, L)
    dist = dist.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L)

    return torch.cat([oh_i, oh_j, dist], dim=1)                  # (B, 9, L, L)


# ---------------------------------------------------------------------------
# Basic CNN building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Sequential):
    """Double Conv2d + BN + ReLU."""
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__(
            nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )


class UpConv(nn.Sequential):
    """Nearest 2× upsample + Conv2d + BN + ReLU."""
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(ch_in, ch_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )


# ---------------------------------------------------------------------------
# Geometry branch blocks
# ---------------------------------------------------------------------------

class PairLiftBlock(nn.Module):
    """Projects node features H → pairwise tensor G via outer-product + distance embedding."""
    def __init__(self, node_dim: int, pair_dim: int, max_dist: int = 128):
        super().__init__()
        self.proj_l = nn.Linear(node_dim, pair_dim, bias=False)
        self.proj_r = nn.Linear(node_dim, pair_dim, bias=False)
        self.dist_emb = nn.Embedding(max_dist + 1, pair_dim)
        self.norm = nn.LayerNorm(pair_dim)
        self.max_dist = max_dist

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        B, L, _ = H.shape
        gl = self.proj_l(H)   # (B, L, p)
        gr = self.proj_r(H)   # (B, L, p)
        G = gl.unsqueeze(2) + gr.unsqueeze(1)   # (B, L, L, p)

        idx = torch.arange(L, device=H.device)
        diffs = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs().clamp(0, self.max_dist)
        G = G + self.dist_emb(diffs)             # broadcast (L, L, p)
        G = self.norm(G)
        return G.permute(0, 3, 1, 2).contiguous()  # (B, p, L, L)


class GeometrySeqBlock(nn.Module):
    """Updates node tensor H with depth-wise 1-D conv + FFN residuals."""
    def __init__(self, node_dim: int, kernel: int = 7):
        super().__init__()
        self.norm1 = nn.LayerNorm(node_dim)
        self.dw = nn.Conv1d(
            node_dim, node_dim, kernel,
            padding=kernel // 2, groups=node_dim, bias=False,
        )
        self.norm2 = nn.LayerNorm(node_dim)
        self.ffn = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.GELU(),
            nn.Linear(node_dim * 2, node_dim),
        )

    def forward(
        self,
        H: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.norm1(H).permute(0, 2, 1)          # (B, d, L)
        h = self.dw(h).permute(0, 2, 1)             # (B, L, d)
        H = H + h
        H = H + self.ffn(self.norm2(H))
        if seq_mask is not None:
            H = H * seq_mask.unsqueeze(-1).float()
        return H


class GeometryPairBlock(nn.Module):
    """Updates pairwise tensor G conditioned on node tensor H."""
    def __init__(self, node_dim: int, pair_dim: int):
        super().__init__()
        self.proj_l = nn.Linear(node_dim, pair_dim, bias=False)
        self.proj_r = nn.Linear(node_dim, pair_dim, bias=False)
        self.conv = nn.Conv2d(pair_dim, pair_dim, 3, padding=1, bias=False)
        self.gn = nn.GroupNorm(1, pair_dim)
        self.act = nn.GELU()

    def forward(self, G: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        delta = (
            self.proj_l(H).unsqueeze(2) + self.proj_r(H).unsqueeze(1)
        ).permute(0, 3, 1, 2)                       # (B, p, L, L)
        return G + self.act(self.gn(self.conv(G) + delta))


# ---------------------------------------------------------------------------
# Fusion blocks
# ---------------------------------------------------------------------------

class GeoFusionBlock(nn.Module):
    """Bidirectional F↔G fusion for the encoder (zero-gated)."""
    def __init__(self, cnn_ch: int, pair_dim: int):
        super().__init__()
        self.g_to_f = nn.Conv2d(pair_dim, cnn_ch, 1, bias=False)
        self.f_to_g = nn.Conv2d(cnn_ch, pair_dim, 1, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.g_to_f.weight)
        nn.init.zeros_(self.f_to_g.weight)

    def forward(
        self, F: torch.Tensor, G: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        F = F + self.alpha * self.g_to_f(G)
        G = G + self.beta * self.f_to_g(F)
        return F, G


class GeoCNNFuse(nn.Module):
    """Unidirectional G→F fusion for the decoder (zero-gated)."""
    def __init__(self, cnn_ch: int, pair_dim: int):
        super().__init__()
        self.g_to_f = nn.Conv2d(pair_dim, cnn_ch, 1, bias=False)
        self.eta = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.g_to_f.weight)

    def forward(self, F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
        return F + self.eta * self.g_to_f(G)


# ---------------------------------------------------------------------------
# Bottleneck
# ---------------------------------------------------------------------------

class HybridBottleneck(nn.Module):
    """Deepest stage: two CNN blocks interleaved with geometry updates."""
    def __init__(self, cnn_ch: int, node_dim: int, pair_dim: int):
        super().__init__()
        self.cnn1 = ConvBlock(cnn_ch, cnn_ch)
        self.cnn2 = ConvBlock(cnn_ch, cnn_ch)
        self.geo_seq = GeometrySeqBlock(node_dim)
        self.geo_pair = GeometryPairBlock(node_dim, pair_dim)
        self.fusion1 = GeoFusionBlock(cnn_ch, pair_dim)
        self.fusion2 = GeoFusionBlock(cnn_ch, pair_dim)

    def forward(
        self,
        F: torch.Tensor,
        G: torch.Tensor,
        H: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        F = self.cnn1(F)
        H = self.geo_seq(H, seq_mask)
        F, G = self.fusion1(F, G)
        F = self.cnn2(F)
        G = self.geo_pair(G, H)
        F, G = self.fusion2(F, G)
        return F, G, H


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class GeoFoldNet(nn.Module):
    """
    Hybrid RNA secondary-structure predictor.

    Outputs
    -------
    pair_logits : (B, L, L) — raw logits for base-pair probability matrix.
    ss_logits   : (B, L, 3) — per-nucleotide secondary-structure class logits.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        node_dim: int = 64,
        pair_dim: int = 16,
        cnn_channels: Optional[List[int]] = None,
        n_scales: int = 4,
        max_pos: int = 4096,
        dropout: float = 0.1,
    ):
        super().__init__()
        if cnn_channels is None:
            cnn_channels = [32 * (2 ** i) for i in range(n_scales)]
        assert len(cnn_channels) == n_scales
        self.n_scales = n_scales
        self.node_dim = node_dim
        self.pair_dim = pair_dim

        # Geometry embedding
        self.token_emb = nn.Embedding(vocab_size, node_dim, padding_idx=PAD_ID)
        self.register_buffer('pos_enc', _sinusoidal_pe(max_pos, node_dim), persistent=False)
        self.drop = nn.Dropout(dropout)
        self.pair_lift = PairLiftBlock(node_dim, pair_dim)

        # Encoder
        self.enc_cnn = nn.ModuleList()
        self.enc_seq = nn.ModuleList()
        self.enc_pair = nn.ModuleList()
        self.enc_fuse = nn.ModuleList()
        self.f_pool = nn.ModuleList()
        self.g_pool = nn.ModuleList()
        ch_in = N_PAIR_FEAT
        for ch in cnn_channels:
            self.enc_cnn.append(ConvBlock(ch_in, ch))
            self.enc_seq.append(GeometrySeqBlock(node_dim))
            self.enc_pair.append(GeometryPairBlock(node_dim, pair_dim))
            self.enc_fuse.append(GeoFusionBlock(ch, pair_dim))
            self.f_pool.append(nn.MaxPool2d(2, 2))
            self.g_pool.append(nn.AvgPool2d(2, 2))
            ch_in = ch

        # Bottleneck
        self.bottleneck = HybridBottleneck(cnn_channels[-1], node_dim, pair_dim)

        # Decoder
        self.dec_up = nn.ModuleList()
        self.dec_conv = nn.ModuleList()
        self.dec_fuse = nn.ModuleList()
        ch_from = cnn_channels[-1]
        for i in range(n_scales):
            ch_skip = cnn_channels[n_scales - 1 - i]
            self.dec_up.append(UpConv(ch_from, ch_skip))
            self.dec_conv.append(ConvBlock(2 * ch_skip, ch_skip))
            self.dec_fuse.append(GeoCNNFuse(ch_skip, pair_dim))
            ch_from = ch_skip

        # Output head
        self.g0_proj = nn.Conv2d(pair_dim, cnn_channels[0], 1, bias=False)
        self.head = nn.Conv2d(cnn_channels[0] * 2, 1, 1)
        self.ss_head = nn.Linear(node_dim, 3)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def _pad_spatial(
        x: torch.Tensor, stride: int
    ) -> Tuple[torch.Tensor, int, int]:
        H, W = x.shape[-2], x.shape[-1]
        pH = (stride - H % stride) % stride
        pW = (stride - W % stride) % stride
        if pH > 0 or pW > 0:
            x = F.pad(x, (0, pW, 0, pH))
        return x, pH, pW

    @staticmethod
    def _crop(x: torch.Tensor, target_H: int, target_W: int) -> torch.Tensor:
        return x[..., :target_H, :target_W]

    @staticmethod
    def _pool_H(H: torch.Tensor) -> torch.Tensor:
        B, L, d = H.shape
        return F.avg_pool1d(H.permute(0, 2, 1), kernel_size=2, stride=2).permute(0, 2, 1)

    @staticmethod
    def _align_H(H: torch.Tensor, target_L: int) -> torch.Tensor:
        L = H.shape[1]
        if L >= target_L:
            return H[:, :target_L, :]
        pad = torch.zeros(H.shape[0], target_L - L, H.shape[2], device=H.device, dtype=H.dtype)
        return torch.cat([H, pad], dim=1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
        edge_idx=None,
        edge_feat=None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        B, L = input_ids.shape
        stride = 2 ** self.n_scales

        # Pair feature map
        X = build_pair_features(input_ids)                          # (B, 9, L, L)
        X, pH, pW = self._pad_spatial(X, stride)
        Lp = X.shape[2]

        # Node embeddings
        H0 = self.drop(self.token_emb(input_ids) + self.pos_enc[:L])  # (B, L, d)

        # Pre-compute downsampled H and masks for each scale
        H_scales: List[torch.Tensor] = [H0]
        mask_scales: List[Optional[torch.Tensor]] = [seq_mask]
        H_cur = H0
        mask_cur = seq_mask
        for _ in range(self.n_scales):
            H_cur = self._pool_H(H_cur)
            H_scales.append(H_cur)
            if mask_cur is not None:
                mask_cur = (
                    F.avg_pool1d(mask_cur.float().unsqueeze(1), kernel_size=2, stride=2)
                    .squeeze(1) > 0.5
                )
            mask_scales.append(mask_cur)

        # Initial pair tensor — padded to Lp × Lp
        G0 = self.pair_lift(H0)                                     # (B, p, L, L)
        if Lp > L:
            G0 = F.pad(G0, (0, Lp - L, 0, Lp - L))

        # Encoder
        F_cur = X
        G_cur = G0
        F_skips: List[torch.Tensor] = []
        G_skips: List[torch.Tensor] = []

        for i in range(self.n_scales):
            Ls = F_cur.shape[-1]
            H_s = self._align_H(H_scales[i], Ls)
            F_cur = self.enc_cnn[i](F_cur)
            H_s = self.enc_seq[i](H_s, mask_scales[i])
            G_cur = self.enc_pair[i](G_cur, H_s)
            F_cur, G_cur = self.enc_fuse[i](F_cur, G_cur)
            F_skips.append(F_cur)
            G_skips.append(G_cur)
            F_cur = self.f_pool[i](F_cur)
            G_cur = self.g_pool[i](G_cur)

        # Bottleneck
        H_bot = self._align_H(H_scales[-1], F_cur.shape[-1])
        F_cur, G_cur, _ = self.bottleneck(F_cur, G_cur, H_bot, mask_scales[-1])

        # Decoder
        for i in range(self.n_scales):
            skip_idx = self.n_scales - 1 - i
            F_skip = F_skips[skip_idx]
            G_skip = G_skips[skip_idx]

            F_cur = self.dec_up[i](F_cur)
            if F_cur.shape[-2:] != F_skip.shape[-2:]:
                F_cur = self._crop(F_cur, F_skip.shape[-2], F_skip.shape[-1])
            F_cur = torch.cat([F_skip, F_cur], dim=1)
            F_cur = self.dec_conv[i](F_cur)

            if G_skip.shape[-2:] != F_cur.shape[-2:]:
                G_skip = F.interpolate(
                    G_skip, size=F_cur.shape[-2:], mode='nearest'
                )
            F_cur = self.dec_fuse[i](F_cur, G_skip)

        # Output
        G0_skip = G_skips[0]
        if G0_skip.shape[-2:] != F_cur.shape[-2:]:
            G0_skip = F.interpolate(G0_skip, size=F_cur.shape[-2:], mode='nearest')
        joint = torch.cat([F_cur, self.g0_proj(G0_skip)], dim=1)  # (B, ch0*2, Lp, Lp)
        logits = self.head(joint).squeeze(1)    # (B, Lp, Lp)
        logits = (logits + logits.transpose(-1, -2)) / 2  # symmetrize
        pair_logits = logits[:, :L, :L]   # (B, L, L)
        ss_logits = self.ss_head(H0)  # (B, L, 3)

        return {'pair_logits': pair_logits, 'ss_logits': ss_logits}
