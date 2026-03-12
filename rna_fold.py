"""
RNA Secondary Structure Prediction: Dataset, Evaluation, and Utilities.

Supports RNAstralign and similar corpora that provide:
  sequence   — nucleotide string
  structure  — dot-bracket string OR a JSON pair list [[i,j], ...]
  family     — RNA family label (for GroupKFold evaluation)

Key components:
  RNAstralignDataset   — loads from CSV or a BPSEQ directory tree
  collate_rnastralign  — pads variable-length samples into batch tensors
  structure_metrics    — pair P / R / F1 / MCC and per-token SS accuracy
  family_kfold_indices — GroupKFold splits by RNA family

Structure field formats (both are handled automatically):
  Dot-bracket string:  "(((....)))"
  JSON pair list:      "[[0, 71], [1, 70], ...]"  (0-indexed, unordered)

Data format notes:
  CSV:   columns  sequence (str), structure (str), family (str)
  BPSEQ: each file one sequence; lines  "<pos> <nuc> <pair_pos>"
         expects a root directory whose sub-directories are family labels

Edge construction:
  For RNAstralign we know the ground-truth structure, so structural edges
  are built directly from the known base pairs (bp_prob = 1.0) rather than
  from ViennaRNA.  This is faster and exact.
"""

import json
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union

from rna_structure_plucker import (
    encode_sequence, build_padded_edges,
    PAD_ID, SS_VOCAB, N_EDGE_FEATS,
)

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    from sklearn.model_selection import GroupKFold
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

try:
    from sklearn.metrics import matthews_corrcoef
    _MCC_SKLEARN = True
except ImportError:
    _MCC_SKLEARN = False


# ─── Dot-bracket helpers ─────────────────────────────────────────────────────

def dotbracket_to_pairs(structure: str) -> List[Tuple[int, int]]:
    """
    Parse a dot-bracket string into a list of (i, j) base-pair tuples with i < j.
    Handles simple dot-bracket ( / ) and pseudo-knot extensions [ / ] { / }.
    """
    pairs: List[Tuple[int, int]] = []
    stacks: Dict[str, List[int]] = {'(': [], '[': [], '{': []}
    close_to_open = {')': '(', ']': '[', '}': '{'}

    for pos, ch in enumerate(structure):
        if ch in stacks:
            stacks[ch].append(pos)
        elif ch in close_to_open:
            opener = close_to_open[ch]
            if stacks[opener]:
                j = stacks[opener].pop()
                pairs.append((min(j, pos), max(j, pos)))
    return pairs


def dotbracket_to_pair_matrix(structure: str) -> np.ndarray:
    """Convert dot-bracket string to a binary symmetric (L, L) pair matrix."""
    L   = len(structure)
    mat = np.zeros((L, L), dtype=np.float32)
    for i, j in dotbracket_to_pairs(structure):
        mat[i, j] = mat[j, i] = 1.0
    return mat


def dotbracket_to_ss_labels(structure: str) -> np.ndarray:
    """Convert dot-bracket string to integer class IDs (0=., 1=(, 2=))."""
    return np.array([SS_VOCAB.get(c, 0) for c in structure], dtype=np.int64)


# ─── Unified structure field parser ───────────────────────────────────────────

def _parse_structure_field(
    struct_val: Union[str, list],
    seq_len:    int,
) -> List[Tuple[int, int]]:
    """
    Parse a structure column value into a canonical list of (i, j) pairs.

    Handles two formats automatically:
      Dot-bracket string:  "(((....)))"
      JSON pair list:      "[[0, 71], [1, 70]]"  (string or already parsed list)

    Detection logic for strings starting with '[':
      - If JSON-parseable and each element is a 2-element numeric list → pair list
      - Otherwise → dot-bracket (pseudo-knot brackets)

    Returns 0-indexed (i, j) pairs with i < j, both in [0, seq_len).
    """
    if isinstance(struct_val, (list, tuple)):
        raw_pairs = struct_val
    elif isinstance(struct_val, str):
        s = struct_val.strip()
        if not s:
            return []
        if s[0] in '(.':
            # Unambiguously dot-bracket
            return dotbracket_to_pairs(s)
        elif s[0] == '[':
            # Could be JSON pair list [[i,j],...] or dot-bracket pseudo-knot [...]
            try:
                parsed = json.loads(s)
                # It's a pair list if every element is a length-2 numeric sequence
                if (isinstance(parsed, list) and parsed
                        and isinstance(parsed[0], (list, tuple))
                        and len(parsed[0]) == 2):
                    raw_pairs = parsed
                else:
                    return dotbracket_to_pairs(s)
            except (ValueError, TypeError):
                return dotbracket_to_pairs(s)
        elif s[0] == '{':
            return dotbracket_to_pairs(s)
        else:
            return []
    else:
        return []

    result = []
    for p in raw_pairs:
        try:
            i, j = int(p[0]), int(p[1])
        except (TypeError, IndexError, ValueError):
            continue
        if 0 <= i < seq_len and 0 <= j < seq_len and i != j:
            result.append((min(i, j), max(i, j)))
    return result


def pairs_to_pair_matrix(pairs: List[Tuple[int, int]], seq_len: int) -> np.ndarray:
    """Build a binary symmetric (seq_len, seq_len) pair matrix from a pairs list."""
    mat = np.zeros((seq_len, seq_len), dtype=np.float32)
    for i, j in pairs:
        mat[i, j] = mat[j, i] = 1.0
    return mat


def pairs_to_ss_labels(pairs: List[Tuple[int, int]], seq_len: int) -> np.ndarray:
    """Per-position SS labels derived from pairs: 0=unpaired, 1=open '(', 2=close ')'."""
    labels = np.zeros(seq_len, dtype=np.int64)
    for i, j in pairs:
        labels[i] = 1   # opening bracket
        labels[j] = 2   # closing bracket
    return labels


# ─── Edge construction from known structure ───────────────────────────────────

def build_edges_from_pairs(
    pairs:         List[Tuple[int, int]],
    seq_len:       int,
    local_offsets: Tuple[int, ...] = (-2, -1, 1, 2),
    top_k_struct:  int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build padded edge tensors from a canonical pairs list.

    Works with both dot-bracket-derived and pair-list-derived structures.

    Returns same format as build_padded_edges:
        edge_index : (L, K_max) int32
        edge_mask  : (L, K_max) bool
        edge_attrs : (L, K_max, 3) float32
    """
    bpp = pairs_to_pair_matrix(pairs, seq_len)
    return build_padded_edges(
        seq_len,
        bpp           = bpp,
        local_offsets = local_offsets,
        top_k_struct  = top_k_struct,
        bp_threshold  = 0.5,
    )


def build_edges_from_structure(
    structure:     str,
    local_offsets: Tuple[int, ...] = (-2, -1, 1, 2),
    top_k_struct:  int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build padded edge tensors from a dot-bracket string (backward compat)."""
    pairs = dotbracket_to_pairs(structure)
    return build_edges_from_pairs(pairs, len(structure), local_offsets, top_k_struct)


# ─── BPSEQ parser ─────────────────────────────────────────────────────────────

def parse_bpseq(path: str) -> Tuple[str, str]:
    """
    Parse a BPSEQ file.  Returns (sequence, dot_bracket).

    BPSEQ format (one residue per line):
        <1-indexed_position>  <nucleotide>  <pair_position (0 if unpaired)>
    Lines starting with # are comments.
    """
    positions: List[Tuple[int, str, int]] = []
    with open(path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            pos, nuc, pair = int(parts[0]), parts[1].upper(), int(parts[2])
            positions.append((pos, nuc, pair))

    if not positions:
        return '', ''

    seq = ''.join(r[1] for r in positions).replace('T', 'U')

    L      = len(positions)
    db     = ['.'] * L
    paired = {r[0]: r[2] for r in positions if r[2] != 0}

    for zero_i, (pos, _, pair) in enumerate(positions):
        if pair != 0 and pair > pos:   # open bracket
            db[zero_i] = '('
        elif pair != 0 and pair < pos: # close bracket
            db[zero_i] = ')'

    return seq, ''.join(db)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class RNAstralignDataset(Dataset):
    """
    Dataset for RNA secondary structure prediction.

    Three loading modes:
      'csv'   — CSV file with columns sequence, structure, family
      'json'  — JSON file: dict of {id: {sequence, structure, family}}
      'bpseq' — root directory; each sub-directory is a family; files are BPSEQ

    Each sample returns a dict with:
      input_ids    : (L,)     int64    — encoded nucleotides
      edge_idx     : (L, K)   int64    — graph neighbour indices
      edge_feat    : (L, K, 3) float32 — [bp_prob, norm_dist, is_struct]
      seq_mask     : (L,)     bool     — True = valid position
      pair_targets : (L, L)   float32  — binary base-pair matrix
      ss_labels    : (L,)     int64    — per-token {. ( )} class IDs
      family       : str               — RNA family label
    """

    def __init__(
        self,
        data_path:              str,
        data_format:            str   = 'csv',   # 'csv' or 'bpseq'
        seq_col:                str   = 'sequence',
        struct_col:             str   = 'structure',
        family_col:             str   = 'family',
        max_len:                Optional[int] = None,
        top_k_struct:           int   = 4,
        use_oracle_struct_edges: bool  = True,
    ):
        """
        Args:
            use_oracle_struct_edges: if True (default), graph edges include the
                ground-truth base pairs (bp_prob=1.0).  Set to False for the
                sequence-only ablation where the model must infer structure
                without access to the answer in the input graph.
        """
        assert data_format in ('csv', 'json', 'bpseq'), \
            f'Unknown data_format: {data_format!r}. Use "csv", "json", or "bpseq".'

        self.max_len                = max_len
        self.top_k_struct           = top_k_struct
        self.use_oracle_struct_edges = use_oracle_struct_edges

        if data_format == 'csv':
            self._load_csv(data_path, seq_col, struct_col, family_col)
        elif data_format == 'json':
            self._load_json(data_path, seq_col, struct_col, family_col)
        else:
            self._load_bpseq(data_path)

    # ── Loading ────────────────────────────────────────────────────────────────

    def _load_csv(self, path, seq_col, struct_col, family_col):
        assert _PANDAS, 'pandas is required for CSV loading: pip install pandas'
        df = pd.read_csv(path)
        assert seq_col in df.columns,    f'Column {seq_col!r} not in {list(df.columns)}'
        assert struct_col in df.columns, f'Column {struct_col!r} not in {list(df.columns)}'

        self.sequences  = df[seq_col].tolist()
        self.structures = df[struct_col].tolist()
        self.families   = df[family_col].tolist() if family_col in df.columns \
                          else ['unknown'] * len(df)

    def _load_json(self, path, seq_col, struct_col, family_col):
        """Load from a JSON dict: {id: {seq_col: ..., struct_col: ..., family_col: ...}}."""
        with open(path, 'r') as fh:
            data = json.load(fh)
        # Support both {id: record} dicts and plain lists of records
        records = data.values() if isinstance(data, dict) else data
        self.sequences, self.structures, self.families = [], [], []
        for rec in records:
            seq = rec.get(seq_col, '')
            if not seq:
                continue
            self.sequences.append(str(seq).replace('T', 'U'))
            self.structures.append(rec.get(struct_col, []))
            self.families.append(str(rec.get(family_col, 'unknown')))

    def _load_bpseq(self, root):
        self.sequences, self.structures, self.families = [], [], []
        for family in sorted(os.listdir(root)):
            fam_dir = os.path.join(root, family)
            if not os.path.isdir(fam_dir):
                continue
            for fname in sorted(os.listdir(fam_dir)):
                if not fname.endswith('.bpseq'):
                    continue
                fpath = os.path.join(fam_dir, fname)
                seq, struct = parse_bpseq(fpath)
                if seq:
                    self.sequences.append(seq)
                    self.structures.append(struct)
                    self.families.append(family)

    # ── Item ──────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        seq     = self.sequences[idx]
        struct  = self.structures[idx]
        family  = self.families[idx]

        # Truncate sequence (structure handled below after parsing)
        if self.max_len is not None and len(seq) > self.max_len:
            seq = seq[:self.max_len]

        L = len(seq)

        # Parse structure: handles both dot-bracket strings and JSON pair lists.
        # Filter to pairs within [0, L) to respect any truncation.
        pairs = _parse_structure_field(struct, seq_len=L)

        input_ids   = torch.tensor(encode_sequence(seq),           dtype=torch.long)
        ss_labels   = torch.tensor(pairs_to_ss_labels(pairs, L),   dtype=torch.long)
        pair_target = torch.tensor(pairs_to_pair_matrix(pairs, L), dtype=torch.float32)

        if self.use_oracle_struct_edges:
            # Oracle: use ground-truth pairs as structural edges (bp_prob = 1.0)
            edge_idx_np, _, edge_feat_np = build_edges_from_pairs(
                pairs, L, top_k_struct=self.top_k_struct
            )
        else:
            # Sequence-only: local/backbone edges only — no ground-truth leakage
            zero_bpp = np.zeros((L, L), dtype=np.float32)
            edge_idx_np, _, edge_feat_np = build_padded_edges(
                L, bpp=zero_bpp, top_k_struct=0, bp_threshold=0.5,
            )
        edge_idx  = torch.tensor(edge_idx_np, dtype=torch.long)
        edge_feat = torch.tensor(edge_feat_np, dtype=torch.float32)
        seq_mask  = torch.ones(L, dtype=torch.bool)

        return {
            'input_ids':    input_ids,
            'edge_idx':     edge_idx,
            'edge_feat':    edge_feat,
            'seq_mask':     seq_mask,
            'pair_targets': pair_target,
            'ss_labels':    ss_labels,
            'family':       family,
        }


# ─── Collation ────────────────────────────────────────────────────────────────

def collate_rnastralign(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate variable-length RNA structure samples into batch tensors.

    Pads to the maximum length in the batch:
      input_ids    : (B, L_max)           — padded with PAD_ID
      edge_idx     : (B, L_max, K_max)    — padded with -1
      edge_feat    : (B, L_max, K_max, 3) — padded with 0
      seq_mask     : (B, L_max)           — True = valid
      pair_targets : (B, L_max, L_max)    — padded with 0
      ss_labels    : (B, L_max)           — padded with -100 (ignored by CE)
    """
    L_max = max(s['input_ids'].shape[0] for s in batch)
    K_max = max(s['edge_idx'].shape[1]  for s in batch)
    B     = len(batch)

    input_ids_b    = torch.full((B, L_max),               PAD_ID,  dtype=torch.long)
    edge_idx_b     = torch.full((B, L_max, K_max),        -1,      dtype=torch.long)
    edge_feat_b    = torch.zeros(B, L_max, K_max, N_EDGE_FEATS,    dtype=torch.float32)
    seq_mask_b     = torch.zeros(B, L_max,                         dtype=torch.bool)
    pair_targets_b = torch.zeros(B, L_max, L_max,                  dtype=torch.float32)
    ss_labels_b    = torch.full((B, L_max),        -100,            dtype=torch.long)
    families       = []

    for i, s in enumerate(batch):
        L = s['input_ids'].shape[0]
        K = s['edge_idx'].shape[1]

        input_ids_b[i, :L]              = s['input_ids']
        edge_idx_b[i, :L, :K]          = s['edge_idx']
        edge_feat_b[i, :L, :K]         = s['edge_feat']
        seq_mask_b[i, :L]              = s['seq_mask']
        pair_targets_b[i, :L, :L]      = s['pair_targets']
        ss_labels_b[i, :L]             = s['ss_labels']
        families.append(s['family'])

    return {
        'input_ids':    input_ids_b,
        'edge_idx':     edge_idx_b,
        'edge_feat':    edge_feat_b,
        'seq_mask':     seq_mask_b,
        'pair_targets': pair_targets_b,
        'ss_labels':    ss_labels_b,
        'families':     families,
    }


# ─── Structure evaluation metrics ─────────────────────────────────────────────

def structure_metrics(
    pair_logits:  np.ndarray,   # (L, L) raw logits
    pair_targets: np.ndarray,   # (L, L) binary ground truth
    ss_logits:    Optional[np.ndarray] = None,   # (L, 3)
    ss_labels:    Optional[np.ndarray] = None,   # (L,)  -100 = ignore
    seq_len:      Optional[int] = None,
    threshold:    float = 0.5,
) -> Dict[str, float]:
    """
    Compute pair prediction and per-token SS metrics for one sequence.

    Pair metrics (P, R, F1, MCC) are evaluated on the upper triangle
    of the valid square region to avoid double-counting symmetric pairs.

    Args:
        pair_logits  : (L, L) raw model logits (before sigmoid)
        pair_targets : (L, L) binary float ground truth
        ss_logits    : (L, 3) per-token logits for {. ( )} (optional)
        ss_labels    : (L,) per-token class IDs (optional)
        seq_len      : actual sequence length (use to mask padding)
        threshold    : sigmoid threshold for pair prediction (default 0.5)
    """
    L = seq_len if seq_len is not None else pair_logits.shape[0]

    # Upper triangle mask (i < j) over valid region — avoids symmetry double-count
    triu = np.triu(np.ones((L, L), dtype=bool), k=1)

    pred_prob = 1.0 / (1.0 + np.exp(-pair_logits[:L, :L]))   # sigmoid
    pred_bin  = (pred_prob > threshold).astype(int)
    tgt_bin   = (pair_targets[:L, :L] > 0.5).astype(int)

    pred_flat = pred_bin[triu]
    tgt_flat  = tgt_bin[triu]

    tp = int((pred_flat & tgt_flat).sum())
    fp = int((pred_flat & ~tgt_flat.astype(bool)).sum())
    fn = int((~pred_flat.astype(bool) & tgt_flat).sum())
    tn = int((~pred_flat.astype(bool) & ~tgt_flat.astype(bool)).sum())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
    mcc   = (tp * tn - fp * fn) / denom

    out: Dict[str, float] = {
        'pair_precision': precision,
        'pair_recall':    recall,
        'pair_f1':        f1,
        'pair_mcc':       mcc,
    }

    if ss_logits is not None and ss_labels is not None:
        valid = (ss_labels[:L] >= 0)
        if valid.any():
            pred_ss = ss_logits[:L][valid].argmax(axis=-1)
            true_ss = ss_labels[:L][valid]
            ss_acc  = float((pred_ss == true_ss).mean())
            out['ss_accuracy'] = ss_acc
        else:
            out['ss_accuracy'] = 0.0

    return out


def aggregate_structure_metrics(per_seq: List[Dict[str, float]]) -> Dict[str, float]:
    """Average per-sequence structure metrics across a dataset."""
    if not per_seq:
        return {}
    keys = per_seq[0].keys()
    return {k: float(np.mean([m[k] for m in per_seq])) for k in keys}


# ─── Family-aware fold splitting ──────────────────────────────────────────────

def family_kfold_indices(
    families: List[str],
    k:        int  = 10,
    seed:     int  = 42,   # unused — GroupKFold is deterministic by family ordering
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    GroupKFold splits that keep each RNA family entirely in train or val.

    Prevents data leakage between homologous sequences from the same family.

    Note: GroupKFold is deterministic given the input ordering; `seed` is
    accepted for API compatibility but has no effect on the splits.

    Args:
        families : list of family label strings (one per sample)
        k        : number of folds
        seed     : ignored (GroupKFold is deterministic)

    Returns:
        list of (train_idx, val_idx) numpy arrays
    """
    assert _SKLEARN, 'scikit-learn required for family_kfold_indices'
    n       = len(families)
    X_dummy = np.arange(n).reshape(-1, 1)
    y_dummy = np.zeros(n)
    groups  = np.array(families)

    gkf     = GroupKFold(n_splits=k)
    return [
        (tr, va)
        for tr, va in gkf.split(X_dummy, y_dummy, groups)
    ]


def random_family_split(
    families: List[str],
    val_frac: float = 0.2,
    seed:     int   = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single train/val split that keeps families intact (no family spans both sets).

    Randomly assigns entire families to train or val until val_frac is reached.
    """
    rng          = np.random.RandomState(seed)
    unique_fams  = list(set(families))
    rng.shuffle(unique_fams)

    n            = len(families)
    target_val   = int(n * val_frac)
    fam_sizes    = {f: families.count(f) for f in unique_fams}

    val_fams: set = set()
    val_count = 0
    for fam in unique_fams:
        if val_count >= target_val:
            break
        val_fams.add(fam)
        val_count += fam_sizes[fam]

    families_arr = np.array(families)
    val_idx  = np.where(np.isin(families_arr, list(val_fams)))[0]
    train_idx = np.where(~np.isin(families_arr, list(val_fams)))[0]
    return train_idx, val_idx


# ─── Loss function for folding task ──────────────────────────────────────────

def folding_loss(
    outputs:       Dict[str, torch.Tensor],
    pair_targets:  torch.Tensor,   # (B, L, L) binary
    ss_labels:     Optional[torch.Tensor] = None,   # (B, L) int, -100=ignore
    seq_mask:      Optional[torch.Tensor] = None,   # (B, L) bool
    lambda_pair:   float = 1.0,
    lambda_ss:     float = 0.1,
    lambda_curv:   float = 0.01,
    lambda_cons:   float = 0.01,
) -> torch.Tensor:
    """
    Compute total folding loss from a model outputs dict.

    Designed to be used as the `compute_loss_fn` argument to train_epoch.

    Loss terms:
        L_pair  — BCE on pair-map logits vs ground-truth base pairs
        L_ss    — cross-entropy on per-token SS class (if ss_logits present)
        L_curv  — curvature regularisation (if kappa_list present)
        L_cons  — backbone–pairing consistency (if p_bb1_list + p_struct_list)
    """
    import torch.nn.functional as F

    loss = pair_targets.new_zeros(())   # scalar 0 on correct device, no spurious leaf

    # ── Pair map loss ────────────────────────────────────────────────────────
    if 'pair_logits' in outputs and lambda_pair > 0:
        pair_logits = outputs['pair_logits']   # (B, L', L')  may be padded differently
        L = min(pair_logits.shape[1], pair_targets.shape[1])

        if seq_mask is not None:
            pm = seq_mask[:, :L].unsqueeze(2) & seq_mask[:, :L].unsqueeze(1)
        else:
            pm = torch.ones(pair_logits.shape[0], L, L,
                            dtype=torch.bool, device=pair_logits.device)

        # Upper-triangle only — avoids double-counting symmetric pairs and diagonal
        tri = torch.triu(torch.ones(L, L, dtype=torch.bool, device=pair_logits.device),
                         diagonal=1)
        pm  = pm & tri.unsqueeze(0)

        loss_pair = F.binary_cross_entropy_with_logits(
            pair_logits[:, :L, :L][pm],
            pair_targets[:, :L, :L][pm],
        )
        loss = loss + lambda_pair * loss_pair

    # ── SS loss ──────────────────────────────────────────────────────────────
    if 'ss_logits' in outputs and ss_labels is not None and lambda_ss > 0:
        ss_logits = outputs['ss_logits']   # (B, L, 3)
        loss_ss = F.cross_entropy(
            ss_logits.reshape(-1, 3),
            ss_labels.reshape(-1).long(),
            ignore_index=-100,
        )
        loss = loss + lambda_ss * loss_ss

    # ── Curvature regularisation ─────────────────────────────────────────────
    if 'kappa_list' in outputs and lambda_curv > 0:
        kappa_list = outputs['kappa_list']
        if kappa_list:
            if seq_mask is not None:
                # Exclude boundary positions: curvature needs both neighbours.
                # Use explicit slicing (not roll — roll wraps around sequence ends).
                interior = torch.zeros_like(seq_mask)
                interior[:, 1:-1] = (seq_mask[:, 1:-1]
                                     & seq_mask[:, :-2]
                                     & seq_mask[:, 2:])
                mf = interior.float()
            else:
                mf = torch.ones(kappa_list[0].shape[:2], device=pair_targets.device)
            n_valid = mf.sum().clamp(min=1)
            loss_curv = sum(
                (k.pow(2).sum(-1) * mf).sum() / n_valid
                for k in kappa_list
            ) / len(kappa_list)
            loss = loss + lambda_curv * loss_curv

    # ── Backbone–pairing consistency ─────────────────────────────────────────
    if ('p_bb1_list' in outputs and 'p_struct_list' in outputs
            and lambda_cons > 0):
        from rna_bender import _consistency_loss
        p_bb1   = outputs['p_bb1_list'][-1]
        p_struct = outputs['p_struct_list'][-1]
        edge_feat = outputs.get('edge_feat')
        if edge_feat is not None:
            loss = loss + lambda_cons * _consistency_loss(p_bb1, p_struct, edge_feat)

    return loss