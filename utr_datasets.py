"""
Dataset loaders, BPP caching, and evaluation metrics for 5'UTR benchmarks.

Datasets (nihms-1998067 / Zenodo: "A 5' UTR Language Model..."):

  MRL:  Synthetic 50-nt libraries (eGFP/mCherry × 3 chemistries × 2 replicates)
        Regression: Mean Ribosome Load
        ~280 k sequences per eGFP library, ~200 k per mCherry library

  TE:   Endogenous 100-bp human 5'UTRs (3 cell lines: Muscle / PC3 / HEK)
        Regression: Translation Efficiency (TE) or Expression Level (EL)
        41,446 unique UTRs total
        Evaluation: 10-fold CV + Spearman ρ (paper protocol)

  IRES: Binary IRES detection (46,774 sequences)
        Classification: IRES (1) vs non-IRES (0)
        Evaluation: AUPR (class-imbalanced)

  RLU:  Designed luciferase 5'UTRs (N=211, regression)
        Use cross-validation + strong regularization (small-data regime)

Column name defaults match the likely Zenodo CSV format; adjust via kwargs.
"""

import os
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    from scipy.stats import spearmanr, pearsonr
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import KFold
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

from rna_structure_plucker import (
    preprocess_sample, collate_rna, compute_bpp, N_EDGE_FEATS, PAD_ID,
    compute_ss_mfe, encode_ss, MASK_ID, VOCAB_SIZE,
)


# ─── Library registry (MRL) ───────────────────────────────────────────────────

# 8 conditions × 2 replicates = 16 total in the paper; extend as needed.
LIBRARY_NAMES: List[str] = [
    'eGFP-U1',    'eGFP-U2',
    'eGFP-Ψ1',    'eGFP-Ψ2',
    'eGFP-m1Ψ1',  'eGFP-m1Ψ2',
    'mCherry-U1',  'mCherry-U2',
    'mCherry-Ψ1',  'mCherry-Ψ2',
    'mCherry-m1Ψ1','mCherry-m1Ψ2',
]
LIBRARY_TO_ID: Dict[str, int] = {n: i for i, n in enumerate(LIBRARY_NAMES)}
NUM_LIBRARIES = len(LIBRARY_NAMES)


# ─── BPP caching ─────────────────────────────────────────────────────────────

def compute_bpp_mfe(seq: str) -> np.ndarray:
    """
    Fast structure approximation via MFE fold (ViennaRNA RNA.fold).

    Returns a binary (L, L) contact matrix:  1.0 for MFE-paired positions.
    Much faster than the partition function but gives only paired/unpaired,
    not probabilities.  Suitable for large datasets (MRL ~280k sequences).
    Falls back to zeros if ViennaRNA is not installed.
    """
    L = len(seq)
    seq_rna = seq.upper().replace('T', 'U')
    try:
        import RNA
        structure, _ = RNA.fold(seq_rna)
        return _dotbracket_to_contacts(structure)
    except ImportError:
        return np.zeros((L, L), dtype=np.float32)


def _dotbracket_to_contacts(structure: str) -> np.ndarray:
    """Convert dot-bracket string to symmetric binary contact matrix."""
    L = len(structure)
    contacts = np.zeros((L, L), dtype=np.float32)
    stack: List[int] = []
    for i, c in enumerate(structure):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            j = stack.pop()
            contacts[i, j] = contacts[j, i] = 1.0
    return contacts


class BPPCache:
    """
    Disk-backed cache for BPP / contact matrices keyed by sequence MD5.

    Backends:
        'viennarna'  — full partition function (accurate, slow, ~50 ms/seq)
        'mfe'        — MFE contact binary matrix (fast, ~5 ms/seq)
        'zero'       — zeros everywhere (disables structure edges; for ablation)

    Usage:
        cache = BPPCache('bpp_cache/', backend='mfe')
        bpp   = cache.get('AUGCAUGCAUGC')   # (L, L) float32
    """

    def __init__(self, cache_dir: str, backend: str = 'mfe'):
        assert backend in ('viennarna', 'mfe', 'zero'), f'Unknown backend: {backend}'
        self.cache_dir = cache_dir
        self.backend = backend
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, seq: str) -> str:
        key = hashlib.md5(seq.upper().encode()).hexdigest()
        return os.path.join(self.cache_dir, f'{key}_{self.backend}.npy')

    def get(self, seq: str) -> np.ndarray:
        """Return cached BPP or compute and cache it."""
        if self.backend == 'zero':
            return np.zeros((len(seq), len(seq)), dtype=np.float32)
        path = self._path(seq)
        if os.path.exists(path):
            return np.load(path)
        bpp = compute_bpp_mfe(seq) if self.backend == 'mfe' else compute_bpp(seq)
        np.save(path, bpp)
        return bpp

    def _ssmfe_path(self, seq: str) -> str:
        key = hashlib.md5(seq.upper().encode()).hexdigest()
        return os.path.join(self.cache_dir, f'{key}_ssmfe.npz')

    def get_ss_mfe(self, seq: str) -> Tuple[np.ndarray, float]:
        """
        Return (ss_ids, mfe) for *seq* with disk caching.

        ss_ids is an int8 array of length L (class IDs: 0=., 1=(, 2=)).
        mfe is the MFE in kcal/mol from RNA.fold().

        Always uses RNA.fold() regardless of the BPP backend, because SS/MFE
        are independent of the partition-function BPP computation.
        Falls back to all-unpaired / 0.0 if ViennaRNA is not installed.
        """
        path = self._ssmfe_path(seq)
        if os.path.exists(path):
            data = np.load(path)
            return data['ss_ids'], float(data['mfe'])
        ss_str, mfe = compute_ss_mfe(seq)
        ss_ids = encode_ss(ss_str)
        np.savez(path, ss_ids=ss_ids, mfe=np.float32(mfe))
        return ss_ids, float(mfe)

    def warm_up(self, sequences: List[str], verbose: bool = True) -> None:
        """Pre-populate cache for a list of sequences (single-threaded)."""
        n = len(sequences)
        for i, seq in enumerate(sequences):
            self.get(seq)
            if verbose and (i + 1) % 1000 == 0:
                print(f'  BPP cache: {i + 1}/{n}', end='\r', flush=True)
        if verbose:
            print(f'  BPP cache: {n}/{n} done.')


# ─── Base dataset ─────────────────────────────────────────────────────────────

class BaseUTRDataset(Dataset):
    """
    Abstract base class for all UTR datasets.

    Subclasses set self.sequences, self.labels, and optionally
    self.library_ids before calling self._build_samples() or by setting
    lazy=True (samples computed in __getitem__).

    Lazy mode is recommended for large libraries (MRL: ~280k sequences)
    where storing all edge arrays in RAM would require several GB.
    With a BPPCache and DataLoader num_workers > 0, preprocessing is
    automatically parallelised across worker processes.
    """

    def __init__(
        self,
        local_offsets: Tuple[int, ...] = (-2, -1, 1, 2),
        top_k_struct:  int = 4,
        bp_threshold:  float = 0.05,
        bpp_cache:     Optional[BPPCache] = None,
        max_len:       Optional[int] = None,
        lazy:          bool = False,
        aux_struct:    bool = False,   # include SS class IDs and MFE in samples
    ):
        self.local_offsets = local_offsets
        self.top_k_struct  = top_k_struct
        self.bp_threshold  = bp_threshold
        self.bpp_cache     = bpp_cache
        self.max_len       = max_len
        self.lazy          = lazy
        self.aux_struct    = aux_struct

        self.sequences:    List[str]           = []
        self.labels:       List[float]         = []
        self.library_ids:  Optional[List[int]] = None
        self._samples:     List[Dict]          = []   # populated when not lazy

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_bpp(self, seq: str) -> np.ndarray:
        if self.bpp_cache is not None:
            return self.bpp_cache.get(seq)
        return compute_bpp(seq)

    def _get_ss_mfe(self, seq: str) -> Tuple[np.ndarray, float]:
        """Return (ss_ids int8 array, mfe float) for seq."""
        if self.bpp_cache is not None:
            return self.bpp_cache.get_ss_mfe(seq)
        ss_str, mfe = compute_ss_mfe(seq)
        return encode_ss(ss_str), mfe

    def _truncate(self, seq: str) -> str:
        return seq[:self.max_len] if self.max_len is not None else seq

    def _make_sample(self, idx: int) -> Dict:
        seq = self._truncate(self.sequences[idx])
        bpp = self._get_bpp(seq)
        sample = preprocess_sample(
            seq, bpp,
            self.local_offsets, self.top_k_struct, self.bp_threshold,
        )
        sample['label'] = float(self.labels[idx])
        if self.library_ids is not None:
            sample['library_id'] = int(self.library_ids[idx])
        if self.aux_struct:
            ss_ids, mfe = self._get_ss_mfe(seq)
            sample['ss_ids'] = ss_ids
            sample['mfe']    = np.float32(mfe)
        return sample

    def _build_samples(self) -> None:
        """Eagerly build all samples (suitable for small–medium datasets)."""
        self._samples = [self._make_sample(i) for i in range(len(self.sequences))]

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        if self.lazy:
            return self._make_sample(idx)
        return self._samples[idx]


# ─── MRL synthetic libraries ──────────────────────────────────────────────────

class MRLDataset(BaseUTRDataset):
    """
    Synthetic 50-nt UTR libraries with Mean Ribosome Load (MRL) labels.

    Each library (e.g. eGFP-U1, eGFP-Ψ1, mCherry-m1Ψ2) has ~200–280k
    sequences.  For cross-library training, pass a merged CSV with a
    library-name column; set ``num_libraries`` in the model accordingly.

    Expected CSV columns (rename via kwargs):
        seq_col:   UTR nucleotide sequence (50 nt)
        label_col: MRL value (float; paper recommends log-transforming)
        lib_col:   library name string (e.g. 'eGFP-U1'); None = single library

    The dataset is large → lazy=True is strongly recommended.
    BPPs should be pre-warmed with BPPCache.warm_up() or computed offline.
    """

    def __init__(
        self,
        csv_path:  str,
        seq_col:   str = 'utr',
        label_col: str = 'rl',    # actual column name in capsule-4214075-data CSVs
        lib_col:   Optional[str] = None,
        library_id: int = 0,         # used when lib_col is None
        lazy: bool = True,           # default True for large libraries
        **kwargs,
    ):
        assert _PANDAS, 'pandas is required: pip install pandas'
        super().__init__(lazy=lazy, **kwargs)
        import pandas as pd
        df = pd.read_csv(csv_path)

        self.sequences = df[seq_col].tolist()
        self.labels    = df[label_col].tolist()

        if lib_col is not None and lib_col in df.columns:
            self.library_ids = [
                LIBRARY_TO_ID.get(n, 0) for n in df[lib_col].tolist()
            ]
        else:
            self.library_ids = [library_id] * len(self.sequences)

        if not lazy:
            self._build_samples()


# ─── Endogenous TE / EL datasets ─────────────────────────────────────────────

class TEDataset(BaseUTRDataset):
    """
    Endogenous human 5'UTRs with Translation Efficiency (TE) or
    Expression Level (EL) labels.

    nihms-1998067: 41,446 unique UTRs × 3 cell lines (Muscle, PC3, HEK).
    Paper uses 100-bp fixed length; set max_len=100 (default) to match.
    Evaluation: 10-fold CV + Spearman ρ (use train_utr.py --folds 10).

    Expected CSV columns:
        seq_col:   UTR sequence
        label_col: 'te' or 'el'
        cell_col:  cell line string ('Muscle' / 'PC3' / 'HEK'); or None
    """

    def __init__(
        self,
        csv_path:    str,
        seq_col:     str = 'utr',
        label_col:   str = 'te',
        cell_col:    Optional[str] = 'cell_line',
        cell_filter: Optional[str] = None,
        **kwargs,
    ):
        assert _PANDAS, 'pandas is required: pip install pandas'
        # Enforce 100-bp cap unless caller overrides
        kwargs.setdefault('max_len', 100)
        super().__init__(**kwargs)
        import pandas as pd
        df = pd.read_csv(csv_path)

        if cell_filter is not None and cell_col is not None and cell_col in df.columns:
            df = df[df[cell_col] == cell_filter].reset_index(drop=True)

        self.sequences = df[seq_col].tolist()
        self.labels    = df[label_col].tolist()

        if not self.lazy:
            self._build_samples()


# ─── IRES classification dataset ─────────────────────────────────────────────

class IRESDataset(BaseUTRDataset):
    """
    Binary IRES detection (46,774 sequences; class-imbalanced).

    Task: 'classification' in RNAStructureGrassmann.
    Loss: BCEWithLogitsLoss.
    Metric: AUPR (more informative than AUROC for imbalanced data).

    Expected CSV columns:
        seq_col:   RNA/DNA sequence
        label_col: 0 (non-IRES) or 1 (IRES)
    """

    def __init__(
        self,
        csv_path:  str,
        seq_col:   str = 'sequence',
        label_col: str = 'label',
        **kwargs,
    ):
        assert _PANDAS, 'pandas is required: pip install pandas'
        super().__init__(**kwargs)
        import pandas as pd
        df = pd.read_csv(csv_path)

        self.sequences = df[seq_col].tolist()
        self.labels    = df[label_col].astype(float).tolist()

        if not self.lazy:
            self._build_samples()


# ─── Luciferase design dataset ────────────────────────────────────────────────

class RLUDataset(BaseUTRDataset):
    """
    Designed 5'UTRs with luciferase (RLU) activity (N=211, regression).

    Small dataset: use heavy regularisation and cross-validation.
    Always uses eager loading (lazy=False) since N=211 fits trivially.

    Expected CSV columns:
        seq_col:   UTR sequence
        label_col: RLU value (log-transform recommended)
    """

    def __init__(
        self,
        csv_path:  str,
        seq_col:   str = 'utr_originial_varylength',  # actual col in Experimental_data_revised_label.csv
        label_col: str = 'label',                     # normalised expression (0.48–1.33)
        **kwargs,
    ):
        assert _PANDAS, 'pandas is required: pip install pandas'
        kwargs['lazy'] = False   # always eager for 211 samples
        super().__init__(**kwargs)
        import pandas as pd
        df = pd.read_csv(csv_path)

        self.sequences = df[seq_col].tolist()
        self.labels    = df[label_col].tolist()
        self._build_samples()


# ─── Collation ────────────────────────────────────────────────────────────────

def collate_utr(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Extended collate_rna that also handles ``library_id`` per sample.

    Use this as the DataLoader's collate_fn for all UTR datasets.
    Delegates padding/stacking to collate_rna then appends library_ids.
    """
    batch = collate_rna(samples)
    if 'library_id' in samples[0]:
        batch['library_ids'] = torch.tensor(
            [s['library_id'] for s in samples], dtype=torch.long
        )
    return batch


# ─── Evaluation metrics ───────────────────────────────────────────────────────

def spearman_r(preds: np.ndarray, labels: np.ndarray) -> float:
    """Spearman rank correlation (primary metric for TE datasets)."""
    assert _SCIPY, 'scipy is required: pip install scipy'
    rho, _ = spearmanr(preds, labels)
    return float(rho)


def pearson_r(preds: np.ndarray, labels: np.ndarray) -> float:
    assert _SCIPY, 'scipy is required: pip install scipy'
    r, _ = pearsonr(preds, labels)
    return float(r)


def r_squared(preds: np.ndarray, labels: np.ndarray) -> float:
    ss_res = np.sum((labels - preds) ** 2)
    ss_tot = np.sum((labels - labels.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def mse(preds: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((preds - labels) ** 2))


def aupr(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Area Under Precision-Recall curve.

    Primary metric for IRES (class-imbalanced binary classification).
    ``probs`` should be probabilities in [0, 1] (apply sigmoid to logits first).
    """
    assert _SKLEARN, 'scikit-learn is required: pip install scikit-learn'
    return float(average_precision_score(labels.astype(int), probs))


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    task: str = 'regression',
) -> Dict[str, float]:
    """
    Compute the full set of task-appropriate metrics.

    Args:
        preds:  raw model logits (B,)
        labels: ground-truth values (B,)
        task:   'regression' or 'classification'

    Returns:
        dict of metric_name → float
    """
    if task == 'classification':
        probs = 1.0 / (1.0 + np.exp(-preds))   # sigmoid
        return {'aupr': aupr(probs, labels)}
    else:
        return {
            'mse':       mse(preds, labels),
            'r2':        r_squared(preds, labels),
            'pearson_r': pearson_r(preds, labels),
            'spearman_r': spearman_r(preds, labels),
        }


# ─── Cross-validation utilities ───────────────────────────────────────────────

def kfold_indices(
    n: int,
    k: int = 10,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Return k (train_idx, val_idx) pairs for k-fold cross-validation.

    Matches the paper's protocol for TE datasets (10-fold CV).
    """
    assert _SKLEARN, 'scikit-learn is required: pip install scikit-learn'
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in kf.split(range(n))]


def stratified_kfold_indices(
    labels: np.ndarray,
    k: int = 10,
    seed: int = 42,
    n_bins: int = 10,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Stratified k-fold for regression: bin labels, then stratify.

    Useful for MRL and TE datasets to ensure each fold has a
    representative label distribution.
    """
    assert _SKLEARN, 'scikit-learn is required: pip install scikit-learn'
    from sklearn.model_selection import StratifiedKFold
    bins = np.digitize(labels, np.quantile(labels, np.linspace(0, 1, n_bins + 1)[1:-1]))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(range(len(labels)), bins)]


# ─── Pretraining dataset ──────────────────────────────────────────────────────

class PretrainDataset(Dataset):
    """
    Unlabeled sequence dataset for self-supervised pretraining.

    Reads one or more (csv_path, seq_col) sources, deduplicates sequences,
    and for each sample produces:

        input_ids   -- nucleotide tokens with ~mlm_prob fraction randomly
                       masked using the BERT 80/10/10 rule
        mlm_labels  -- original token IDs at masked positions; -100 elsewhere
        edge_index, edge_mask, edge_attrs, seq_mask
                    -- same graph tensors as BaseUTRDataset
        ss_ids      -- per-token dot-bracket class IDs (if aux_struct=True)
        mfe         -- scalar MFE float32 (if aux_struct=True)

    MLM masking rule (per selected token):
        80 %  replaced with MASK_ID (= 'N', id 4)
        10 %  replaced with a random valid nucleotide (A/C/G/U, ids 0-3)
        10 %  left unchanged  (but still contribute to the MLM loss)

    Sources can be the same CSVs as downstream tasks -- only the sequence
    column is read; all label columns are ignored.

    Use ``exclude_sources`` to pass held-out test CSVs whose sequences must
    NOT appear in the pretraining corpus.  This is essential for a clean
    evaluation: even self-supervised pretraining gives the model a head-start
    on sequences it has already processed.
    """

    def __init__(
        self,
        sources:         List[Tuple[str, str]],         # [(csv_path, seq_col), ...]
        bpp_cache:       'BPPCache',
        max_len:         Optional[int] = None,
        top_k_struct:    int   = 4,
        bp_threshold:    float = 0.05,
        mlm_prob:        float = 0.15,
        aux_struct:      bool  = True,
        deduplicate:     bool  = True,
        rng_seed:        Optional[int] = None,
        exclude_sources: Optional[List[Tuple[str, str]]] = None,
        # [(csv_path, seq_col), ...] of test / val CSVs whose sequences must
        # be excluded from the pretraining corpus for a clean evaluation.
    ):
        assert _PANDAS, 'pandas is required: pip install pandas'

        self.bpp_cache    = bpp_cache
        self.max_len      = max_len
        self.top_k_struct = top_k_struct
        self.bp_threshold = bp_threshold
        self.mlm_prob     = mlm_prob
        self.aux_struct   = aux_struct
        self.rng          = np.random.default_rng(rng_seed)

        # Build exclusion set from held-out CSVs (truncated to same max_len)
        excluded: set = set()
        for csv_path, seq_col in (exclude_sources or []):
            df = pd.read_csv(csv_path)
            for s in df[seq_col].astype(str).str.upper().str.strip():
                excluded.add(s[:max_len] if max_len else s)

        # Collect and optionally deduplicate sequences across all sources
        raw: List[str] = []
        for csv_path, seq_col in sources:
            df = pd.read_csv(csv_path)
            raw.extend(df[seq_col].astype(str).str.upper().str.strip().tolist())

        if max_len:
            raw = [s[:max_len] for s in raw]

        if deduplicate:
            seen: set = set()
            self.sequences: List[str] = []
            for s in raw:
                if s not in seen and s not in excluded:
                    seen.add(s)
                    self.sequences.append(s)
        else:
            self.sequences = [s for s in raw if s not in excluded]

        if excluded:
            n_before = len(raw)
            n_after  = len(self.sequences)
            print(f'  PretrainDataset: excluded {n_before - n_after} sequences '
                  f'that appeared in held-out test/val sources '
                  f'({n_after} remaining)')

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict:
        seq = self.sequences[idx]

        # Build graph (same as BaseUTRDataset)
        bpp    = self.bpp_cache.get(seq)
        sample = preprocess_sample(
            seq, bpp,
            top_k_struct = self.top_k_struct,
            bp_threshold = self.bp_threshold,
        )

        # Apply MLM masking (BERT 80/10/10)
        original_ids = np.array(sample['input_ids'], dtype=np.int64)
        L            = len(original_ids)
        mlm_labels   = np.full(L, -100, dtype=np.int64)

        real_pos = original_ids != PAD_ID
        selected = (self.rng.random(L) < self.mlm_prob) & real_pos

        if selected.any():
            mlm_labels[selected] = original_ids[selected]
            masked_ids = original_ids.copy()
            decision   = self.rng.random(L)
            # 80 % -> MASK_ID (N)
            replace  = selected & (decision < 0.8)
            masked_ids[replace] = MASK_ID
            # 10 % -> random valid nucleotide (A=0, C=1, G=2, U=3)
            rand_rep = selected & (decision >= 0.8) & (decision < 0.9)
            masked_ids[rand_rep] = self.rng.integers(0, 4, size=int(rand_rep.sum()))
            # 10 % -> unchanged (contribute to loss via mlm_labels)
            sample['input_ids'] = masked_ids.astype(np.int32)

        sample['mlm_labels'] = mlm_labels

        if self.aux_struct:
            ss_ids, mfe      = self.bpp_cache.get_ss_mfe(seq)
            sample['ss_ids'] = ss_ids
            sample['mfe']    = np.float32(mfe)

        return sample


def collate_pretrain(samples: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate pretrain samples.

    Extends collate_rna (which handles input_ids, edge tensors, seq_mask,
    and optionally ss_ids / mfe) with mlm_labels padding.
    """
    batch = collate_rna(samples)   # handles ss_ids/mfe if present

    # mlm_labels: pad to L_max with -100
    if 'mlm_labels' in samples[0]:
        L_max = batch['input_ids'].shape[1]
        mlm   = np.full((len(samples), L_max), -100, dtype=np.int64)
        for i, s in enumerate(samples):
            lb = s['mlm_labels']
            mlm[i, :len(lb)] = lb
        batch['mlm_labels'] = torch.tensor(mlm)

    return batch
