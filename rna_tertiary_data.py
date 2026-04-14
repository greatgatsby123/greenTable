"""
RNA 3D Tertiary Structure Dataset  (rna_tertiary_data.py)

Source: RNA3DB  —  single-chain mmCIF files, structurally and sequentially
        nonredundant train/test splits.
        Archive: rna3db-mmcifs.tar.xz  (hierarchical train/test folders)

Representation (3 atoms / nucleotide):
    0  P      phosphate
    1  C4'    sugar carbon (frame origin)
    2  N_gly  glycosidic nitrogen  (N9 for A/G purines, N1 for C/U/T pyrimidines)

Each cached sample:
    {
        "tokens":      torch.LongTensor  [L]        A/C/G/U/N vocab
        "coords_true": torch.FloatTensor [L, 3, 3]  [P, C4', N_gly]
        "id":          str               pdbid_chain
        "length":      int               L
    }

Usage:
    ds = RNA3DTertiaryDataset("path/to/rna3db/train", max_len=512,
                              cache_dir="outputs/rna3d_cache")
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_rna3d)
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# ─── Vocabulary ────────────────────────────────────────────────────────────────

NUC_VOCAB: Dict[str, int] = {
    'A': 0, 'C': 1, 'G': 2, 'U': 3,
    'T': 3, 'N': 4, '<PAD>': 5,
}
PAD_ID = NUC_VOCAB['<PAD>']

# Purine  → N9, Pyrimidine → N1
_GLYCOSIDIC_ATOM: Dict[str, str] = {
    'A': 'N9', 'G': 'N9',          # purines
    'C': 'N1', 'U': 'N1', 'T': 'N1',  # pyrimidines
}

# Modified-residue → canonical parent (common cases)
_MODIFIED_MAP: Dict[str, str] = {
    # methylated
    'M2G': 'G', '1MG': 'G', '2MG': 'G', 'OMG': 'G', '7MG': 'G',
    '1MA': 'A', 'MIA': 'A',
    'OMC': 'C', '5MC': 'C', '3MC': 'C',
    'PSU': 'U', '5MU': 'U', '4SU': 'U', 'H2U': 'U', 'UR3': 'U',
    # dihydro / other
    'DHU': 'U', 'CBV': 'C', 'A2M': 'A', 'APC': 'A',
}

# ─── mmCIF parsing ─────────────────────────────────────────────────────────────

def _parse_mmcif_chain(cif_path: Path) -> Optional[Dict]:
    """
    Parse a single-chain mmCIF file (RNA3DB format) into a 3-bead tensor.

    Tries gemmi first, falls back to biopython MMCIFParser.
    Returns None if parsing fails or the chain is too short.

    Returns:
        {
            "tokens":      np.ndarray [L]      int64
            "coords_true": np.ndarray [L,3,3]  float32  [[P,C4',N_gly]]
            "id":          str
            "length":      int
        }
    """
    try:
        return _parse_with_gemmi(cif_path)
    except ImportError:
        pass
    try:
        return _parse_with_biopython(cif_path)
    except ImportError:
        pass
    except Exception as e:
        log.warning(f'Biopython parse failed for {cif_path}: {e}')
    try:
        return _parse_with_pure_python(cif_path)
    except Exception as e:
        log.warning(f'Failed to parse {cif_path}: {e}')
        return None


def _parse_with_gemmi(cif_path: Path) -> Optional[Dict]:
    import gemmi  # type: ignore

    st = gemmi.read_structure(str(cif_path))
    if len(st) == 0:
        return None
    model = st[0]
    if len(model) == 0:
        return None
    chain = model[0]   # first (and usually only) chain in RNA3DB files

    tokens_list: List[int] = []
    coords_list: List[np.ndarray] = []

    for res in chain:
        res_name = res.name.strip().upper()

        # Map modified residues
        canonical = _MODIFIED_MAP.get(res_name, res_name)
        if canonical not in NUC_VOCAB:
            canonical = 'N'
        tok = NUC_VOCAB[canonical]

        # Determine glycosidic atom name
        ngly_name = _GLYCOSIDIC_ATOM.get(canonical)
        if ngly_name is None:
            ngly_name = 'N1'   # safe default for N

        # Extract the three atoms
        try:
            P   = _gemmi_atom_xyz(res, 'P')
            C4p = _gemmi_atom_xyz(res, "C4'")
            Ngly = _gemmi_atom_xyz(res, ngly_name)
        except KeyError:
            continue   # skip residues missing any required atom

        tokens_list.append(tok)
        coords_list.append(np.stack([P, C4p, Ngly], axis=0))   # (3,3)

    if len(tokens_list) < 4:
        return None

    chain_id = f'{st.name}_{chain.name}'
    return _pack(tokens_list, coords_list, chain_id)


def _gemmi_atom_xyz(residue, atom_name: str) -> np.ndarray:
    """Extract xyz for a named atom; raises KeyError if absent."""
    for atom in residue:
        if atom.name == atom_name:
            return np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float32)
    raise KeyError(atom_name)


def _parse_with_biopython(cif_path: Path) -> Optional[Dict]:
    from Bio.PDB import FastMMCIFParser  # type: ignore
    from Bio.PDB.PDBExceptions import PDBConstructionWarning  # type: ignore
    import warnings

    parser = FastMMCIFParser(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PDBConstructionWarning)
        structure = parser.get_structure(cif_path.stem, str(cif_path))

    model = next(iter(structure))
    chain = next(iter(model))

    tokens_list: List[int] = []
    coords_list: List[np.ndarray] = []

    for res in chain.get_residues():
        res_name = res.resname.strip().upper()
        canonical = _MODIFIED_MAP.get(res_name, res_name)
        if canonical not in NUC_VOCAB:
            canonical = 'N'
        tok = NUC_VOCAB[canonical]
        ngly_name = _GLYCOSIDIC_ATOM.get(canonical, 'N1')

        try:
            P    = res['P'].get_vector().get_array().astype(np.float32)
            C4p  = res["C4'"].get_vector().get_array().astype(np.float32)
            Ngly = res[ngly_name].get_vector().get_array().astype(np.float32)
        except KeyError:
            continue

        tokens_list.append(tok)
        coords_list.append(np.stack([P, C4p, Ngly], axis=0))

    if len(tokens_list) < 4:
        return None

    chain_id = f'{cif_path.stem}_{chain.id}'
    return _pack(tokens_list, coords_list, chain_id)


def _pack(tokens_list: List[int], coords_list: List[np.ndarray], chain_id: str) -> Dict:
    L = len(tokens_list)
    tokens      = np.array(tokens_list, dtype=np.int64)
    coords_true = np.stack(coords_list, axis=0).astype(np.float32)   # (L,3,3)
    return {
        'tokens':      tokens,
        'coords_true': coords_true,
        'id':          chain_id,
        'length':      L,
    }


# ─── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_path(cif_path: Path, cache_dir: Path) -> Path:
    h = hashlib.md5(str(cif_path.resolve()).encode()).hexdigest()[:8]
    return cache_dir / f'{cif_path.stem}_{h}.pt'


def _load_or_parse(cif_path: Path, cache_dir: Optional[Path]) -> Optional[Dict]:
    if cache_dir is not None:
        cp = _cache_path(cif_path, cache_dir)
        if cp.exists():
            try:
                return torch.load(cp, map_location='cpu', weights_only=False)
            except Exception:
                pass   # corrupt cache — re-parse

    rec = _parse_mmcif_chain(cif_path)
    if rec is None:
        return None

    # Convert to tensors for storage
    rec_t = {
        'tokens':      torch.from_numpy(rec['tokens']),
        'coords_true': torch.from_numpy(rec['coords_true']),
        'id':          rec['id'],
        'length':      rec['length'],
    }

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = _cache_path(cif_path, cache_dir)
        try:
            torch.save(rec_t, cp)
        except Exception as e:
            log.warning(f'Could not write cache {cp}: {e}')

    return rec_t


# ─── Dataset ──────────────────────────────────────────────────────────────────

class RNA3DTertiaryDataset(Dataset):
    """
    PyTorch Dataset over a folder (or tree) of single-chain mmCIF files.

    Args:
        root       : path to RNA3DB split folder (e.g. rna3db/train)
        max_len    : truncate / random-crop long chains to this length
        cache_dir  : directory for pre-parsed .pt cache files.
                     Strongly recommended; parsing mmCIF is ~10 ms/file.
        min_len    : skip chains shorter than this (default 8)
        crop_mode  : 'truncate' (take :max_len) or 'random' (random window)
        preload    : if True, parse/load all files at init (uses RAM)

    Example:
        ds = RNA3DTertiaryDataset(
            root      = "data/rna3db/train",
            max_len   = 512,
            cache_dir = "outputs/rna3d_cache",
        )
        loader = DataLoader(ds, batch_size=4, collate_fn=collate_rna3d)
    """

    def __init__(
        self,
        root:      str,
        max_len:   Optional[int] = None,
        cache_dir: Optional[str] = None,
        min_len:   int = 8,
        crop_mode: str = 'truncate',
        preload:   bool = False,
    ):
        self.root      = Path(root)
        self.max_len   = max_len
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.min_len   = min_len
        self.crop_mode = crop_mode

        # Discover all .cif / .mmcif files recursively
        files = sorted(self.root.rglob('*.cif')) + sorted(self.root.rglob('*.mmcif'))
        if not files:
            raise FileNotFoundError(
                f'No .cif/.mmcif files found under {self.root}. '
                f'Did you unpack rna3db-mmcifs.tar.xz there?'
            )
        self.files = files

        self._cache: Dict[int, Optional[Dict]] = {}
        if preload:
            for i, f in enumerate(files):
                self._cache[i] = _load_or_parse(f, self.cache_dir)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        if idx in self._cache:
            rec = self._cache[idx]
        else:
            rec = _load_or_parse(self.files[idx], self.cache_dir)

        if rec is None:
            # Return a minimal stub; DataLoader collate will skip in _safe_collate
            return None

        tokens      = rec['tokens']        # LongTensor [L]
        coords_true = rec['coords_true']   # FloatTensor [L,3,3]
        L           = tokens.shape[0]

        # Filter by minimum length
        if L < self.min_len:
            return None

        # Crop/truncate
        if self.max_len is not None and L > self.max_len:
            if self.crop_mode == 'random':
                start = torch.randint(0, L - self.max_len + 1, ()).item()
            else:
                start = 0
            end         = start + self.max_len
            tokens      = tokens[start:end]
            coords_true = coords_true[start:end]

        return {
            'tokens':      tokens,
            'coords_true': coords_true,
            'id':          rec['id'],
            'length':      tokens.shape[0],
        }


# ─── Collate ──────────────────────────────────────────────────────────────────

def collate_rna3d(batch: List[Optional[Dict]]) -> Dict:
    """
    Pad a list of samples to the same length.

    Samples that failed to parse (None) are silently dropped.  If the entire
    batch is None, an empty dict is returned (train_epoch will produce loss=0).
    """
    batch = [x for x in batch if x is not None]
    if not batch:
        return {}

    B    = len(batch)
    Lmax = max(x['tokens'].shape[0] for x in batch)

    tokens      = torch.full((B, Lmax), PAD_ID, dtype=torch.long)
    coords_true = torch.zeros((B, Lmax, 3, 3), dtype=torch.float32)
    seq_mask    = torch.zeros((B, Lmax), dtype=torch.bool)
    ids: List[str] = []

    for i, x in enumerate(batch):
        L = x['tokens'].shape[0]
        tokens[i, :L]         = x['tokens']
        coords_true[i, :L]    = x['coords_true']
        seq_mask[i, :L]       = True
        ids.append(x['id'])

    return {
        'tokens':      tokens,
        'coords_true': coords_true,
        'seq_mask':    seq_mask,
        'ids':         ids,
    }


# ─── Quick validation ─────────────────────────────────────────────────────────

def validate_dataset(root: str, n_check: int = 10) -> None:
    """Print a summary of the first n_check parseable files."""
    ds = RNA3DTertiaryDataset(root, max_len=None)
    print(f'Found {len(ds)} files under {root}')
    checked = 0
    for i in range(min(len(ds), 50)):
        rec = ds[i]
        if rec is None:
            continue
        L = rec['tokens'].shape[0]
        seq = ''.join('ACGUN'[t.item()] for t in rec['tokens'])[:20]
        c   = rec['coords_true']
        print(f'  [{i:4d}] {rec["id"]:30s}  L={L:4d}  seq={seq}...'
              f'  C4\'[0]={c[0,1].numpy().round(1)}')
        checked += 1
        if checked >= n_check:
            break


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python rna_tertiary_data.py <rna3db_root>')
        sys.exit(1)
    validate_dataset(sys.argv[1])
