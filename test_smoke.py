"""
Smoke test for the aux_struct extension.

Runs entirely on CPU, requires no real data files, and does NOT need ViennaRNA
(compute_ss_mfe falls back to all-dots / 0.0 MFE if RNA is absent).

Usage:
    python test_smoke.py
    python test_smoke.py --verbose   # print per-batch losses

Exits with code 0 on success, 1 on any failure.
"""

import sys
import traceback
import argparse
import numpy as np
import torch

VERBOSE = '--verbose' in sys.argv


def section(name):
    print(f'\n{"-"*50}\n{name}\n{"-"*50}')


def ok(msg):
    print(f'  [PASS] {msg}')


def fail(msg, exc=None):
    print(f'  [FAIL] {msg}')
    if exc:
        traceback.print_exc()
    sys.exit(1)


# ─── helpers ──────────────────────────────────────────────────────────────────

def make_synthetic_samples(n=32, min_len=20, max_len=50, with_aux=False, seed=0):
    """
    Create n synthetic preprocess_sample dicts using zero BPP (no ViennaRNA).
    When with_aux=True, also add ss_ids and mfe.
    """
    from rna_structure_plucker import preprocess_sample, compute_ss_mfe, encode_ss
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        L   = int(rng.integers(min_len, max_len + 1))
        seq = ''.join(rng.choice(['A', 'C', 'G', 'U'], size=L))
        bpp = np.zeros((L, L), dtype=np.float32)   # zero -> local edges only
        s   = preprocess_sample(seq, bpp)
        s['label'] = float(rng.standard_normal())
        if with_aux:
            ss_str, mfe = compute_ss_mfe(seq)      # fallback: all-dots, 0.0
            s['ss_ids'] = encode_ss(ss_str)
            s['mfe']    = np.float32(mfe)
        samples.append(s)
    return samples


def make_loader(samples, batch_size=8):
    from rna_structure_plucker import collate_rna
    from torch.utils.data import DataLoader

    class _DS(torch.utils.data.Dataset):
        def __init__(self, s): self.s = s
        def __len__(self): return len(self.s)
        def __getitem__(self, i): return self.s[i]

    return DataLoader(_DS(samples), batch_size=batch_size,
                      shuffle=False, collate_fn=collate_rna)


# ─── test 1: collate produces expected keys ───────────────────────────────────

section('Test 1 — collate_rna keys (with and without aux)')

try:
    from rna_structure_plucker import collate_rna
    plain   = make_synthetic_samples(8, with_aux=False)
    with_aux = make_synthetic_samples(8, with_aux=True)

    b_plain = collate_rna(plain)
    b_aux   = collate_rna(with_aux)

    required = {'input_ids', 'edge_index', 'edge_mask', 'edge_attrs', 'seq_mask', 'labels'}
    for k in required:
        assert k in b_plain, f'missing key {k!r} in plain batch'
        assert k in b_aux,   f'missing key {k!r} in aux batch'

    assert 'ss_ids'  not in b_plain, 'ss_ids should be absent in plain batch'
    assert 'mfe'     not in b_plain, 'mfe should be absent in plain batch'
    assert 'ss_ids'  in b_aux,       'ss_ids missing from aux batch'
    assert 'mfe'     in b_aux,       'mfe missing from aux batch'

    # shapes
    B, L = b_aux['input_ids'].shape
    assert b_aux['ss_ids'].shape  == (B, L), f'ss_ids shape {b_aux["ss_ids"].shape}'
    assert b_aux['mfe'].shape     == (B,),   f'mfe shape {b_aux["mfe"].shape}'

    # padding: padded positions should have SS_IGNORE_IDX (-100)
    from rna_structure_plucker import SS_IGNORE_IDX
    pad_mask  = ~b_aux['seq_mask']             # True where padded
    if pad_mask.any():
        assert (b_aux['ss_ids'][pad_mask] == SS_IGNORE_IDX).all(), \
            'padded ss_ids should be SS_IGNORE_IDX'

    ok('plain batch has correct keys')
    ok('aux batch has correct keys, shapes, and padding')

except Exception as e:
    fail('collate_rna', e)


# ─── test 2: model forward, aux_struct=False ──────────────────────────────────

section('Test 2 — RNAStructureGrassmann forward, aux_struct=False (baseline)')

try:
    from rna_structure_plucker import RNAStructureGrassmann

    model = RNAStructureGrassmann(model_dim=32, num_layers=2, reduced_dim=8,
                                  aux_struct=False)
    model.eval()

    loader = make_loader(make_synthetic_samples(16, with_aux=False))
    for batch in loader:
        labels = batch.pop('labels')
        batch.pop('ss_ids', None)
        batch.pop('mfe', None)
        with torch.no_grad():
            logits, loss = model(**batch, labels=labels)
        assert logits.shape == (labels.shape[0],), f'logits shape {logits.shape}'
        assert loss is not None and loss.isfinite(), f'loss not finite: {loss}'
        if VERBOSE:
            print(f'    logits={logits[:3].tolist()}, loss={loss.item():.4f}')
        break   # one batch is enough

    ok('forward pass OK, loss finite')

except Exception as e:
    fail('aux_struct=False forward', e)


# ─── test 3: model forward, aux_struct=True, labels present ──────────────────

section('Test 3 — RNAStructureGrassmann forward, aux_struct=True (multi-task loss)')

try:
    from rna_structure_plucker import RNAStructureGrassmann

    model = RNAStructureGrassmann(model_dim=32, num_layers=2, reduced_dim=8,
                                  aux_struct=True, lambda_ss=0.1, lambda_mfe=0.01)
    model.eval()

    loader = make_loader(make_synthetic_samples(16, with_aux=True))
    for batch in loader:
        labels     = batch.pop('labels')
        ss_labels  = batch.pop('ss_ids')
        mfe_labels = batch.pop('mfe')
        with torch.no_grad():
            logits, loss = model(**batch, labels=labels,
                                 ss_labels=ss_labels, mfe_labels=mfe_labels)
        assert logits.shape == (labels.shape[0],), f'logits shape {logits.shape}'
        assert loss is not None and loss.isfinite(), f'loss not finite: {loss}'
        if VERBOSE:
            print(f'    logits={logits[:3].tolist()}, loss={loss.item():.4f}')
        break

    ok('multi-task forward pass OK, combined loss finite')

except Exception as e:
    fail('aux_struct=True forward', e)


# ─── test 4: forward without labels -> loss is None ───────────────────────────

section('Test 4 — no labels -> loss is None')

try:
    from rna_structure_plucker import RNAStructureGrassmann

    model = RNAStructureGrassmann(model_dim=32, num_layers=2, reduced_dim=8,
                                  aux_struct=True)
    model.eval()

    loader = make_loader(make_synthetic_samples(8, with_aux=False))
    for batch in loader:
        batch.pop('labels', None)
        batch.pop('ss_ids', None)
        batch.pop('mfe', None)
        with torch.no_grad():
            logits, loss = model(**batch)
        assert loss is None, f'expected None loss, got {loss}'
        break

    ok('loss is None when no labels provided')

except Exception as e:
    fail('no-labels forward', e)


# ─── test 5: backward pass (gradient flow) ───────────────────────────────────

section('Test 5 — backward pass through multi-task loss')

try:
    from rna_structure_plucker import RNAStructureGrassmann

    model = RNAStructureGrassmann(model_dim=32, num_layers=2, reduced_dim=8,
                                  aux_struct=True, lambda_ss=0.1, lambda_mfe=0.01)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    loader = make_loader(make_synthetic_samples(16, with_aux=True))
    losses = []
    for batch in loader:
        labels     = batch.pop('labels')
        ss_labels  = batch.pop('ss_ids')
        mfe_labels = batch.pop('mfe')
        opt.zero_grad()
        _, loss = model(**batch, labels=labels,
                        ss_labels=ss_labels, mfe_labels=mfe_labels)
        loss.backward()
        # check gradients exist and are finite
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert p.grad.isfinite().all(), f'non-finite grad in {name}'
        opt.step()
        losses.append(loss.item())

    if VERBOSE:
        print(f'    batch losses: {[f"{l:.4f}" for l in losses]}')
    ok(f'backward OK over {len(losses)} batches, all gradients finite')

except Exception as e:
    fail('backward pass', e)


# ─── test 6: aux_struct=True, aux labels absent -> only primary loss ───────────

section('Test 6 — aux_struct=True but aux labels absent at inference')

try:
    from rna_structure_plucker import RNAStructureGrassmann

    model = RNAStructureGrassmann(model_dim=32, num_layers=2, reduced_dim=8,
                                  aux_struct=True)
    model.eval()

    loader = make_loader(make_synthetic_samples(8, with_aux=False))
    for batch in loader:
        labels = batch.pop('labels')
        with torch.no_grad():
            logits, loss = model(**batch, labels=labels)  # no ss/mfe labels
        assert loss is not None and loss.isfinite(), f'loss: {loss}'
        ok('primary-only loss computed without aux labels even when aux_struct=True')
        break

except Exception as e:
    fail('primary-only with aux_struct=True', e)


# ─── test 7: parameter count changes correctly ────────────────────────────────

section('Test 7 — parameter count with and without aux heads')

try:
    from rna_structure_plucker import RNAStructureGrassmann

    cfg = dict(model_dim=64, num_layers=2, reduced_dim=16)
    m_base = RNAStructureGrassmann(**cfg, aux_struct=False)
    m_aux  = RNAStructureGrassmann(**cfg, aux_struct=True)

    n_base = m_base.get_num_params()
    n_aux  = m_aux.get_num_params()
    diff   = n_aux - n_base

    # ss_head: 64*3 + 3 = 195, mfe_head: 64*1 + 1 = 65 -> total 260
    expected_diff = 64 * 3 + 3 + 64 * 1 + 1
    assert diff == expected_diff, \
        f'param diff {diff} != expected {expected_diff}'

    if VERBOSE:
        print(f'    base={n_base:,}  aux={n_aux:,}  diff={diff} (expected {expected_diff})')
    ok(f'aux heads add exactly {diff} parameters (ss_head + mfe_head)')

except Exception as e:
    fail('parameter count', e)


# ─── test 8: mini train loop via train_utr helpers ────────────────────────────

section('Test 8 — mini training loop through train_utr.train_epoch + evaluate')

try:
    from train_utr import TrainConfig, build_model, WarmupCosineScheduler
    from rna_structure_plucker import collate_rna
    from torch.utils.data import DataLoader

    # Minimal config — no CSV needed
    cfg = TrainConfig(
        task='mrl', data='', bpp_backend='zero',
        model_dim=32, num_layers=2, reduced_dim=8, dropout=0.0,
        epochs=1, batch_size=8, lr=1e-3, warmup_steps=1,
        aux_struct=True, lambda_ss=0.1, lambda_mfe=0.01,
        use_amp=False, device='cpu',
    )

    model = build_model(cfg)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = WarmupCosineScheduler(opt, warmup_steps=1, total_steps=10)

    samples_train = make_synthetic_samples(24, with_aux=True)
    samples_val   = make_synthetic_samples(8,  with_aux=True)

    # collate_utr adds library_ids; our test samples lack it — use collate_rna
    train_loader = DataLoader(
        samples_train, batch_size=8, shuffle=False, collate_fn=collate_rna,
        # wrap list as a Dataset
    )

    # patch: make list subscriptable for DataLoader
    class _L(torch.utils.data.Dataset):
        def __init__(self, s): self.s = s
        def __len__(self): return len(self.s)
        def __getitem__(self, i): return self.s[i]

    from train_utr import train_epoch, evaluate

    train_loader = DataLoader(_L(samples_train), batch_size=8,
                              shuffle=False, collate_fn=collate_rna)
    val_loader   = DataLoader(_L(samples_val),   batch_size=8,
                              shuffle=False, collate_fn=collate_rna)

    # Patch pop('library_ids') — not in batch; already handled via .pop(key, None)
    avg_loss = train_epoch(model, train_loader, opt, sched,
                           torch.device('cpu'), clip_grad=1.0, scaler=None)
    metrics  = evaluate(model, val_loader, torch.device('cpu'), task='regression')

    assert np.isfinite(avg_loss), f'avg_loss={avg_loss}'
    for k, v in metrics.items():
        assert np.isfinite(v), f'{k}={v} is not finite'

    if VERBOSE:
        print(f'    avg_loss={avg_loss:.4f} | val: {metrics}')
    ok(f'train_epoch + evaluate OK  (avg_loss={avg_loss:.4f}, '
       f'pearson_r={metrics.get("pearson_r", "n/a"):.4f})')

except Exception as e:
    fail('train_utr loop', e)


# ─── done ─────────────────────────────────────────────────────────────────────

section('All tests passed')
print('  Safe to transfer to the pod.\n')
sys.exit(0)
