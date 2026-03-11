"""
XAI toolkit for the Structure-Edge Plücker model.

Three analysis families:

  A. Edge / token attribution
       explain_sample()          gradient × weighted-edge-message per edge
       edge_table()              ranked DataFrame of edge contributions
       ablate_edges()            causal test: re-predict with edges zeroed
       ablation_sanity_check()   top-k vs random ablation comparison

  B. Geometric feature analysis
       coordinate_importance()   E[|grad_p · p_hat|] per Plücker coordinate
       p_norm_statistics()       ‖u ∧ v‖ distribution per layer / edge-type
       sequence_plucker_features() per-sample mean |p_hat| vector for clustering

  C. Biological faithfulness
       stem_overlap_metrics()    precision/recall vs ViennaRNA stems
       position_motif_enrichment() attribution mass near Kozak / uAUGs
       mutation_sensitivity()    Δprediction when top-attributed positions mutated

Usage example
─────────────
    import torch, pandas as pd
    from rna_structure_plucker import RNAStructureGrassmann, preprocess_sample, collate_rna
    from xai_plucker import explain_sample, edge_table, ablation_sanity_check

    model = RNAStructureGrassmann(model_dim=128, num_layers=6, reduced_dim=16)
    ckpt  = torch.load("outputs/mrl_fold1_best.pt", map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    seq   = "ACGUACGUACGU..."
    bpp   = compute_bpp(seq)
    batch = collate_rna([preprocess_sample(seq, bpp)])

    result = explain_sample(model, batch)
    df     = edge_table(result, seq, bpp)
    print(df.head(20))
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from rna_structure_plucker import (
    RNAStructureGrassmann,
    compute_bpp,
    compute_ss_mfe,
)


# ─── A. Edge / token attribution ──────────────────────────────────────────────

def explain_sample(
    model: RNAStructureGrassmann,
    batch: Dict[str, torch.Tensor],
    batch_idx: int = 0,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Gradient × weighted-edge-message attribution for one sample.

    The score for edge (t, slot k) at layer ℓ is:

        s[ℓ, t, k] = Σ_d | ∂y/∂w[ℓ,t,k,d] · w[ℓ,t,k,d] |

    where w = weighted_edge_msg = attn_weights * m_per_edge.

    This combines message content and attention weighting, and is anchored
    to the actual scalar prediction via backprop.

    Args:
        model:     trained RNAStructureGrassmann (eval mode recommended)
        batch:     collated batch from collate_rna(); only batch[batch_idx] is used
        batch_idx: which element in the batch to explain
        device:    if None, uses the device of model parameters

    Returns dict with keys:
        logit            float — raw scalar prediction
        seq_len          int   — actual sequence length (no padding)
        layer_caches     list of per-layer cache dicts (B=1 view, no grad)
        edge_scores      (num_layers, L, K) float32 numpy — per-layer attribution
        edge_scores_sum  (L, K) float32 numpy — summed over layers
        position_scores  (L,)   float32 numpy — per-token (outgoing + incoming)
    """
    if device is None:
        device = next(model.parameters()).device

    # Extract single sample as B=1 batch so shapes stay consistent
    b = batch_idx
    single = {
        k: v[b:b+1].to(device) for k, v in batch.items()
        if isinstance(v, torch.Tensor)
    }

    model.eval()

    # Forward with cache; keep computation graph
    out = model(
        input_ids   = single['input_ids'],
        edge_index  = single['edge_index'],
        edge_mask   = single['edge_mask'],
        edge_attrs  = single['edge_attrs'],
        seq_mask    = single['seq_mask'],
        return_cache = True,
    )
    logits, _, layer_caches = out
    logit = logits[0]  # scalar

    # Register retain_grad on all weighted_edge_msg tensors
    for cache in layer_caches:
        cache['weighted_edge_msg'].retain_grad()

    # Backprop through the scalar prediction
    logit.backward()

    # Collect scores — shape (num_layers, L, K)
    num_layers = len(layer_caches)
    L = single['input_ids'].shape[1]
    K = single['edge_index'].shape[2]

    edge_scores = np.zeros((num_layers, L, K), dtype=np.float32)

    for ℓ, cache in enumerate(layer_caches):
        wmsg = cache['weighted_edge_msg']   # (1, L, K, d)
        grad = wmsg.grad                    # (1, L, K, d)
        if grad is None:
            continue
        # gradient × input, sum over feature dim, squeeze batch
        score = (grad * wmsg).abs().sum(dim=-1)[0]  # (L, K)
        edge_scores[ℓ] = score.detach().cpu().numpy()

    edge_scores_sum = edge_scores.sum(axis=0)  # (L, K)

    # Position attribution: outgoing + incoming at each token
    # Use topology from layer 0 (same across all layers)
    safe_idx = layer_caches[0]['safe_idx'][0].cpu().numpy()   # (L, K)
    edge_mask_np = layer_caches[0]['edge_mask'][0].cpu().numpy()  # (L, K)

    pos_scores = edge_scores_sum.sum(axis=-1).copy()  # outgoing sum
    for t in range(L):
        for k in range(K):
            if not edge_mask_np[t, k]:
                continue
            j = int(safe_idx[t, k])
            if 0 <= j < L:
                pos_scores[j] += edge_scores_sum[t, k]

    # Detach caches for inspection (no longer need grads)
    clean_caches = []
    for cache in layer_caches:
        clean_caches.append({
            ck: cv.detach().cpu() for ck, cv in cache.items()
        })

    return {
        'logit':           logit.item(),
        'seq_len':         int(single['seq_mask'][0].sum().item()),
        'layer_caches':    clean_caches,
        'edge_scores':     edge_scores,
        'edge_scores_sum': edge_scores_sum,
        'position_scores': pos_scores,
    }


def edge_table(
    result: Dict,
    seq: str,
    bpp: Optional[np.ndarray] = None,
) -> 'pd.DataFrame':
    """
    Build a ranked DataFrame from the output of explain_sample().

    Columns:
        rank, layer_max, t (source pos), j (target pos),
        edge_type (local/structure), score_sum, score_max,
        attn_weight, bp_prob, seq_dist, nuc_src, nuc_tgt
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for edge_table()")

    caches = result['layer_caches']
    edge_scores = result['edge_scores']          # (num_layers, L, K)
    edge_scores_sum = result['edge_scores_sum']  # (L, K)

    L = result['seq_len']
    safe_idx   = caches[0]['safe_idx'][0].numpy()    # (L, K)
    edge_mask  = caches[0]['edge_mask'][0].numpy()   # (L, K)
    edge_attrs = caches[0]['edge_attrs'][0].numpy()  # (L, K, 3)

    # Use layer-averaged attention weights
    attn_stack = np.stack(
        [c['attn_weights'][0].numpy() for c in caches], axis=0
    )  # (num_layers, L, K)
    mean_attn = attn_stack.mean(axis=0)  # (L, K)

    rows = []
    for t in range(L):
        for k in range(K):
            if not edge_mask[t, k]:
                continue
            j = int(safe_idx[t, k])
            if j < 0 or j >= L:
                continue
            score_sum  = float(edge_scores_sum[t, k])
            score_max  = float(edge_scores[:, t, k].max())
            layer_max  = int(edge_scores[:, t, k].argmax())
            etype      = 'structure' if edge_attrs[t, k, 2] > 0.5 else 'local'
            bp_prob    = float(bpp[t, j]) if bpp is not None else float(edge_attrs[t, k, 0])
            seq_dist   = abs(t - j)
            attn_w     = float(mean_attn[t, k])
            nuc_src    = seq[t] if t < len(seq) else '?'
            nuc_tgt    = seq[j] if j < len(seq) else '?'
            rows.append(dict(
                t=t, j=j,
                edge_type=etype,
                score_sum=score_sum,
                score_max=score_max,
                layer_max=layer_max,
                attn_weight=attn_w,
                bp_prob=bp_prob,
                seq_dist=seq_dist,
                nuc_src=nuc_src,
                nuc_tgt=nuc_tgt,
            ))

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values('score_sum', ascending=False).reset_index(drop=True)
    df.insert(0, 'rank', df.index + 1)
    return df


# ─── Edge ablation ─────────────────────────────────────────────────────────────

def ablate_edges(
    model: RNAStructureGrassmann,
    batch: Dict[str, torch.Tensor],
    edges: List[Tuple[int, int]],
    batch_idx: int = 0,
    device: Optional[torch.device] = None,
) -> float:
    """
    Re-run forward with selected edges zeroed. Returns new scalar prediction.

    Args:
        edges: list of (token_pos t, slot_index k) pairs to remove.
               Use edge_table() to find the slot index (column k in edge_index).
    """
    if device is None:
        device = next(model.parameters()).device

    b = batch_idx
    single = {
        k: v[b:b+1].clone().to(device) for k, v in batch.items()
        if isinstance(v, torch.Tensor)
    }

    # Zero out the selected edges
    for t, k in edges:
        single['edge_mask'][0, t, k] = False
        single['edge_attrs'][0, t, k] = 0.0
        single['edge_index'][0, t, k] = -1

    model.eval()
    with torch.no_grad():
        logits, _ = model(
            input_ids  = single['input_ids'],
            edge_index = single['edge_index'],
            edge_mask  = single['edge_mask'],
            edge_attrs = single['edge_attrs'],
            seq_mask   = single['seq_mask'],
        )
    return logits[0].item()


def ablation_sanity_check(
    model: RNAStructureGrassmann,
    batch: Dict[str, torch.Tensor],
    result: Dict,
    top_k: int = 10,
    n_random_trials: int = 20,
    seed: int = 42,
    batch_idx: int = 0,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Compare |Δy| for top-attributed edges vs randomly sampled edges.

    Returns:
        original_logit   float
        top_k_delta      float — |Δy| when top-k edges ablated
        random_deltas    list of floats — |Δy| for each random k-edge ablation
        top_k_edges      list of (t, k) pairs that were ablated
        enrichment       top_k_delta / mean(random_deltas)
    """
    rng = random.Random(seed)

    caches   = result['layer_caches']
    edge_sum = result['edge_scores_sum']   # (L, K)
    edge_mask_np = caches[0]['edge_mask'][0].numpy()  # (L, K)
    L, K = edge_mask_np.shape

    # All valid (t, k) slots
    valid = [(t, k) for t in range(L) for k in range(K) if edge_mask_np[t, k]]
    if not valid:
        return {}

    # Top-k by attribution
    scored = sorted(valid, key=lambda tk: edge_sum[tk[0], tk[1]], reverse=True)
    top_edges = scored[:top_k]

    original = result['logit']
    top_pred  = ablate_edges(model, batch, top_edges, batch_idx, device)
    top_delta = abs(original - top_pred)

    # Random trials
    random_deltas = []
    for _ in range(n_random_trials):
        rand_edges = rng.sample(valid, min(top_k, len(valid)))
        rand_pred  = ablate_edges(model, batch, rand_edges, batch_idx, device)
        random_deltas.append(abs(original - rand_pred))

    mean_rand = float(np.mean(random_deltas)) if random_deltas else 1.0
    return {
        'original_logit': original,
        'top_k_delta':    top_delta,
        'random_deltas':  random_deltas,
        'top_k_edges':    top_edges,
        'enrichment':     top_delta / max(mean_rand, 1e-9),
    }


# ─── B. Geometric feature analysis ────────────────────────────────────────────

def coordinate_importance(
    model: RNAStructureGrassmann,
    dataloader,
    device: torch.device,
    max_samples: int = 500,
) -> np.ndarray:
    """
    Dataset-level Plücker coordinate importance.

    For each layer ℓ and each Plücker coordinate c:
        I[ℓ, c] = E_{samples, valid edges} [ |∂y/∂p_hat[ℓ,t,k,c] · p_hat[ℓ,t,k,c]| ]

    Returns:
        importance: (num_layers, plucker_dim) float32 numpy array
    """
    model.eval()
    num_layers  = len(model.blocks)
    plucker_dim = model.blocks[0].plucker_mix.plucker_dim

    accum  = np.zeros((num_layers, plucker_dim), dtype=np.float64)
    counts = np.zeros((num_layers, plucker_dim), dtype=np.float64)

    n_done = 0
    for batch in dataloader:
        if n_done >= max_samples:
            break
        B = batch['input_ids'].shape[0]

        for b in range(B):
            if n_done >= max_samples:
                break

            single = {
                k: v[b:b+1].to(device) for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

            # Forward with cache
            out = model(
                input_ids    = single['input_ids'],
                edge_index   = single['edge_index'],
                edge_mask    = single['edge_mask'],
                edge_attrs   = single['edge_attrs'],
                seq_mask     = single['seq_mask'],
                return_cache = True,
            )
            logits, _, layer_caches = out
            logit = logits[0]

            for cache in layer_caches:
                cache['p_hat'].retain_grad()

            logit.backward()

            for ℓ, cache in enumerate(layer_caches):
                p_hat = cache['p_hat']      # (1, L, K, P)
                grad  = p_hat.grad
                if grad is None:
                    continue
                mask = cache['edge_mask'][0]  # (L, K) bool
                # importance: |grad * p_hat| over valid edges
                imp  = (grad * p_hat).abs()[0]  # (L, K, P)
                valid = mask.unsqueeze(-1).expand_as(imp)
                accum[ℓ]  += imp[valid].detach().cpu().numpy().reshape(-1, plucker_dim).sum(axis=0)
                counts[ℓ] += valid.float().sum(dim=(0, 1)).detach().cpu().numpy()

            n_done += 1

    # Safe mean
    with np.errstate(invalid='ignore'):
        importance = np.where(counts > 0, accum / counts, 0.0)
    return importance.astype(np.float32)


def p_norm_statistics(
    model: RNAStructureGrassmann,
    dataloader,
    device: torch.device,
    max_samples: int = 500,
) -> Dict:
    """
    Compute per-layer p_norm (‖u ∧ v‖) statistics across the dataset.

    Returns dict:
        mean_by_layer        (num_layers,) — mean p_norm per layer
        mean_by_layer_local  (num_layers,) — only local edges
        mean_by_layer_struct (num_layers,) — only structure edges
        all_pnorms           list of (layer_idx, edge_type, p_norm) tuples
                             (truncated to max_samples * ~50 edges)
    """
    model.eval()
    num_layers = len(model.blocks)

    sums_all    = np.zeros(num_layers, dtype=np.float64)
    sums_local  = np.zeros(num_layers, dtype=np.float64)
    sums_struct = np.zeros(num_layers, dtype=np.float64)
    cnt_all     = np.zeros(num_layers, dtype=np.float64)
    cnt_local   = np.zeros(num_layers, dtype=np.float64)
    cnt_struct  = np.zeros(num_layers, dtype=np.float64)

    n_done = 0
    for batch in dataloader:
        if n_done >= max_samples:
            break
        B = batch['input_ids'].shape[0]
        for b in range(min(B, max_samples - n_done)):
            single = {
                k: v[b:b+1].to(device) for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            with torch.no_grad():
                out = model(
                    input_ids    = single['input_ids'],
                    edge_index   = single['edge_index'],
                    edge_mask    = single['edge_mask'],
                    edge_attrs   = single['edge_attrs'],
                    seq_mask     = single['seq_mask'],
                    return_cache = True,
                )
            _, _, layer_caches = out

            for ℓ, cache in enumerate(layer_caches):
                pn   = cache['p_norm'][0].cpu().numpy()       # (L, K)
                mask = cache['edge_mask'][0].cpu().numpy()     # (L, K)
                is_s = cache['edge_attrs'][0, :, :, 2].cpu().numpy() > 0.5  # struct flag
                valid = mask

                sums_all[ℓ]   += pn[valid].sum()
                cnt_all[ℓ]    += valid.sum()
                sums_local[ℓ] += pn[valid & ~is_s].sum()
                cnt_local[ℓ]  += (valid & ~is_s).sum()
                sums_struct[ℓ]+= pn[valid & is_s].sum()
                cnt_struct[ℓ] += (valid & is_s).sum()

            n_done += 1

    def safe_div(a, b):
        return np.where(b > 0, a / b, 0.0).astype(np.float32)

    return {
        'mean_by_layer':        safe_div(sums_all,    cnt_all),
        'mean_by_layer_local':  safe_div(sums_local,  cnt_local),
        'mean_by_layer_struct': safe_div(sums_struct, cnt_struct),
    }


def sequence_plucker_features(
    model: RNAStructureGrassmann,
    dataloader,
    device: torch.device,
    max_samples: int = 2000,
    use_attribution: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a per-sequence feature vector for clustering/UMAP.

    Option 1 (use_attribution=False, default):
        f = concat over layers of mean_valid_edges(|p_hat|)   shape: (num_layers * P,)

    Option 2 (use_attribution=True):
        f = concat over layers of mean_valid_edges(score * |p_hat|)
        Requires backprop; much slower.

    Returns:
        features:  (N, num_layers * plucker_dim) float32
        labels:    (N,) float32 — prediction logits from the model
    """
    model.eval()
    num_layers  = len(model.blocks)
    plucker_dim = model.blocks[0].plucker_mix.plucker_dim
    feat_dim    = num_layers * plucker_dim

    feats  = []
    logits_out = []
    n_done = 0

    for batch in dataloader:
        if n_done >= max_samples:
            break
        B = batch['input_ids'].shape[0]
        for b in range(min(B, max_samples - n_done)):
            single = {
                k: v[b:b+1].to(device) for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

            if use_attribution:
                result = explain_sample(model, {k: v.cpu() for k, v in single.items()},
                                        batch_idx=0, device=device)
                layer_caches = result['layer_caches']
                edge_scores  = result['edge_scores']   # (num_layers, L, K)
                f = []
                for ℓ, cache in enumerate(layer_caches):
                    p_hat = cache['p_hat'][0].numpy()        # (L, K, P)
                    mask  = cache['edge_mask'][0].numpy()    # (L, K)
                    scores = edge_scores[ℓ, :, :]            # (L, K)
                    w      = scores[:, :, None] * np.abs(p_hat)  # (L, K, P)
                    if mask.sum() > 0:
                        f.append(w[mask].mean(axis=0))
                    else:
                        f.append(np.zeros(plucker_dim, dtype=np.float32))
                feats.append(np.concatenate(f))
                logits_out.append(result['logit'])
            else:
                with torch.no_grad():
                    out = model(
                        input_ids    = single['input_ids'],
                        edge_index   = single['edge_index'],
                        edge_mask    = single['edge_mask'],
                        edge_attrs   = single['edge_attrs'],
                        seq_mask     = single['seq_mask'],
                        return_cache = True,
                    )
                logit, _, layer_caches = out
                f = []
                for cache in layer_caches:
                    p_hat = cache['p_hat'][0].cpu().numpy()   # (L, K, P)
                    mask  = cache['edge_mask'][0].cpu().numpy()
                    if mask.sum() > 0:
                        f.append(np.abs(p_hat[mask]).mean(axis=0))
                    else:
                        f.append(np.zeros(plucker_dim, dtype=np.float32))
                feats.append(np.concatenate(f))
                logits_out.append(logit[0].item())

            n_done += 1

    return np.stack(feats, axis=0), np.array(logits_out, dtype=np.float32)


# ─── C. Biological faithfulness ───────────────────────────────────────────────

def stem_overlap_metrics(
    df_edges: 'pd.DataFrame',
    bpp: np.ndarray,
    bp_threshold: float = 0.3,
    top_k: int = 20,
) -> Dict:
    """
    Measure how well top-attributed structure edges align with ViennaRNA stems.

    Args:
        df_edges:      output of edge_table(), sorted by score_sum descending
        bpp:           (L, L) base-pair probability matrix
        bp_threshold:  minimum BPP to count a pair as a ViennaRNA stem pair
        top_k:         number of top edges to consider

    Returns dict:
        precision_at_k      fraction of top-k struct edges that are stem pairs
        mean_bp_prob_top_k  mean BPP among top-k structure edges
        mean_bp_prob_all    mean BPP among all structure edges in the model graph
        enrichment          mean_bp_prob_top_k / mean_bp_prob_all
    """
    struct_df = df_edges[df_edges['edge_type'] == 'structure'].copy()
    if struct_df.empty:
        return {'precision_at_k': 0.0, 'mean_bp_prob_top_k': 0.0,
                'mean_bp_prob_all': 0.0, 'enrichment': 1.0}

    top_struct = struct_df.head(top_k)
    hits = sum(
        1 for _, row in top_struct.iterrows()
        if bpp[int(row['t']), int(row['j'])] >= bp_threshold
    )
    precision = hits / len(top_struct) if len(top_struct) > 0 else 0.0

    mean_top  = float(top_struct['bp_prob'].mean()) if len(top_struct) > 0 else 0.0
    mean_all  = float(struct_df['bp_prob'].mean())  if len(struct_df)  > 0 else 1.0

    return {
        'precision_at_k':     precision,
        'mean_bp_prob_top_k': mean_top,
        'mean_bp_prob_all':   mean_all,
        'enrichment':         mean_top / max(mean_all, 1e-9),
    }


def mutation_sensitivity(
    model: RNAStructureGrassmann,
    seq: str,
    bpp: np.ndarray,
    result: Dict,
    top_k_positions: int = 5,
    n_random_positions: int = 10,
    device: Optional[torch.device] = None,
    seed: int = 42,
) -> Dict:
    """
    Mutate top-attributed positions vs random positions and compare |Δprediction|.

    For each selected position, tries all 3 alternative nucleotides and records
    the maximum |Δy|.

    Returns:
        original_logit          float
        top_positions           list of (pos, score) sorted by attribution
        top_position_deltas     list of max|Δy| for each top position
        random_position_deltas  list of max|Δy| for each random position
        enrichment              mean(top) / mean(random)
    """
    from rna_structure_plucker import collate_rna, preprocess_sample

    if device is None:
        device = next(model.parameters()).device

    rng    = random.Random(seed)
    NUCS   = ['A', 'C', 'G', 'U']
    L      = result['seq_len']
    pos_sc = result['position_scores'][:L]

    # Top-k positions by attribution
    ranked     = sorted(range(L), key=lambda t: pos_sc[t], reverse=True)
    top_pos    = ranked[:top_k_positions]
    rand_pos   = rng.sample([p for p in range(L) if p not in top_pos],
                             min(n_random_positions, L - top_k_positions))

    original = result['logit']

    def _max_delta(pos: int) -> float:
        orig_nuc = seq[pos].upper().replace('T', 'U')
        alts     = [n for n in NUCS if n != orig_nuc]
        deltas   = []
        for alt in alts:
            mut_seq = seq[:pos] + alt + seq[pos+1:]
            mut_bpp = compute_bpp(mut_seq)
            sample  = preprocess_sample(mut_seq, mut_bpp)
            bat     = collate_rna([sample])
            bat     = {k: v.to(device) for k, v in bat.items()}
            model.eval()
            with torch.no_grad():
                lgts, _ = model(**bat)
            deltas.append(abs(original - lgts[0].item()))
        return max(deltas) if deltas else 0.0

    top_deltas  = [_max_delta(p) for p in top_pos]
    rand_deltas = [_max_delta(p) for p in rand_pos]

    mean_top  = float(np.mean(top_deltas))  if top_deltas  else 0.0
    mean_rand = float(np.mean(rand_deltas)) if rand_deltas else 1.0

    return {
        'original_logit':         original,
        'top_positions':          list(zip(top_pos, [float(pos_sc[p]) for p in top_pos])),
        'top_position_deltas':    top_deltas,
        'random_position_deltas': rand_deltas,
        'enrichment':             mean_top / max(mean_rand, 1e-9),
    }


def position_motif_enrichment(
    position_scores: np.ndarray,
    seq: str,
    uaug_positions: Optional[List[int]] = None,
    kozak_window: Optional[Tuple[int, int]] = None,
) -> Dict:
    """
    Test whether high-attribution positions are enriched in motif windows.

    Args:
        position_scores:  (L,) attribution scores from explain_sample()
        seq:              nucleotide string
        uaug_positions:   list of uAUG start positions (0-indexed)
        kozak_window:     (start, end) positions of Kozak consensus region

    Returns dict with attribution mass per region vs background.
    """
    L        = len(position_scores)
    total    = position_scores.sum() + 1e-12
    results  = {}

    # Auto-detect uAUGs if not provided
    if uaug_positions is None:
        uaug_positions = []
        for i in range(L - 2):
            codon = seq[i:i+3].upper().replace('T', 'U')
            if codon == 'AUG':
                uaug_positions.append(i)

    if uaug_positions:
        # ±2 nt window around each uAUG
        uaug_mask = np.zeros(L, dtype=bool)
        for p in uaug_positions:
            for offset in range(-2, 5):   # codon + flanks
                if 0 <= p + offset < L:
                    uaug_mask[p + offset] = True
        uaug_mass = float(position_scores[uaug_mask].sum() / total)
        uaug_frac = float(uaug_mask.mean())
        results['uaug_attribution_mass']   = uaug_mass
        results['uaug_sequence_fraction']  = uaug_frac
        results['uaug_enrichment']         = uaug_mass / max(uaug_frac, 1e-9)

    if kozak_window is not None:
        s, e   = kozak_window
        koz_mask = np.zeros(L, dtype=bool)
        koz_mask[max(0, s):min(L, e)] = True
        koz_mass = float(position_scores[koz_mask].sum() / total)
        koz_frac = float(koz_mask.mean())
        results['kozak_attribution_mass']  = koz_mass
        results['kozak_sequence_fraction'] = koz_frac
        results['kozak_enrichment']        = koz_mass / max(koz_frac, 1e-9)

    # GC-rich windows (top 25% by local GC)
    win = 5
    gc_scores = np.zeros(L)
    for i in range(L):
        w = seq[max(0, i-win):i+win+1].upper()
        gc_scores[i] = (w.count('G') + w.count('C')) / max(len(w), 1)
    gc_threshold   = np.percentile(gc_scores, 75)
    gc_mask        = gc_scores >= gc_threshold
    gc_mass        = float(position_scores[gc_mask].sum() / total)
    gc_frac        = float(gc_mask.mean())
    results['gc_rich_attribution_mass']  = gc_mass
    results['gc_rich_sequence_fraction'] = gc_frac
    results['gc_rich_enrichment']        = gc_mass / max(gc_frac, 1e-9)

    return results


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _load_model(ckpt_path: str, device: torch.device) -> RNAStructureGrassmann:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg  = ckpt.get('config', {})
    model = RNAStructureGrassmann(
        model_dim   = cfg.get('model_dim',   128),
        num_layers  = cfg.get('num_layers',  6),
        reduced_dim = cfg.get('reduced_dim', 16),
        dropout     = 0.0,
    )
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.to(device).eval()
    return model


if __name__ == '__main__':
    import argparse, json

    p = argparse.ArgumentParser(description='XAI analysis for the Plücker model')
    p.add_argument('--checkpoint', required=True, help='Path to _best.pt checkpoint')
    p.add_argument('--seq',        required=True, help='RNA sequence (A/C/G/U/T)')
    p.add_argument('--top_k',      type=int, default=20,
                   help='Top edges to report / ablate')
    p.add_argument('--ablate',     action='store_true',
                   help='Run ablation sanity check')
    p.add_argument('--mutate',     action='store_true',
                   help='Run mutation sensitivity analysis')
    p.add_argument('--device',     default='cpu')
    args = p.parse_args()

    from rna_structure_plucker import collate_rna, preprocess_sample

    device = torch.device(args.device)
    model  = _load_model(args.checkpoint, device)

    seq = args.seq.upper().replace('T', 'U')
    bpp = compute_bpp(seq)
    sample = preprocess_sample(seq, bpp)
    batch  = collate_rna([sample])

    print(f"Sequence length : {len(seq)}")
    print(f"Model           : {model.get_num_params():,} params")

    result = explain_sample(model, batch, device=device)
    print(f"Prediction      : {result['logit']:.4f}")

    df = edge_table(result, seq, bpp)
    print(f"\nTop {args.top_k} attributed edges:")
    print(df.head(args.top_k).to_string(index=False))

    stems = stem_overlap_metrics(df, bpp, top_k=args.top_k)
    print(f"\nStem overlap (top {args.top_k} structure edges):")
    print(json.dumps(stems, indent=2))

    if args.ablate:
        san = ablation_sanity_check(model, batch, result, top_k=args.top_k, device=device)
        print(f"\nAblation sanity check:")
        print(f"  top-{args.top_k} delta : {san['top_k_delta']:.4f}")
        print(f"  random mean delta : {np.mean(san['random_deltas']):.4f}")
        print(f"  enrichment        : {san['enrichment']:.2f}×")

    if args.mutate:
        mut = mutation_sensitivity(model, seq, bpp, result, device=device)
        print(f"\nMutation sensitivity:")
        print(f"  top positions : {mut['top_positions']}")
        print(f"  top deltas    : {[f'{d:.4f}' for d in mut['top_position_deltas']]}")
        print(f"  rand deltas   : {[f'{d:.4f}' for d in mut['random_position_deltas']]}")
        print(f"  enrichment    : {mut['enrichment']:.2f}×")

    motif = position_motif_enrichment(result['position_scores'][:result['seq_len']], seq)
    print(f"\nMotif enrichment:")
    print(json.dumps(motif, indent=2))
