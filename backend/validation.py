"""
validation.py -- method-agnostic validation: agreement, rank reversal, sensitivity.

Imports NO ranking method (only schema, to re-validate config variants). Methods are passed in as
callables. Reports measured numbers only; nothing here ranks one method above another.

RANK-FUNCTION PROTOCOL
----------------------
    rank_fn(df, weights=None) -> dict | pd.Series
      df       : DataFrame of alternatives (any subset of the full data; index must be unique)
      weights  : None  -> the method computes its own weights on THIS df (weights recomputed)
                 Series -> the method must use exactly these weights (weights frozen)
      returns  : dict with "ranks" (Series, 1 = best, aligned to df.index) and, if the method has
                 weights, "weights" (Series). A bare ranks Series is accepted, but then the
                 frozen-weights mode cannot run (no weights to freeze) and raises ValueError.
    rank_fn_factory(cfg_variant, df_full) -> rank_fn     (used by sensitivity)
      The factory must (re)fit anything frozen (appurtenance anchors, contradiction) on df_full
      under cfg_variant, then return a rank_fn closed over it.

All settings (iterations, drop fraction, pair cap, top-k, perturbation size, sweep grids) are read
from cfg['validation']; all are [D] team choices. Every function returns JSON-serialisable dicts.

ON FROZEN-WEIGHT RANK REVERSAL: if the score of an alternative does not depend on the other
alternatives (an additive score such as sum_j w_j * x_ij, e.g. PIV with normalization 'none' and
frozen weights) then removing rows cannot flip the order of two remaining rows, so 0% reversal is
true BY CONSTRUCTION and says nothing about the method's quality. With 'vector' normalisation the
column norms depend on the rows present, so the PIV order can change even with frozen weights; that
case is measured, not assumed. The PIV paper (Mufazzal & Muzakkir 2018) is described in the project
spec as saying rank reversal is minimised, not eliminated.
"""
from __future__ import annotations

import copy
import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

import schema as sch

MODE_RECOMPUTED = "weights_recomputed_on_subset"
MODE_FROZEN = "weights_frozen_from_full_data"
NOTE_FROZEN = ("Frozen weights: if the method's score is additive and independent of the other rows, "
               "0% reversal holds by construction and is not evidence about quality.")


# ---------------------------------------------------------------- JSON helper
def to_jsonable(o):
    """numpy / pandas / NaN-safe conversion to plain JSON types (NaN, inf -> None)."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, pd.Series):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return [to_jsonable(v) for v in o.tolist()]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


# ---------------------------------------------------------------- calling a rank function
def _call(rank_fn, df, weights=None):
    out = rank_fn(df) if weights is None else rank_fn(df, weights=weights)
    ranks = out["ranks"] if isinstance(out, dict) else out
    w = out.get("weights") if isinstance(out, dict) else None
    if not isinstance(ranks, pd.Series):
        raise ValueError("rank_fn must return a Series of ranks (or a dict with key 'ranks').")
    if set(ranks.index) != set(df.index) or len(ranks) != len(df):
        raise ValueError("rank_fn returned ranks whose index does not match the rows it was given.")
    r = ranks.loc[df.index].astype(float)
    if r.isna().any():
        raise ValueError("rank_fn returned NaN ranks.")
    return r, w


def _check_df(df):
    if not df.index.is_unique:
        raise ValueError("df.index must be unique (it identifies alternatives across subsets).")


# ---------------------------------------------------------------- agreement metrics
def compare_rankings(r1: pd.Series, r2: pd.Series, top_k) -> dict:
    """Spearman rho, Kendall tau-b and top-k overlap between two rank Series on the same index.
    top-k set = the k best by (rank, then row position); 'tie_at_cutoff' tells when that choice
    was ambiguous. Undefined correlations (constant ranks) are None, not a number."""
    r2 = r2.loc[r1.index]
    out = {"n": int(len(r1))}
    if len(r1) < 2 or r1.nunique() < 2 or r2.nunique() < 2:
        out["spearman"], out["kendall"] = None, None
    else:
        out["spearman"] = float(stats.spearmanr(r1, r2)[0])
        out["kendall"] = float(stats.kendalltau(r1, r2)[0])
    pos = pd.Series(np.arange(len(r1)), index=r1.index)
    for k in top_k:
        kk = min(int(k), len(r1))
        sets, ties = [], False
        for r in (r1, r2):
            order = pd.DataFrame({"r": r, "p": pos}).sort_values(["r", "p"]).index
            sets.append(set(order[:kk]))
            if kk < len(r):
                ties = ties or (r.sort_values().iloc[kk - 1] == r.sort_values().iloc[kk])
        inter = len(sets[0] & sets[1])
        out[f"top_{k}_overlap_count"] = inter
        out[f"top_{k}_overlap_fraction"] = inter / kk
        out[f"top_{k}_tie_at_cutoff"] = bool(ties)
    return to_jsonable(out)


# ---------------------------------------------------------------- rank reversal
def _sample_pairs(n: int, max_pairs: int, rng) -> tuple:
    """Up to max_pairs distinct unordered index pairs (i<j) out of n items; all pairs if few."""
    total = n * (n - 1) // 2
    if total <= max_pairs:
        i, j = np.triu_indices(n, 1)
        return i, j
    seen = set()
    while len(seen) < max_pairs:
        a = rng.integers(0, n, size=2 * (max_pairs - len(seen)))
        b = rng.integers(0, n, size=a.size)
        for x, y in zip(a.tolist(), b.tolist()):
            if x != y:
                seen.add((x, y) if x < y else (y, x))
                if len(seen) == max_pairs:
                    break
    arr = np.array(sorted(seen))
    return arr[:, 0], arr[:, 1]


def rank_reversal(df, rank_fn, cfg, mode: str, seed: int = 0, frozen_weights=None) -> dict:
    """
    One mode only (never mixed): MODE_RECOMPUTED (rank_fn(subset) recomputes weights) or
    MODE_FROZEN (rank_fn(subset, weights=frozen_weights from the FULL data)).
    Each iteration drops drop_fraction of the rows (seeded), ranks the rest, and on up to max_pairs
    sampled pairs of retained rows counts order flips against the FULL ranking. A pair is valid only
    if it is untied in both rankings; tied pairs are counted separately and excluded.
    The row subsets and pair samples depend only on (seed, iteration, n), so the two modes and all
    methods see identical subsets.
    """
    _check_df(df)
    rv = cfg["validation"]["reversal"]
    n_iter, frac, max_pairs = rv["n_iterations"], rv["drop_fraction"], rv["max_pairs"]
    if mode not in (MODE_RECOMPUTED, MODE_FROZEN):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == MODE_FROZEN and frozen_weights is None:
        raise ValueError("frozen mode needs frozen_weights.")
    full_ranks, _ = _call(rank_fn, df)
    n = len(df)
    n_drop = max(1, int(round(frac * n)))
    if n - n_drop < 2:
        raise ValueError(f"too few rows ({n}) to drop {n_drop} and still compare pairs.")
    iters = []
    for it in range(n_iter):
        rng = np.random.default_rng([int(seed), it])
        keep = np.sort(rng.choice(n, size=n - n_drop, replace=False))
        sub = df.iloc[keep]
        sub_ranks, _ = _call(rank_fn, sub, weights=frozen_weights if mode == MODE_FROZEN else None)
        i, j = _sample_pairs(len(sub), max_pairs, rng)
        ids = sub.index.to_numpy()
        rf = full_ranks.loc[ids].to_numpy()
        rs = sub_ranks.loc[ids].to_numpy()
        s_full, s_sub = np.sign(rf[i] - rf[j]), np.sign(rs[i] - rs[j])
        valid = (s_full != 0) & (s_sub != 0)
        rev = int((valid & (s_full != s_sub)).sum())
        nv = int(valid.sum())
        iters.append({"iteration": it, "rows_kept": int(len(sub)), "rows_dropped": int(n_drop),
                      "pairs_sampled": int(len(i)), "tied_pairs_excluded": int(len(i) - nv),
                      "valid_pairs": nv, "reversals": rev,
                      "reversal_pct": (100.0 * rev / nv) if nv else None})
    tv, tr = sum(x["valid_pairs"] for x in iters), sum(x["reversals"] for x in iters)
    pcts = [x["reversal_pct"] for x in iters if x["reversal_pct"] is not None]
    res = {"mode": mode, "seed": int(seed), "n_iterations": n_iter, "drop_fraction": frac,
           "max_pairs": max_pairs, "valid_pairs": tv, "reversals": tr,
           "reversal_pct": (100.0 * tr / tv) if tv else None,
           "per_iteration_pct_mean": float(np.mean(pcts)) if pcts else None,
           "per_iteration_pct_max": float(np.max(pcts)) if pcts else None,
           "iterations": iters}
    if mode == MODE_FROZEN:
        res["note"] = NOTE_FROZEN
    return to_jsonable(res)


# ---------------------------------------------------------------- evaluate
def evaluate(df: pd.DataFrame, cfg: dict, rank_fns: dict, seed: int = 0) -> dict:
    """
    rank_fns: {name: rank_fn} following the protocol above.
    Returns {"settings", "n_rows", "methods", "agreement": {"A__vs__B": metrics},
             "rank_reversal": {name: {MODE_RECOMPUTED: ..., MODE_FROZEN: ...}}}.
    The two reversal modes are separate keys and are never combined.
    """
    _check_df(df)
    if not rank_fns:
        raise ValueError("rank_fns is empty.")
    top_k = cfg["validation"]["top_k"]
    full = {name: _call(fn, df) for name, fn in rank_fns.items()}       # (ranks, weights)
    out = {"settings": {"seed": int(seed), "validation": copy.deepcopy(cfg["validation"]["reversal"]),
                        "top_k": list(top_k)},
           "n_rows": int(len(df)), "methods": list(rank_fns), "agreement": {}, "rank_reversal": {}}
    for a, b in itertools.combinations(rank_fns, 2):
        out["agreement"][f"{a}__vs__{b}"] = compare_rankings(full[a][0], full[b][0], top_k)
    for name in rank_fns:
        if full[name][1] is None:
            raise ValueError(f"method '{name}' returned no 'weights'; the frozen-weights mode cannot run. "
                             "Return a dict with 'ranks' and 'weights'.")
    for name, fn in rank_fns.items():
        w = full[name][1]
        entry = {MODE_RECOMPUTED: rank_reversal(df, fn, cfg, MODE_RECOMPUTED, seed)}
        entry[MODE_FROZEN] = rank_reversal(df, fn, cfg, MODE_FROZEN, seed, frozen_weights=w)
        out["rank_reversal"][name] = entry
    return to_jsonable(out)


# ---------------------------------------------------------------- sensitivity
def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Deep copy of cfg with dotted-path overrides applied TOGETHER. Unknown paths raise (typo
    guard). 'contradiction.uniform': c0 expands to method='expert' with an expert_matrix over the
    config's value names (zero diagonal, every off-diagonal = c0)."""
    new = copy.deepcopy(cfg)
    for path, val in overrides.items():
        if path == "contradiction.uniform":
            if not (isinstance(val, (int, float)) and not isinstance(val, bool) and 0 <= val <= 1):
                raise ValueError(f"contradiction.uniform must be a number in [0,1], got {val!r}")
            names = [v["name"] for v in new["plithogenic"]["values"]]
            new["contradiction"]["method"] = "expert"
            new["contradiction"]["expert_matrix"] = {a: {b: (0.0 if a == b else float(val)) for b in names}
                                                     for a in names}
            continue
        keys = path.split(".")
        node = new
        for k in keys[:-1]:
            if not isinstance(node, dict) or k not in node:
                raise ValueError(f"override path '{path}' does not exist in the config.")
            node = node[k]
        if not isinstance(node, dict) or keys[-1] not in node:
            raise ValueError(f"override path '{path}' does not exist in the config.")
        node[keys[-1]] = copy.deepcopy(val)
    return new


def _same_cfg(a, b) -> bool:
    strip = lambda c: {k: v for k, v in c.items() if not k.startswith("_")}
    return strip(a) == strip(b)


def _run_variant(label, cfg_variant, factory, df, base_ranks, top_k, **extra) -> dict:
    rec = {"label": label, **extra}
    try:
        sch.validate_config(cfg_variant)
        fn = factory(cfg_variant, df)
        ranks, _ = _call(fn, df)
    except Exception as e:                                    # recorded, not hidden
        rec.update(status="error", error_type=type(e).__name__, error=str(e))
        return rec
    rec["status"] = "ok"
    rec.update(compare_rankings(base_ranks, ranks, top_k))
    rec["max_abs_rank_shift"] = float((ranks - base_ranks).abs().max())
    return rec


def sensitivity(rank_fn_factory, param_grid, df: pd.DataFrame, cfg: dict) -> dict:
    """
    Compares every variant's ranking with the base ranking (factory(cfg, df)):
      weight_perturbation : each weight x (1 +/- p) one at a time, renormalised, held FROZEN;
      leave_one_value_out : each plithogenic value removed from the config (the dominant value is
                            skipped, it defines the contradictions); weights recomputed;
      param_sweeps        : for each name in param_grid (None -> cfg['validation']['sensitivity']
                            ['param_grid']) every entry of dotted-path overrides, weights recomputed.
    A variant that raises is reported with status='error' and the message, never silently dropped.
    Metrics: spearman, kendall, top-k overlap, max_abs_rank_shift. Descriptive only.
    """
    _check_df(df)
    sv = cfg["validation"]["sensitivity"]
    top_k = cfg["validation"]["top_k"]
    grid = sv["param_grid"] if param_grid is None else param_grid
    base_fn = rank_fn_factory(cfg, df)
    base_ranks, base_w = _call(base_fn, df)
    out = {"n_rows": int(len(df)), "weight_perturbation": [], "leave_one_value_out": [],
           "param_sweeps": {}}

    # --- weights +/- p, renormalised, frozen
    p = sv["weight_perturbation"]
    out["weight_perturbation_size"] = p
    if base_w is None:
        out["weight_perturbation"] = [{"status": "skipped", "reason": "method returned no weights"}]
    else:
        for crit in base_w.index:
            for sign in (+1, -1):
                w = base_w.astype(float).copy()
                w[crit] *= (1 + sign * p)
                w = w / w.sum()
                rec = {"label": f"{crit} {'+' if sign > 0 else '-'}{int(round(p * 100))}%",
                       "criterion": str(crit), "direction": sign}
                try:
                    ranks, _ = _call(base_fn, df, weights=w)
                except Exception as e:
                    rec.update(status="error", error_type=type(e).__name__, error=str(e))
                else:
                    rec["status"] = "ok"
                    rec.update(compare_rankings(base_ranks, ranks, top_k))
                    rec["max_abs_rank_shift"] = float((ranks - base_ranks).abs().max())
                out["weight_perturbation"].append(rec)

    # --- leave-one-value-out
    dominant = cfg["plithogenic"]["dominant_value"]
    for v in [v["name"] for v in cfg["plithogenic"]["values"]]:
        if v == dominant:
            out["leave_one_value_out"].append({"label": f"without {v}", "removed": v, "status": "skipped",
                                               "reason": "dominant value defines the contradictions"})
            continue
        variant = copy.deepcopy(cfg)
        variant["plithogenic"]["values"] = [x for x in variant["plithogenic"]["values"] if x["name"] != v]
        out["leave_one_value_out"].append(
            _run_variant(f"without {v}", variant, rank_fn_factory, df, base_ranks, top_k, removed=v))

    # --- parameter sweeps
    for name, entries in grid.items():
        recs = []
        for ov in entries:
            label = ", ".join(f"{k}={v}" for k, v in ov.items())
            try:
                variant = apply_overrides(cfg, ov)
            except ValueError as e:
                recs.append({"label": label, "overrides": ov, "status": "error",
                             "error_type": "ValueError", "error": str(e)})
                continue
            recs.append(_run_variant(label, variant, rank_fn_factory, df, base_ranks, top_k, overrides=ov,
                                     identical_to_base=_same_cfg(variant, cfg)))
        out["param_sweeps"][name] = recs
    return to_jsonable(out)
