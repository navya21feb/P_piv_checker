"""
baseline_piv.py -- MEREC weights, PIV ranking and the CLASSICAL baseline (no plithogenic grouping).

Single-valued fuzzy P-PIV project; NOT neutrosophic. Nothing here knows a column name: criteria
are the columns of the matrix you pass in, directions are +1 (benefit) / -1 (cost), and
classical_piv_from_config reads everything else from ppiv_config.json.

Tags
----
[S] MEREC  : Keshavarz-Ghorabaee et al. (2021). Formulas as given in the project spec, item 6.
[S] PIV    : Mufazzal & Muzakkir (2018).        Formulas as given in the project spec, item 7.
[D] tie_decimals (scores rounded before ranking), custom-weight renormalisation, constant-column
    handling and the baseline definition (spec item 8) are team design choices.
Equation / table numbers in either paper: UNVERIFIED (no PDF was available when this was written).

ACCEPTANCE EXAMPLES (see test_baseline_piv.py)
----------------------------------------------
MEREC: 5 alternatives x 4 criteria, C1,C2 benefit, C3,C4 cost -> weights 0.5752, 0.0141, 0.4016,
       0.0091. Expected values were supplied in the task. They are reproduced by the spec formulas
       (0.57522, 0.01410, 0.40156, 0.00912). Source table / equation number in the MEREC paper:
       UNVERIFIED (not confirmed against the PDF).
PIV  : [[1,9,1,3],[9,1,9,1],[8,1,4,5]], all benefit, weights 0.25, vector normalisation ->
       d = 0.452, 0.389, 0.366, ranks 3,2,1. Reproduced by the spec formulas (0.45207, 0.38856,
       0.36649). Source table in the PIV paper: UNVERIFIED (not confirmed against the PDF).
Cost-criterion PIV case: hand-computed in the test file (arithmetic written out there). It is a
       team-made check of the formulas, not a published example.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import schema as sch

VALID_NORMALIZATIONS = sch.VALID_NORMALIZATIONS      # ("vector", "none")


# ---------------------------------------------------------------- input validation helpers
def _as_frame(X) -> pd.DataFrame:
    df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X))
    if df.shape[0] < 1 or df.shape[1] < 1:
        raise ValueError(f"decision matrix must have at least 1 row and 1 column, got {df.shape}.")
    if df.columns.duplicated().any():
        raise ValueError("decision matrix has duplicate column names.")
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            raise ValueError(f"column {c!r} must be numeric (dtype {df[c].dtype}).")
    return df.astype(float)


def _check_finite(df: pd.DataFrame) -> None:
    arr = df.to_numpy()
    if np.isnan(arr).any():
        bad = df.columns[np.isnan(arr).any(axis=0)].tolist()
        raise ValueError(f"NaN found in column(s) {bad}.")
    if np.isinf(arr).any():
        bad = df.columns[np.isinf(arr).any(axis=0)].tolist()
        raise ValueError(f"inf found in column(s) {bad}.")


def _align(columns, obj, what: str) -> pd.Series:
    """obj (Series / dict / list-like) -> Series indexed by columns. Keys must match exactly."""
    if isinstance(obj, (pd.Series, dict)):
        s = pd.Series(obj)
        missing = [c for c in columns if c not in s.index]
        extra = [k for k in s.index if k not in set(columns)]
        if missing or extra:
            raise ValueError(f"{what}: keys must match the matrix columns exactly "
                             f"(missing {missing}, unexpected {extra}).")
        return s.loc[list(columns)]
    arr = obj if isinstance(obj, np.ndarray) else list(obj)       # keep list items as given (True stays bool)
    if getattr(arr, "ndim", 1) != 1 or len(arr) != len(columns):
        raise ValueError(f"{what}: expected {len(columns)} entries, got {len(arr)}.")
    return pd.Series(arr, index=list(columns), dtype=object if isinstance(arr, list) else None)


def _directions(columns, directions) -> np.ndarray:
    s = _align(columns, directions, "directions")
    out = []
    for c, d in s.items():
        if isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d not in (1, -1):
            raise ValueError(f"direction for {c!r} must be +1 (benefit) or -1 (cost), got {d!r}.")
        out.append(int(d))
    return np.array(out, dtype=int)


def _weights(columns, weights) -> tuple:
    """Validate, renormalise to sum 1. Returns (Series, was_renormalised)."""
    s = _align(columns, weights, "weights")
    try:
        w = s.to_numpy(dtype=float)
    except (TypeError, ValueError):
        raise ValueError("weights must be numeric.")
    if not np.isfinite(w).all():
        raise ValueError("weights contain NaN or inf.")
    if (w < 0).any():
        raise ValueError(f"weights must be >= 0; negative for {s.index[w < 0].tolist()}.")
    tot = w.sum()
    if not tot > 0:
        raise ValueError("weights sum to zero.")
    renorm = not np.isclose(tot, 1.0, rtol=0, atol=1e-9)
    return pd.Series(w / tot, index=s.index, name="weight"), renorm


# ---------------------------------------------------------------- MEREC [S]
def merec_weights(X, directions) -> pd.Series:
    """
    MEREC objective weights [S: Keshavarz-Ghorabaee et al. 2021; equation numbers UNVERIFIED].

      benefit (+1): n_ij = min_k x_kj / x_ij        cost (-1): n_ij = x_ij / max_k x_kj
      S_i   = ln(1 + (1/m) * sum_j |ln n_ij|)                  m = NUMBER OF CRITERIA
      S'_ij = ln(1 + (1/m) * sum_{k != j} |ln n_ik|)           still divided by m
      E_j   = sum_i |S'_ij - S_i|          w_j = E_j / sum_k E_k

    Raises ValueError if any x <= 0, NaN or inf, if directions are not +/-1, or if sum E = 0
    (e.g. every column constant). A constant column is handled explicitly: n_ij = 1, ln n = 0,
    so it contributes nothing and gets weight 0. [D] It is listed in result.attrs['constant_columns']
    and a UserWarning is issued.
    The returned Series carries attrs: constant_columns, E (per criterion), S (per alternative).
    """
    df = _as_frame(X)
    _check_finite(df)
    if (df.to_numpy() <= 0).any():
        bad = df.columns[(df.to_numpy() <= 0).any(axis=0)].tolist()
        raise ValueError(f"MEREC needs strictly positive values (x > 0); column(s) {bad} have x <= 0.")
    dirs = _directions(df.columns, directions)
    x = df.to_numpy()
    n, m = x.shape
    cmin, cmax = x.min(axis=0), x.max(axis=0)
    const = cmax == cmin

    N = np.where(dirs == 1, cmin / x, x / cmax)
    N[:, const] = 1.0                                   # explicit: ln 1 = 0
    L = np.abs(np.log(N))
    S = np.log1p(L.sum(axis=1) / m)
    E = np.empty(m)
    for j in range(m):
        S_wo_j = np.log1p(np.delete(L, j, axis=1).sum(axis=1) / m)    # criterion j removed, /m kept
        E[j] = np.abs(S_wo_j - S).sum()
    total = E.sum()
    if not (np.isfinite(total) and total > 0):
        raise ValueError("MEREC: sum of E_j is zero (all alternatives identical on every criterion "
                         "or only one distinguishing criterion): weights are undefined.")
    w = pd.Series(E / total, index=df.columns, name="merec_weight")
    constant_cols = df.columns[const].tolist()
    w.attrs = {"constant_columns": constant_cols,
               "E": dict(zip(df.columns.tolist(), E.tolist())),
               "S": dict(zip(df.index.tolist(), S.tolist()))}
    if constant_cols:
        warnings.warn(f"MEREC: constant column(s) {constant_cols} get weight 0 (n_ij = 1).", UserWarning)
    return w


# ---------------------------------------------------------------- PIV [S]
def piv_rank(X, weights, directions, normalization: str = "vector", tie_decimals: int = 12) -> dict:
    """
    Proximity Indexed Value ranking [S: Mufazzal & Muzakkir 2018; table/equation numbers UNVERIFIED].

      vector : r_ij = x_ij / sqrt(sum_i x_ij^2)        (a zero-norm column gives r = 0; reported)
      none   : r_ij = x_ij                             ([D] sensitivity option)
      v_ij = w_j * r_ij
      benefit: u_ij = max_i v_ij - v_ij        cost: u_ij = v_ij - min_i v_ij
      d_i = sum_j u_ij                         rank ascending, smallest d = rank 1, method='min'

    weights: validated (>= 0, finite, keys match columns) and renormalised to sum 1 [D]; the
    renormalised Series is returned. [D] Scores are rounded to `tie_decimals` before ranking so
    floating-point noise cannot split equal scores; the unrounded scores are returned.
    Returns dict: scores, ranks, normalized_matrix, weighted_matrix, weights, directions, meta.
    """
    if normalization not in VALID_NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {VALID_NORMALIZATIONS}, got {normalization!r}.")
    df = _as_frame(X)
    _check_finite(df)
    dirs = _directions(df.columns, directions)
    w, renorm = _weights(df.columns, weights)
    x = df.to_numpy()

    zero_norm = []
    if normalization == "vector":
        norm = np.sqrt((x ** 2).sum(axis=0))
        zmask = norm == 0
        zero_norm = df.columns[zmask].tolist()
        r = x / np.where(zmask, 1.0, norm)               # zero-norm column: x is all 0 -> r = 0
    else:
        r = x.copy()
    v = r * w.to_numpy()
    u = np.where(dirs == 1, v.max(axis=0) - v, v - v.min(axis=0))
    d = u.sum(axis=1)
    scores = pd.Series(d, index=df.index, name="piv_score")
    ranks = pd.Series(np.round(d, tie_decimals), index=df.index).rank(method="min").astype(int)
    ranks.name = "rank"
    return {
        "scores": scores,
        "ranks": ranks,
        "normalized_matrix": pd.DataFrame(r, index=df.index, columns=df.columns),
        "weighted_matrix": pd.DataFrame(v, index=df.index, columns=df.columns),
        "weights": w,
        "directions": pd.Series(dirs, index=df.columns, name="direction"),
        "meta": {"normalization": normalization, "weights_renormalised": bool(renorm),
                 "zero_norm_columns": zero_norm, "tie_decimals": int(tie_decimals)},
    }


# ---------------------------------------------------------------- classical baseline [D]
def classical_columns(cfg: dict) -> tuple:
    """(columns, directions Series) of the classical baseline: every source column of every value
    as its own criterion, in config order. Raw columns unless baseline.use_adjusted_columns."""
    use_adj = cfg["baseline"]["use_adjusted_columns"]
    srcs = sch.all_sources(cfg)
    cols = [s["effective_column"] if use_adj else s["column"] for s in srcs]
    return cols, pd.Series([s["direction"] for s in srcs], index=cols, name="direction")


def classical_piv_from_config(df: pd.DataFrame, cfg: dict, weights=None) -> dict:
    """
    [D] Classical baseline (spec item 8): each source column is a SEPARATE criterion on its RAW
    value (no transform, no appurtenance, no plithogenic grouping, no contradiction), with the
    direction from the config. weights=None -> MEREC on the raw positive values of `df` (i.e. on
    whatever rows are passed, so subsets recompute weights); otherwise custom/frozen weights keyed
    by column (validated, renormalised). Then piv_rank with piv.normalization / piv.tie_decimals.
    Returns piv_rank's dict plus: columns, weights_source ('merec'|'custom'), constant_columns.
    """
    cols, dirs = classical_columns(cfg)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"classical baseline: columns missing from the data: {missing}")
    X = df[cols]
    if weights is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)    # constant columns are reported in the result
            w = merec_weights(X, dirs)
        source, const = "merec", w.attrs["constant_columns"]
    else:
        w, source, const = weights, "custom", []
    out = piv_rank(X, w, dirs, normalization=cfg["piv"]["normalization"],
                   tie_decimals=cfg["piv"]["tie_decimals"])
    out["columns"] = cols
    out["weights_source"] = source
    out["constant_columns"] = const
    return out
