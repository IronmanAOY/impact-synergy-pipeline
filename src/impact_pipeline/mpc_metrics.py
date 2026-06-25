import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix, run_glm
from nilearn.glm.first_level import glover_hrf
import pandas as pd
import warnings
import itertools
import json
import os
import hashlib
import time
import math
import atexit
import tempfile
import shutil
import concurrent.futures
import sqlite3
from collections import OrderedDict
from multiprocessing import shared_memory
from scipy.stats import entropy
from scipy.stats import median_abs_deviation
from scipy.signal import butter, sosfiltfilt, hilbert
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_regression
import logging

from impact_pipeline.hardware_backend import (
    accelerated_corrcoef,
    accelerated_dot,
    accelerated_pinv_dot,
    accelerated_psd_invsqrt,
    accelerated_row_norm,
    accelerated_solve,
    accelerated_svd_values,
    accelerated_zscore,
    get_array_module,
    resolve_hardware_backend,
    to_numpy,
)

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def _wrap(fn):
            return fn
        return _wrap

log = logging.getLogger(__name__)


class _IIMDiskKernelCache:
    """
    Disk-backed cache for cut-kernel values keyed by:
      (direction, m_mask, m_key, z_mask, induced_partition_key)

    Uses a small in-memory LRU front-cache plus SQLite persistence with
    batched writes to keep RAM bounded while allowing cache reuse across cuts
    and resumes.
    """

    def __init__(
        self,
        path: str,
        signature: dict | None = None,
        memory_entries: int = 300_000,
        flush_batch: int = 5_000,
    ):
        self.path = str(path)
        self.memory_entries = max(10_000, int(memory_entries))
        self.flush_batch = max(100, int(flush_batch))
        self._mem = OrderedDict()
        self._pending = {}
        self.hits_mem = 0
        self.hits_disk = 0
        self.misses = 0
        self.writes = 0

        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.conn = sqlite3.connect(self.path, timeout=60.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA temp_store=MEMORY")
        self.conn.execute("PRAGMA cache_size=-100000")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kernel_cache (
                d INTEGER NOT NULL,
                m_mask INTEGER NOT NULL,
                m_key INTEGER NOT NULL,
                z_mask INTEGER NOT NULL,
                pi INTEGER NOT NULL,
                v REAL NOT NULL,
                PRIMARY KEY (d, m_mask, m_key, z_mask, pi)
            )
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                k TEXT PRIMARY KEY,
                v TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

        if signature is not None:
            sig_txt = json.dumps(signature, sort_keys=True, ensure_ascii=True)
            row = self.conn.execute("SELECT v FROM meta WHERE k='signature'").fetchone()
            old_sig = None if row is None else str(row[0])
            if old_sig is not None and old_sig != sig_txt:
                self.conn.execute("DELETE FROM kernel_cache")
                self.conn.execute("DELETE FROM meta WHERE k='signature'")
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta(k, v) VALUES ('signature', ?)",
                    (sig_txt,),
                )
                self.conn.commit()
            elif old_sig is None:
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta(k, v) VALUES ('signature', ?)",
                    (sig_txt,),
                )
                self.conn.commit()

    def _touch_mem(self, key, val):
        self._mem[key] = float(val)
        self._mem.move_to_end(key, last=True)
        while len(self._mem) > self.memory_entries:
            self._mem.popitem(last=False)

    def get(self, key):
        if key in self._mem:
            self.hits_mem += 1
            v = self._mem[key]
            self._mem.move_to_end(key, last=True)
            return float(v)

        if key in self._pending:
            self.hits_mem += 1
            v = float(self._pending[key])
            self._touch_mem(key, v)
            return v

        row = None
        for attempt in range(6):
            try:
                row = self.conn.execute(
                    """
                    SELECT v FROM kernel_cache
                    WHERE d=? AND m_mask=? AND m_key=? AND z_mask=? AND pi=?
                    """,
                    (int(key[0]), int(key[1]), int(key[2]), int(key[3]), int(key[4])),
                ).fetchone()
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt >= 5:
                    raise
                time.sleep(0.01 * (2 ** attempt))
        if row is None:
            self.misses += 1
            return None
        v = float(row[0])
        self.hits_disk += 1
        self._touch_mem(key, v)
        return v

    def set(self, key, value):
        v = float(value)
        self._touch_mem(key, v)
        self._pending[key] = v
        if len(self._pending) >= self.flush_batch:
            self.flush()

    def flush(self):
        if not self._pending:
            return
        rows = [
            (int(k[0]), int(k[1]), int(k[2]), int(k[3]), int(k[4]), float(v))
            for k, v in self._pending.items()
        ]
        for attempt in range(6):
            try:
                self.conn.executemany(
                    """
                    INSERT OR REPLACE INTO kernel_cache(d, m_mask, m_key, z_mask, pi, v)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                self.conn.commit()
                break
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() or attempt >= 5:
                    raise
                time.sleep(0.01 * (2 ** attempt))
        self.writes += int(len(rows))
        self._pending.clear()

    def stats(self):
        return {
            "hits_mem": int(self.hits_mem),
            "hits_disk": int(self.hits_disk),
            "misses": int(self.misses),
            "writes": int(self.writes),
            "path": self.path,
        }

    def close(self):
        try:
            self.flush()
        finally:
            try:
                self.conn.close()
            except Exception:
                pass


@njit(cache=True)
def _build_partition_maps_numba(base, z_len, posA, posB):
    nZ = 1
    for _ in range(z_len):
        nZ *= base
    mapA = np.empty(nZ, dtype=np.int64)
    mapB = np.empty(nZ, dtype=np.int64)
    zv = np.empty(z_len, dtype=np.int64)

    nA = int(posA.shape[0])
    nB = int(posB.shape[0])
    for kz in range(nZ):
        v = kz
        for i in range(z_len - 1, -1, -1):
            zv[i] = v % base
            v //= base

        keyA = 0
        for i in range(nA):
            keyA = keyA * base + int(zv[int(posA[i])])
        mapA[kz] = keyA

        keyB = 0
        for i in range(nB):
            keyB = keyB * base + int(zv[int(posB[i])])
        mapB[kz] = keyB
    return mapA, mapB


@njit(cache=True)
def _compose_product_distribution_numba(pA, pB, mapA, mapB):
    n = int(mapA.shape[0])
    q = np.empty(n, dtype=np.float64)
    s = 0.0
    for i in range(n):
        v = float(pA[int(mapA[i])]) * float(pB[int(mapB[i])])
        q[i] = v
        s += v
    if s <= 0.0:
        inv = 1.0 / float(max(1, n))
        for i in range(n):
            q[i] = inv
    else:
        inv = 1.0 / s
        for i in range(n):
            q[i] *= inv
    return q


@njit(cache=True)
def _jsd_numba(p, q):
    n = int(p.shape[0])
    sp = 0.0
    sq = 0.0
    for i in range(n):
        sp += float(p[i])
        sq += float(q[i])
    if sp <= 0.0 or sq <= 0.0:
        return 0.0

    kl_pm = 0.0
    kl_qm = 0.0
    for i in range(n):
        pi = float(p[i]) / sp
        qi = float(q[i]) / sq
        mi = 0.5 * (pi + qi)
        if pi > 0.0 and mi > 0.0:
            kl_pm += pi * (np.log(pi / mi) / np.log(2.0))
        if qi > 0.0 and mi > 0.0:
            kl_qm += qi * (np.log(qi / mi) / np.log(2.0))
    return 0.5 * (kl_pm + kl_qm)


_IIM_PHASE1_CTX = None


class _IIMKernelCacheMissError(RuntimeError):
    """Raised when lookup-only IIM aggregation hits a missing kernel cache key."""


def _iim_encode_vals(vals, base):
    key = 0
    for v in vals:
        key = key * base + int(v)
    return int(key)


def _iim_decode_key(key, k, base):
    out = [0] * k
    v = int(key)
    for i in range(k - 1, -1, -1):
        out[i] = v % base
        v //= base
    return tuple(out)


def _iim_subset_key_matrix(states, subset, base):
    if len(subset) == 0:
        return np.zeros(states.shape[0], dtype=np.int64)
    cols = states[:, subset].astype(np.int64, copy=False)
    mult = (base ** np.arange(len(subset) - 1, -1, -1, dtype=np.int64))
    return (cols * mult).sum(axis=1).astype(np.int64)


def _iim_enumerate_bipartitions(nodes):
    n = len(nodes)
    if n < 2:
        return []
    out = []
    node_set = tuple(nodes)
    for k in range(1, (n // 2) + 1):
        for A in itertools.combinations(node_set, k):
            A = tuple(A)
            B = tuple(x for x in node_set if x not in A)
            if k == n - k and A[0] > B[0]:
                continue
            out.append((A, B))
    return out


def _iim_marginalize(dist_full, keys, n_keys):
    p = np.bincount(keys, weights=dist_full, minlength=n_keys).astype(float)
    s = p.sum()
    if s <= 0:
        return np.ones(n_keys, dtype=float) / float(n_keys)
    return p / s


def _iim_enumerate_subsets(nodes, max_size):
    out = []
    for r in range(1, min(int(max_size), len(nodes)) + 1):
        out.extend(tuple(c) for c in itertools.combinations(nodes, r))
    return out


def _iim_discretize_per_node(arr, base_bins):
    n, t = arr.shape
    out = np.zeros((n, t), dtype=np.int16)
    for i in range(n):
        x = arr[i]
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            continue
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if np.isclose(vmin, vmax):
            continue
        edges = np.quantile(finite, np.linspace(0.0, 1.0, int(base_bins) + 1))
        if np.unique(edges).size != int(base_bins) + 1:
            edges = np.linspace(vmin, vmax, int(base_bins) + 1)
        bins_inner = edges[1:-1]
        z = np.digitize(x, bins_inner, right=False)
        z = np.clip(z, 0, int(base_bins) - 1)
        z[~np.isfinite(x)] = 0
        out[i] = z.astype(np.int16)
    return out


def _iim_select_nodes(arr, max_n):
    n = int(arr.shape[0])
    if n <= int(max_n):
        return np.arange(n, dtype=int)
    var = np.nanvar(arr, axis=1)
    var = np.where(np.isfinite(var), var, -np.inf)
    idx = np.argsort(var)[::-1][: int(max_n)]
    return np.sort(idx.astype(int))


def _iim_build_states_and_tpm(disc, base, lag, alpha, hardware_backend=None):
    n, t = disc.shape
    t_eff = int(t - lag)
    curr = disc[:, :t_eff].T.astype(np.int16, copy=False)
    nxt = disc[:, lag:].T.astype(np.int16, copy=False)

    full_subset = tuple(range(int(n)))
    curr_keys = _iim_subset_key_matrix(curr, full_subset, int(base))
    nxt_keys = _iim_subset_key_matrix(nxt, full_subset, int(base))

    n_states = int(int(base) ** int(n))
    backend = resolve_hardware_backend(hardware_backend)
    if backend.accelerator:
        xp = get_array_module(backend)
        counts_d = xp.zeros((n_states, n_states), dtype=xp.float64)
        xp.add.at(counts_d, (xp.asarray(curr_keys), xp.asarray(nxt_keys)), 1.0)
        row_sum_d = counts_d.sum(axis=1, keepdims=True)
        tpm = to_numpy((counts_d + float(alpha)) / (row_sum_d + float(alpha) * n_states))
    else:
        counts = np.zeros((n_states, n_states), dtype=float)
        np.add.at(counts, (curr_keys, nxt_keys), 1.0)
        row_sum = counts.sum(axis=1, keepdims=True)
        tpm = (counts + float(alpha)) / (row_sum + float(alpha) * n_states)

    sid = np.arange(n_states, dtype=np.int64)[:, None]
    powv = (int(base) ** np.arange(n - 1, -1, -1, dtype=np.int64))[None, :]
    states_full = ((sid // powv) % int(base)).astype(np.int16)
    return curr, tpm, states_full


def _iim_build_phase1_chunks_adaptive(
    remaining_mechanisms,
    max_chunk_size,
    workers,
):
    mechs = tuple(remaining_mechanisms)
    if not mechs:
        return []
    max_chunk = max(1, int(max_chunk_size))
    workers_eff = max(1, int(workers))
    if workers_eff <= 1:
        return [
            tuple(mechs[i:i + max_chunk])
            for i in range(0, len(mechs), max_chunk)
        ]

    chunks = []
    idx = 0
    n_total = int(len(mechs))
    while idx < n_total:
        rem = int(n_total - idx)
        active_workers = max(1, min(workers_eff, rem))

        progress = float(idx) / float(max(1, n_total))
        if progress >= 0.80:
            stage_target = 1
        elif progress >= 0.50:
            stage_target = 2
        else:
            stage_target = 4

        stage_target = min(int(stage_target), int(max_chunk))
        c = min(stage_target, max(1, rem // active_workers))
        c = max(1, min(int(c), rem))
        chunks.append(tuple(mechs[idx:idx + c]))
        idx += c
    return chunks


def _iim_build_cut_tpm(tpm, states_full, base, A, B, hardware_backend=None):
    A = tuple(sorted(A))
    B = tuple(sorted(B))
    n_states = int(states_full.shape[0])
    keyA = _iim_subset_key_matrix(states_full, A, int(base))
    keyB = _iim_subset_key_matrix(states_full, B, int(base))
    nA = int(int(base) ** len(A))
    nB = int(int(base) ** len(B))

    rowsA = [np.where(keyA == k)[0] for k in range(nA)]
    rowsB = [np.where(keyB == k)[0] for k in range(nB)]

    backend = resolve_hardware_backend(hardware_backend)
    if backend.accelerator:
        xp = get_array_module(backend)
        tpm_d = xp.asarray(tpm, dtype=xp.float64)
        keyA_d = xp.asarray(keyA)
        keyB_d = xp.asarray(keyB)

        def _marginalize_d(dist_full_d, keys_d, n_keys):
            p_d = xp.bincount(keys_d, weights=dist_full_d, minlength=int(n_keys)).astype(xp.float64)
            s_d = p_d.sum()
            if float(to_numpy(s_d)) <= 0:
                return xp.ones(int(n_keys), dtype=xp.float64) / float(n_keys)
            return p_d / s_d

        pA_next_d = xp.zeros((nA, nA), dtype=xp.float64)
        pB_next_d = xp.zeros((nB, nB), dtype=xp.float64)
        for ka in range(nA):
            rr = rowsA[ka]
            if rr.size == 0:
                pA_next_d[ka, :] = 1.0 / float(nA)
            else:
                dnext_d = tpm_d[xp.asarray(rr), :].mean(axis=0)
                pA_next_d[ka, :] = _marginalize_d(dnext_d, keyA_d, nA)

        for kb in range(nB):
            rr = rowsB[kb]
            if rr.size == 0:
                pB_next_d[kb, :] = 1.0 / float(nB)
            else:
                dnext_d = tpm_d[xp.asarray(rr), :].mean(axis=0)
                pB_next_d[kb, :] = _marginalize_d(dnext_d, keyB_d, nB)

        tpm_cut_d = pA_next_d[keyA_d][:, keyA_d] * pB_next_d[keyB_d][:, keyB_d]
        row_sums_d = tpm_cut_d.sum(axis=1, keepdims=True)
        bad_d = row_sums_d[:, 0] <= 0
        if bool(to_numpy((~bad_d).any())):
            tpm_cut_d[~bad_d, :] = tpm_cut_d[~bad_d, :] / row_sums_d[~bad_d]
        if bool(to_numpy(bad_d.any())):
            tpm_cut_d[bad_d, :] = 1.0 / float(n_states)
        return to_numpy(tpm_cut_d)

    pA_next = np.zeros((nA, nA), dtype=float)
    pB_next = np.zeros((nB, nB), dtype=float)
    for ka in range(nA):
        rr = rowsA[ka]
        if rr.size == 0:
            pA_next[ka, :] = 1.0 / float(nA)
        else:
            dnext = tpm[rr, :].mean(axis=0)
            pA_next[ka, :] = _iim_marginalize(dnext, keyA, nA)

    for kb in range(nB):
        rr = rowsB[kb]
        if rr.size == 0:
            pB_next[kb, :] = 1.0 / float(nB)
        else:
            dnext = tpm[rr, :].mean(axis=0)
            pB_next[kb, :] = _iim_marginalize(dnext, keyB, nB)

    tpm_cut = pA_next[keyA][:, keyA] * pB_next[keyB][:, keyB]
    row_sums = tpm_cut.sum(axis=1, keepdims=True)
    bad = row_sums[:, 0] <= 0
    if np.any(~bad):
        tpm_cut[~bad, :] = tpm_cut[~bad, :] / row_sums[~bad]
    if np.any(bad):
        tpm_cut[bad, :] = 1.0 / float(n_states)
    return tpm_cut


def _iim_all_system_cuts(n_nodes_sys, part_mode):
    nodes = tuple(range(int(n_nodes_sys)))
    all_cuts = _iim_enumerate_bipartitions(nodes)
    if str(part_mode) == "balanced":
        out = []
        for A, B in all_cuts:
            if abs(len(A) - len(B)) <= 1:
                out.append((A, B))
        return out
    return all_cuts


def _iim_cut_to_key(A, B):
    return f"{','.join(map(str, A))}|{','.join(map(str, B))}"


def _iim_cut_from_payload(payload):
    if (
        isinstance(payload, (list, tuple))
        and len(payload) == 2
        and isinstance(payload[0], (list, tuple))
        and isinstance(payload[1], (list, tuple))
    ):
        return (tuple(int(x) for x in payload[0]), tuple(int(x) for x in payload[1]))
    return None


def prepare_iim_problem(
    ts: np.ndarray,
    *,
    bins: int = 3,
    lag_trs: int = 1,
    n_parts: int | None = None,
    rng: int | np.random.RandomState = 0,
    partition_mode: str = "all",
    max_nodes: int | None = None,
    max_mechanism_size: int | None = None,
    max_purview_size: int | None = None,
    tpm_alpha: float = 1e-3,
    max_state_space: int = 1500,
    hardware_backend=None,
):
    if ts.ndim != 2:
        raise ValueError(f"ts should be 2D (n_regions × n_time), got shape {ts.shape}")
    if partition_mode not in {"all", "balanced"}:
        raise ValueError("partition_mode must be 'all' or 'balanced'")
    if bins < 2:
        raise ValueError("bins must be >= 2")
    if lag_trs < 1:
        raise ValueError("lag_trs must be >= 1")
    if n_parts is not None and int(n_parts) < 1:
        raise ValueError("n_parts must be >= 1 or None for exhaustive search")
    if max_nodes is not None and int(max_nodes) < 2:
        raise ValueError("max_nodes must be >= 2 or None")
    if max_mechanism_size is not None and int(max_mechanism_size) < 1:
        raise ValueError("max_mechanism_size must be >= 1 or None")
    if max_purview_size is not None and int(max_purview_size) < 1:
        raise ValueError("max_purview_size must be >= 1 or None")
    if tpm_alpha <= 0:
        raise ValueError("tpm_alpha must be > 0")
    if max_state_space < 16:
        raise ValueError("max_state_space must be >= 16")

    n_regions, n_time = ts.shape
    if n_regions < 2 or n_time - int(lag_trs) < 1:
        return {
            "defined": False,
            "undefined_reason": "insufficient_shape",
            "n_regions_input": int(n_regions),
            "n_time_input": int(n_time),
        }

    if isinstance(rng, np.random.RandomState):
        rand_state = rng
    else:
        rand_state = np.random.RandomState(rng)

    use_nodes = int(n_regions) if max_nodes is None else min(int(max_nodes), int(n_regions))
    selected = _iim_select_nodes(ts, use_nodes)
    ts_sel = np.asarray(ts[selected, :], dtype=float)
    eff_bins = int(bins)
    n_sel = int(ts_sel.shape[0])
    while n_sel >= 2 and (int(eff_bins) ** int(n_sel)) > int(max_state_space):
        if eff_bins > 2:
            eff_bins -= 1
            continue
        n_sel -= 1
        selected = _iim_select_nodes(ts, n_sel)
        ts_sel = np.asarray(ts[selected, :], dtype=float)
    if n_sel < 2:
        return {
            "defined": False,
            "undefined_reason": "state_space_too_large",
            "n_regions_input": int(n_regions),
            "n_time_input": int(n_time),
            "n_nodes_used": int(n_sel),
            "bins_used": int(eff_bins),
        }

    mech_size_eff = (
        int(n_sel)
        if max_mechanism_size is None
        else int(min(int(max_mechanism_size), int(n_sel)))
    )
    purv_size_eff = (
        int(n_sel)
        if max_purview_size is None
        else int(min(int(max_purview_size), int(n_sel)))
    )
    all_nodes = tuple(range(int(n_sel)))
    mechanisms_all = tuple(_iim_enumerate_subsets(all_nodes, mech_size_eff))
    purviews_all = tuple(_iim_enumerate_subsets(all_nodes, purv_size_eff))
    if not mechanisms_all or not purviews_all:
        return {
            "defined": False,
            "undefined_reason": "no_valid_mechanisms_or_purviews",
            "n_regions_input": int(n_regions),
            "n_time_input": int(n_time),
            "n_nodes_used": int(n_sel),
            "bins_used": int(eff_bins),
            "max_mechanism_size_used": int(mech_size_eff),
            "max_purview_size_used": int(purv_size_eff),
        }

    cuts = _iim_all_system_cuts(n_sel, partition_mode)
    if not cuts:
        return {
            "defined": False,
            "undefined_reason": "no_valid_cuts",
            "n_regions_input": int(n_regions),
            "n_time_input": int(n_time),
            "n_nodes_used": int(n_sel),
            "bins_used": int(eff_bins),
            "max_mechanism_size_used": int(mech_size_eff),
            "max_purview_size_used": int(purv_size_eff),
        }
    if n_parts is None or int(n_parts) >= len(cuts):
        cuts_eval = tuple(cuts)
    else:
        idx = rand_state.choice(len(cuts), size=int(n_parts), replace=False)
        cuts_eval = tuple(cuts[int(i)] for i in idx)

    disc = _iim_discretize_per_node(ts_sel, eff_bins)
    curr_obs, tpm_full, states_full = _iim_build_states_and_tpm(
        disc,
        eff_bins,
        int(lag_trs),
        float(tpm_alpha),
        hardware_backend=hardware_backend,
    )

    return {
        "defined": True,
        "undefined_reason": None,
        "n_regions_input": int(n_regions),
        "n_time_input": int(n_time),
        "selected_nodes": tuple(int(x) for x in selected.tolist()),
        "ts_selected": ts_sel,
        "disc": disc,
        "curr_obs": curr_obs,
        "tpm_full": tpm_full,
        "states_full": states_full,
        "bins_used": int(eff_bins),
        "lag_trs": int(lag_trs),
        "n_nodes_used": int(n_sel),
        "max_nodes_requested": (None if max_nodes is None else int(max_nodes)),
        "max_mechanism_size_used": int(mech_size_eff),
        "max_purview_size_used": int(purv_size_eff),
        "n_parts_requested": (None if n_parts is None else int(n_parts)),
        "mechanisms_all": mechanisms_all,
        "purviews_all": purviews_all,
        "cuts_eval": cuts_eval,
        "cuts_payload": tuple((list(A), list(B)) for A, B in cuts_eval),
        "partition_mode": str(partition_mode),
        "tpm_alpha": float(tpm_alpha),
        "max_state_space": int(max_state_space),
    }


def _iim_phase1_chunk_contribution(
    mechanisms_chunk,
    purviews,
    base,
    tpm,
    curr_obs,
    states_full,
    static_cache=None,
    obs_state_cache=None,
    kernel_cache=None,
    cut_mask_a=None,
    use_induced_partition_cache=False,
    kernel_cache_lookup_only=False,
):
    n_states = int(states_full.shape[0])
    if static_cache is None:
        static_cache = {}
    static_sig = (int(base), int(states_full.shape[0]), int(states_full.shape[1]))
    if static_cache.get("_sig") != static_sig:
        static_cache.clear()
        static_cache["_sig"] = static_sig
    subset_cache = static_cache.setdefault("subset_cache", {})
    map_cache = static_cache.setdefault("map_cache", {})
    bip_cache = static_cache.setdefault("bip_cache", {})
    subset_mask_cache = static_cache.setdefault("subset_mask_cache", {})

    if obs_state_cache is None:
        obs_state_cache = {}
    obs_sig = (
        int(base),
        int(curr_obs.shape[0]),
        int(curr_obs.shape[1]),
    )
    if obs_state_cache.get("_sig") != obs_sig:
        obs_state_cache.clear()
        obs_state_cache["_sig"] = obs_sig

    # tpm-specific caches: valid only for this call
    effect_cache = {}
    cause_cache = {}

    def _subset_mask(subset):
        subset = tuple(sorted(subset))
        if subset in subset_mask_cache:
            return int(subset_mask_cache[subset])
        mask = 0
        for nn in subset:
            mask |= (1 << int(nn))
        subset_mask_cache[subset] = int(mask)
        return int(mask)

    def _induced_partition_key(m_mask, z_mask):
        if cut_mask_a is None:
            return 0
        u_mask = int(m_mask) | int(z_mask)
        part_a = int(u_mask & int(cut_mask_a))
        part_b = int(u_mask ^ part_a)
        if part_a == 0 or part_b == 0:
            return 0
        return int(part_a if part_a < part_b else part_b)

    def _get_subset_cache(subset):
        subset = tuple(sorted(subset))
        if subset in subset_cache:
            return subset_cache[subset]
        keys = _iim_subset_key_matrix(states_full, subset, base)
        n_keys = int(base ** len(subset))
        rows_by_key = [np.where(keys == k)[0] for k in range(n_keys)]
        rec = {
            "subset": subset,
            "keys": keys,
            "n_keys": n_keys,
            "rows_by_key": rows_by_key,
        }
        subset_cache[subset] = rec
        return rec

    def _partition_maps(Z, ZA, ZB):
        key = (tuple(Z), tuple(ZA), tuple(ZB))
        if key in map_cache:
            return map_cache[key]
        Z = tuple(Z)
        ZA = tuple(ZA)
        ZB = tuple(ZB)
        posA = [Z.index(x) for x in ZA]
        posB = [Z.index(x) for x in ZB]
        if NUMBA_AVAILABLE:
            posA_arr = np.asarray(posA, dtype=np.int64)
            posB_arr = np.asarray(posB, dtype=np.int64)
            mapA, mapB = _build_partition_maps_numba(
                int(base),
                int(len(Z)),
                posA_arr,
                posB_arr,
            )
        else:
            nZ = int(base ** len(Z))
            mapA = np.zeros(nZ, dtype=np.int64)
            mapB = np.zeros(nZ, dtype=np.int64)
            for kz in range(nZ):
                zv = _iim_decode_key(kz, len(Z), base)
                mapA[kz] = _iim_encode_vals([zv[p] for p in posA], base)
                mapB[kz] = _iim_encode_vals([zv[p] for p in posB], base)
        map_cache[key] = (mapA, mapB)
        return mapA, mapB

    def _effect(M, m_key, Z):
        M = tuple(sorted(M))
        Z = tuple(sorted(Z))
        key = (M, int(m_key), Z)
        if key in effect_cache:
            return effect_cache[key]
        cM = _get_subset_cache(M)
        cZ = _get_subset_cache(Z)
        rows = cM["rows_by_key"][int(m_key)]
        if rows.size == 0:
            p = np.ones(cZ["n_keys"], dtype=float) / float(cZ["n_keys"])
        else:
            d_next = tpm[rows, :].mean(axis=0)
            p = _iim_marginalize(d_next, cZ["keys"], cZ["n_keys"])
        effect_cache[key] = p
        return p

    def _cause(M, m_key, Z):
        M = tuple(sorted(M))
        Z = tuple(sorted(Z))
        key = (M, int(m_key), Z)
        if key in cause_cache:
            return cause_cache[key]
        cM = _get_subset_cache(M)
        cZ = _get_subset_cache(Z)
        cols = cM["rows_by_key"][int(m_key)]
        if cols.size == 0:
            p = np.ones(cZ["n_keys"], dtype=float) / float(cZ["n_keys"])
        else:
            likelihood = tpm[:, cols].sum(axis=1)
            s = float(np.sum(likelihood))
            if s <= 0:
                post_prev = np.ones(n_states, dtype=float) / float(n_states)
            else:
                post_prev = likelihood / s
            p = _iim_marginalize(post_prev, cZ["keys"], cZ["n_keys"])
        cause_cache[key] = p
        return p

    def _min_partition_divergence(M, m_key, Z, direction):
        M = tuple(sorted(M))
        Z = tuple(sorted(Z))
        if len(M) < 2 or len(Z) < 2:
            return 0.0
        if bool(use_induced_partition_cache) and (kernel_cache is not None):
            m_mask = _subset_mask(M)
            z_mask = _subset_mask(Z)
            pi_key = _induced_partition_key(m_mask, z_mask)
            d_key = 0 if direction == "effect" else 1
            cache_key = (int(d_key), int(m_mask), int(m_key), int(z_mask), int(pi_key))
            v_cached = kernel_cache.get(cache_key)
            if v_cached is not None:
                return float(v_cached)
            if bool(kernel_cache_lookup_only):
                raise _IIMKernelCacheMissError(
                    f"Missing kernel cache key: d={d_key}, m_mask={m_mask}, m_key={int(m_key)}, z_mask={z_mask}, pi={pi_key}"
                )
        if M not in bip_cache:
            bip_cache[M] = _iim_enumerate_bipartitions(M)
        if Z not in bip_cache:
            bip_cache[Z] = _iim_enumerate_bipartitions(Z)
        m_parts = bip_cache[M]
        z_parts = bip_cache[Z]
        if not m_parts or not z_parts:
            return 0.0
        m_vals = _iim_decode_key(int(m_key), len(M), base)
        posM = {node: i for i, node in enumerate(M)}
        p_full = _effect(M, m_key, Z) if direction == "effect" else _cause(M, m_key, Z)
        best = np.inf
        nZ = int(base ** len(Z))
        for MA, MB in m_parts:
            keyA = _iim_encode_vals([m_vals[posM[nn]] for nn in MA], base)
            keyB = _iim_encode_vals([m_vals[posM[nn]] for nn in MB], base)
            for ZA, ZB in z_parts:
                pA = _effect(MA, keyA, ZA) if direction == "effect" else _cause(MA, keyA, ZA)
                pB = _effect(MB, keyB, ZB) if direction == "effect" else _cause(MB, keyB, ZB)
                mapA, mapB = _partition_maps(Z, ZA, ZB)
                if NUMBA_AVAILABLE:
                    q = _compose_product_distribution_numba(pA, pB, mapA, mapB)
                    d = float(_jsd_numba(p_full, q))
                else:
                    q = pA[mapA] * pB[mapB]
                    q_sum = float(np.sum(q))
                    if q_sum <= 0:
                        q = np.ones(nZ, dtype=float) / float(nZ)
                    else:
                        q = q / q_sum
                    p_full_n = p_full / max(float(np.sum(p_full)), 1e-12)
                    d = float(0.5 * entropy(p_full_n, 0.5 * (p_full_n + q), base=2) + 0.5 * entropy(q, 0.5 * (p_full_n + q), base=2))
                if d < best:
                    best = d
        out = float(best) if np.isfinite(best) else 0.0
        if bool(use_induced_partition_cache) and (kernel_cache is not None):
            kernel_cache.set(cache_key, out)
        return out

    def _get_mechanism_state_freq(M):
        M = tuple(sorted(M))
        cached = obs_state_cache.get(M)
        if cached is not None:
            return cached
        obs_keys = _iim_subset_key_matrix(curr_obs, M, base)
        if obs_keys.size == 0:
            uk = np.asarray([], dtype=np.int64)
            wt = np.asarray([], dtype=float)
        else:
            uk, cnt = np.unique(obs_keys, return_counts=True)
            wt = cnt.astype(float) / float(cnt.sum())
            uk = uk.astype(np.int64, copy=False)
            wt = wt.astype(float, copy=False)
        obs_state_cache[M] = (uk, wt)
        return uk, wt

    psi_terms = []
    for M in mechanisms_chunk:
        M = tuple(sorted(M))
        uk, wt = _get_mechanism_state_freq(M)
        if uk.size == 0:
            continue
        phi_M_terms = []
        for m_key, w_m in zip(uk, wt):
            phi_e = 0.0
            phi_c = 0.0
            for Z in purviews:
                d_e = _min_partition_divergence(M, int(m_key), Z, "effect")
                d_c = _min_partition_divergence(M, int(m_key), Z, "cause")
                if d_e > phi_e:
                    phi_e = d_e
                if d_c > phi_c:
                    phi_c = d_c
            phi_state = min(phi_e, phi_c)
            phi_M_terms.append(float(w_m) * float(phi_state))
        phi_M = float(math.fsum(phi_M_terms)) if phi_M_terms else 0.0
        w_size = 1.0 / float(len(M))
        psi_terms.append(float(w_size) * phi_M)
    return float(math.fsum(psi_terms)) if psi_terms else 0.0


def _iim_phase1_worker_cleanup():
    global _IIM_PHASE1_CTX
    if not isinstance(_IIM_PHASE1_CTX, dict):
        _IIM_PHASE1_CTX = None
        return
    kernel_cache = _IIM_PHASE1_CTX.get("kernel_cache")
    if kernel_cache is not None:
        try:
            kernel_cache.close()
        except Exception:
            pass
    shms = _IIM_PHASE1_CTX.get("shms", [])
    for shm in shms:
        try:
            shm.close()
        except Exception:
            pass
    tpm_shm = _IIM_PHASE1_CTX.get("tpm_shm")
    if tpm_shm is not None:
        try:
            tpm_shm.close()
        except Exception:
            pass
    _IIM_PHASE1_CTX = None


def _iim_open_readonly_array_spec(spec):
    mode = str(spec.get("mode", "shared_memory"))
    if mode == "shared_memory":
        shm = shared_memory.SharedMemory(name=str(spec["name"]))
        arr = np.ndarray(
            tuple(spec["shape"]),
            dtype=np.dtype(spec["dtype"]),
            buffer=shm.buf,
        )
        arr.flags.writeable = False
        return arr, shm
    if mode == "memmap":
        arr = np.load(str(spec["path"]), mmap_mode="r")
        arr.flags.writeable = False
        return arr, None
    raise RuntimeError(f"Unsupported IIM array mode: {mode}")


def _iim_phase_worker_init_static(spec_curr, spec_states, base, purviews):
    global _IIM_PHASE1_CTX
    _iim_phase1_worker_cleanup()
    curr_obs, shm_curr = _iim_open_readonly_array_spec(spec_curr)
    states_full, shm_states = _iim_open_readonly_array_spec(spec_states)
    shms = []
    if shm_curr is not None:
        shms.append(shm_curr)
    if shm_states is not None:
        shms.append(shm_states)
    _IIM_PHASE1_CTX = {
        "base": int(base),
        "purviews": tuple(tuple(z) for z in purviews),
        "curr_obs": curr_obs,
        "states_full": states_full,
        "shms": shms,
        "tpm": None,
        "tpm_shm": None,
        "tpm_token": None,
        # Reused across chunks/cuts for this worker.
        "static_cache": {},
        "obs_state_cache": {},
        "kernel_cache": None,
        "kernel_cache_token": None,
    }
    atexit.register(_iim_phase1_worker_cleanup)


def _iim_phase_worker_bind_tpm(spec_tpm):
    global _IIM_PHASE1_CTX
    if not isinstance(_IIM_PHASE1_CTX, dict):
        raise RuntimeError("IIM worker context not initialized.")
    mode = str(spec_tpm.get("mode", "shared_memory"))
    token = (
        mode,
        str(spec_tpm.get("name", "")),
        str(spec_tpm.get("path", "")),
    )
    if _IIM_PHASE1_CTX.get("tpm_token") == token and _IIM_PHASE1_CTX.get("tpm") is not None:
        return

    old_shm = _IIM_PHASE1_CTX.get("tpm_shm")
    if old_shm is not None:
        try:
            old_shm.close()
        except Exception:
            pass
    _IIM_PHASE1_CTX["tpm_shm"] = None

    tpm, shm = _iim_open_readonly_array_spec(spec_tpm)
    _IIM_PHASE1_CTX["tpm"] = tpm
    _IIM_PHASE1_CTX["tpm_shm"] = shm
    _IIM_PHASE1_CTX["tpm_token"] = token


def _iim_phase_worker_bind_kernel_cache(cache_spec):
    global _IIM_PHASE1_CTX
    if not isinstance(_IIM_PHASE1_CTX, dict):
        raise RuntimeError("IIM worker context not initialized.")

    if not isinstance(cache_spec, dict) or not bool(cache_spec.get("enabled", False)):
        old_cache = _IIM_PHASE1_CTX.get("kernel_cache")
        if old_cache is not None:
            try:
                old_cache.close()
            except Exception:
                pass
        _IIM_PHASE1_CTX["kernel_cache"] = None
        _IIM_PHASE1_CTX["kernel_cache_token"] = None
        return

    path = str(cache_spec.get("path", "")).strip()
    if not path:
        raise RuntimeError("IIM worker kernel cache path is empty.")
    mem_entries = int(cache_spec.get("memory_entries", 50_000))
    flush_batch = int(cache_spec.get("flush_batch", 5_000))
    token = (path, int(mem_entries), int(flush_batch))

    if (
        _IIM_PHASE1_CTX.get("kernel_cache_token") == token
        and _IIM_PHASE1_CTX.get("kernel_cache") is not None
    ):
        return

    old_cache = _IIM_PHASE1_CTX.get("kernel_cache")
    if old_cache is not None:
        try:
            old_cache.close()
        except Exception:
            pass

    cache = _IIMDiskKernelCache(
        path,
        signature=None,
        memory_entries=int(mem_entries),
        flush_batch=int(flush_batch),
    )
    _IIM_PHASE1_CTX["kernel_cache"] = cache
    _IIM_PHASE1_CTX["kernel_cache_token"] = token


def _iim_phase_worker_run_chunk_for_tpm(
    spec_tpm,
    mechanisms_chunk,
    cache_spec=None,
    cut_mask_a=None,
    use_induced_partition_cache=False,
    kernel_cache_lookup_only=False,
):
    if not isinstance(_IIM_PHASE1_CTX, dict):
        raise RuntimeError("IIM worker context not initialized.")
    _iim_phase_worker_bind_tpm(spec_tpm)
    _iim_phase_worker_bind_kernel_cache(cache_spec)
    mechanisms = tuple(tuple(m) for m in mechanisms_chunk)
    kernel_cache = _IIM_PHASE1_CTX.get("kernel_cache")
    psi_chunk = _iim_phase1_chunk_contribution(
        mechanisms,
        _IIM_PHASE1_CTX["purviews"],
        int(_IIM_PHASE1_CTX["base"]),
        _IIM_PHASE1_CTX["tpm"],
        _IIM_PHASE1_CTX["curr_obs"],
        _IIM_PHASE1_CTX["states_full"],
        static_cache=_IIM_PHASE1_CTX.get("static_cache"),
        obs_state_cache=_IIM_PHASE1_CTX.get("obs_state_cache"),
        kernel_cache=kernel_cache,
        cut_mask_a=cut_mask_a,
        use_induced_partition_cache=bool(use_induced_partition_cache) and (kernel_cache is not None),
        kernel_cache_lookup_only=bool(kernel_cache_lookup_only),
    )
    return float(psi_chunk), int(len(mechanisms))


def _safe_mutual_info_score(labels_a, labels_b):
    """
    Compute MI while suppressing sklearn's high-cardinality class warning.
    For discretized continuous time-series this warning is expected and not
    informative for our use case.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50% of the number of samples.*",
            category=UserWarning,
        )
        return mutual_info_score(labels_a, labels_b)


def _coerce_ram_event_bundle(stimulus_onsets):
    """
    Normalize RAM event input into a structured bundle.

    Supported inputs:
      - list/array-like of onset seconds (legacy)
      - dict with keys:
          onsets (or stimulus_onsets/stim_onsets/events)
          goal_onsets
          feedback_onsets
          feedback_values
    """
    bundle = {
        "onsets": [],
        "goal_onsets": [],
        "feedback_onsets": [],
        "feedback_values": None,
    }
    if stimulus_onsets is None:
        return bundle
    if isinstance(stimulus_onsets, dict):
        onsets = stimulus_onsets.get("onsets")
        if onsets is None:
            for k in ("stimulus_onsets", "stim_onsets", "events"):
                if k in stimulus_onsets:
                    onsets = stimulus_onsets.get(k)
                    break
        bundle["onsets"] = [] if onsets is None else onsets
        bundle["goal_onsets"] = stimulus_onsets.get("goal_onsets", [])
        bundle["feedback_onsets"] = stimulus_onsets.get("feedback_onsets", [])
        bundle["feedback_values"] = stimulus_onsets.get("feedback_values")
        return bundle

    bundle["onsets"] = stimulus_onsets
    return bundle


def _coerce_numeric_feedback(values):
    """
    Convert feedback labels/values to numeric.

    If numeric conversion fails for all entries, factor-encode categorical
    labels deterministically.
    """
    if values is None:
        return np.empty(0, dtype=float)

    raw = np.asarray(list(values), dtype=object).reshape(-1)
    if raw.size == 0:
        return np.empty(0, dtype=float)

    out = np.full(raw.shape, np.nan, dtype=float)
    for i, v in enumerate(raw):
        try:
            out[i] = float(v)
        except Exception:
            out[i] = np.nan

    if np.isfinite(out).any():
        return out

    labels = np.asarray([str(v) for v in raw], dtype=str)
    _, inv = np.unique(labels, return_inverse=True)
    return inv.astype(float)


def _sanitize_onset_seconds(onsets, tr, n_tp):
    """
    Keep finite in-range onsets, map to nearest sample index, and deduplicate
    by index while preserving temporal order.
    """
    arr = np.asarray(onsets if onsets is not None else [], dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr, np.empty(0, dtype=np.int64)

    tr = float(tr)
    max_t = max(0.0, (int(n_tp) - 1) * tr)
    arr = arr[(arr >= 0.0) & (arr <= (max_t + 0.5 * tr))]
    if arr.size == 0:
        return arr, np.empty(0, dtype=np.int64)

    idx = np.rint(arr / tr).astype(np.int64)
    valid = (idx >= 0) & (idx < int(n_tp))
    arr = arr[valid]
    idx = idx[valid]
    if idx.size == 0:
        return arr, np.empty(0, dtype=np.int64)

    order = np.argsort(idx, kind="mergesort")
    idx = idx[order]
    arr = arr[order]
    keep = np.ones(idx.shape[0], dtype=bool)
    if idx.size > 1:
        keep[1:] = idx[1:] != idx[:-1]
    return arr[keep], idx[keep]


def _window_mean_vectors(ts, event_idx, pre_samples, post_samples):
    """
    Compute event-locked mean vectors over pre/post windows.

    Returns
    -------
    pre_vecs, post_vecs, used_idx
      pre_vecs/post_vecs: arrays of shape (n_events, n_regions)
      used_idx: event indices (sample domain) after edge filtering
    """
    n_regions, n_tp = ts.shape
    pre_samples = max(0, int(pre_samples))
    post_samples = max(1, int(post_samples))

    if event_idx.size == 0:
        empty = np.empty((0, n_regions), dtype=float)
        return empty, empty, np.empty(0, dtype=np.int64)

    valid = (event_idx - pre_samples >= 0) & (event_idx + post_samples <= n_tp)
    idx = event_idx[valid]
    if idx.size == 0:
        empty = np.empty((0, n_regions), dtype=float)
        return empty, empty, np.empty(0, dtype=np.int64)

    post_offsets = np.arange(0, post_samples, dtype=np.int64)
    post_ix = idx[:, None] + post_offsets[None, :]
    post_vecs = ts[:, post_ix].mean(axis=2).T

    if pre_samples > 0:
        pre_offsets = np.arange(-pre_samples, 0, dtype=np.int64)
        pre_ix = idx[:, None] + pre_offsets[None, :]
        pre_vecs = ts[:, pre_ix].mean(axis=2).T
    else:
        pre_vecs = np.zeros((idx.size, n_regions), dtype=float)

    return pre_vecs, post_vecs, idx


def _matrix_invsqrt_psd(mat, eps=1e-10, hardware_backend=None):
    return accelerated_psd_invsqrt(mat, eps=eps, backend=hardware_backend)


def _regularized_first_canonical_corr(x, y, ridge=1e-4, hardware_backend=None):
    """
    First canonical correlation with ridge-regularized covariance matrices.
    """
    if x.ndim != 2 or y.ndim != 2:
        return np.nan

    n = int(min(x.shape[0], y.shape[0]))
    if n < 3:
        return np.nan
    x = np.asarray(x[:n], dtype=float)
    y = np.asarray(y[:n], dtype=float)

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    keep_x = np.var(x, axis=0) > 1e-12
    keep_y = np.var(y, axis=0) > 1e-12
    if not np.any(keep_x) or not np.any(keep_y):
        return np.nan
    x = x[:, keep_x]
    y = y[:, keep_y]

    denom = float(n - 1)
    sxx = (x.T @ x) / denom
    syy = (y.T @ y) / denom
    sxy = (x.T @ y) / denom

    ridge = float(max(ridge, 1e-12))
    lam_x = ridge * max(np.trace(sxx) / max(1, sxx.shape[0]), 1.0)
    lam_y = ridge * max(np.trace(syy) / max(1, syy.shape[0]), 1.0)

    try:
        wx = _matrix_invsqrt_psd(
            sxx + lam_x * np.eye(sxx.shape[0], dtype=float),
            hardware_backend=hardware_backend,
        )
        wy = _matrix_invsqrt_psd(
            syy + lam_y * np.eye(syy.shape[0], dtype=float),
            hardware_backend=hardware_backend,
        )
        k = wx @ sxy @ wy
        svals = accelerated_svd_values(k, backend=hardware_backend)
    except np.linalg.LinAlgError:
        return np.nan

    if svals.size == 0:
        return np.nan
    return float(np.clip(svals[0], 0.0, 1.0))


def _safe_abs_corr(a, b):
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    n = int(min(a.size, b.size))
    if n < 3:
        return np.nan
    a = a[:n]
    b = b[:n]
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return np.nan
    aa = a[mask]
    bb = b[mask]
    if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
        return np.nan
    r = np.corrcoef(aa, bb)[0, 1]
    if not np.isfinite(r):
        return np.nan
    return float(np.clip(abs(r), 0.0, 1.0))


def _sample_reliability(n_samples, tau=4.0):
    n = max(0, int(n_samples))
    tau = max(float(tau), 1e-6)
    return float(1.0 - np.exp(-float(n) / tau))


def _prediction_error_signal(values):
    """
    Simple online prediction-error proxy:
      delta_t = x_t - mean(x_<t)
    """
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        return np.empty(0, dtype=float)

    pred = np.empty_like(x)
    pred[0] = x[0]
    if x.size > 1:
        csum = np.cumsum(x[:-1], dtype=float)
        pred[1:] = csum / np.arange(1, x.size, dtype=float)
    return x - pred


def _goal_alignment_component(
    ts_z,
    stim_idx,
    goal_idx,
    goal_pre_samples,
    response_samples,
    goal_objective_samples,
    ridge,
    hardware_backend=None,
):
    """
    Goal-alignment component G in [0,1].

    If explicit goal events exist, pairs each response event with its nearest
    preceding goal event. Otherwise uses pre-stimulus activity as an implicit
    objective-state proxy.
    """
    goal_pre_vecs, resp_vecs, stim_used = _window_mean_vectors(
        ts_z,
        stim_idx,
        pre_samples=goal_pre_samples,
        post_samples=response_samples,
    )
    if stim_used.size < 3:
        return 0.0, resp_vecs

    x_goal = goal_pre_vecs
    y_resp = resp_vecs

    if goal_idx.size > 0:
        _, goal_post_vecs, goal_used = _window_mean_vectors(
            ts_z,
            goal_idx,
            pre_samples=0,
            post_samples=goal_objective_samples,
        )
        if goal_used.size > 0:
            pos = np.searchsorted(goal_used, stim_used, side="right") - 1
            valid = pos >= 0
            if int(valid.sum()) >= 3:
                x_goal = goal_post_vecs[pos[valid]]
                y_resp = resp_vecs[valid]

    rho = _regularized_first_canonical_corr(
        x_goal,
        y_resp,
        ridge=ridge,
        hardware_backend=hardware_backend,
    )
    if not np.isfinite(rho):
        return 0.0, resp_vecs
    rel = _sample_reliability(x_goal.shape[0], tau=4.0)
    score = float(np.clip(rho * rel, 0.0, 1.0))
    return score, resp_vecs


def _feedback_integration_component(
    ts_z,
    feedback_idx,
    feedback_values,
    stim_response_vecs,
    feedback_samples,
    hardware_backend=None,
):
    """
    Feedback-integration component F in [0,1].

    Correlates feedback-locked neural response strength with an explicit
    feedback signal when available, otherwise with a prediction-error proxy.
    """
    fb_pre_vecs, fb_post_vecs, fb_used = _window_mean_vectors(
        ts_z,
        feedback_idx,
        pre_samples=feedback_samples,
        post_samples=feedback_samples,
    )
    if fb_used.size < 2:
        return 0.0, np.empty(0, dtype=float)

    fb_neural = accelerated_row_norm(
        fb_post_vecs - fb_pre_vecs,
        axis=1,
        backend=hardware_backend,
    )
    fb_signal = _coerce_numeric_feedback(feedback_values)

    if fb_signal.size == 0:
        if stim_response_vecs.size == 0:
            return 0.0, np.empty(0, dtype=float)
        resp_energy = np.linalg.norm(stim_response_vecs, axis=1)
        fb_signal = _prediction_error_signal(resp_energy)

    n = int(min(fb_neural.size, fb_signal.size))
    if n < 2:
        return 0.0, np.empty(0, dtype=float)
    fb_neural = fb_neural[:n]
    fb_signal = np.asarray(fb_signal[:n], dtype=float)

    corr = _safe_abs_corr(fb_neural, fb_signal)
    if not np.isfinite(corr):
        return 0.0, fb_signal
    rel = _sample_reliability(n, tau=4.0)
    score = float(np.clip(corr * rel, 0.0, 1.0))
    return score, fb_signal


def _adaptive_update_component(stim_response_vecs, feedback_signal, hardware_backend=None):
    """
    Adaptive-update component U in [0,1].

    Quantifies whether stronger feedback drives larger trial-to-trial policy
    updates in neural response patterns.
    """
    if stim_response_vecs.ndim != 2 or stim_response_vecs.shape[0] < 3:
        return 0.0

    update_mag = accelerated_row_norm(
        stim_response_vecs[1:, :] - stim_response_vecs[:-1, :],
        axis=1,
        backend=hardware_backend,
    )
    if update_mag.size < 2:
        return 0.0

    feedback_signal = np.asarray(feedback_signal, dtype=float).reshape(-1)
    if feedback_signal.size < 2:
        resp_energy = accelerated_row_norm(
            stim_response_vecs,
            axis=1,
            backend=hardware_backend,
        )
        feedback_signal = _prediction_error_signal(resp_energy)

    drive = np.abs(feedback_signal)
    n = int(min(update_mag.size, drive.size))
    if n < 2:
        return 0.0

    corr = _safe_abs_corr(update_mag[:n], drive[:n])
    if not np.isfinite(corr):
        return 0.0
    rel = _sample_reliability(n, tau=4.0)
    return float(np.clip(corr * rel, 0.0, 1.0))


def compute_RAM(
    ts: np.ndarray,
    tr: float = 1.0,
    stimulus_onsets=None,
    epsilon: float = None,
    magnitude_scale: float = 0.5,
    response_model: str = "hrf",
    response_boxcar_width_sec: float = None,
    latency_method: str = "hrf_peak",
    fir_window: float = 20.0,
    xcorr_maxlag: int = 10,
    quality_weights=(1.0, 1.0, 1.0),
    goal_pre_window_sec: float = 2.0,
    response_window_sec: float = 3.0,
    goal_objective_window_sec: float = 2.0,
    feedback_window_sec: float = 2.0,
    quality_ridge: float = 1e-4,
    require_explicit_feedback: bool = True,
    return_details: bool = False,
    hardware_backend=None,
):
    """
    Responsiveness–Adaptation Metric (RAM).

    Parameters
    ----------
    ts : ndarray, shape (n_regions, n_time)
        Timeseries matrix for each ROI.
    tr : float
        Repetition time (sampling interval) in seconds.
    stimulus_onsets : list/array-like or dict, optional
        Event specification. Legacy mode accepts a list/array of stimulus onset
        times (in seconds). Structured mode accepts a dict with:
          - ``onsets`` (or ``stimulus_onsets``): stimulus onsets in seconds
          - ``goal_onsets``: objective/cue onsets in seconds
          - ``feedback_onsets``: feedback/outcome onsets in seconds
          - ``feedback_values``: scalar feedback labels/values aligned to
            ``feedback_onsets`` (numeric or categorical)
    epsilon : float, optional
        Small constant added to denominator. When ``None`` (default), uses
        ``tr`` so the stabilizer equals the temporal resolution (Δt_res).
    magnitude_scale : float, optional
        Multiplicative scale applied to the response magnitude term M
        (default 0.5).
    response_model : {'hrf', 'boxcar', 'stick'}, optional
        Event-response regressor used for the magnitude term. ``'hrf'`` keeps
        the original fMRI-oriented canonical HRF convolution. ``'boxcar'`` and
        ``'stick'`` are intended for high-sampling-rate modalities such as EEG,
        where convolving a sub-second sampling grid with an fMRI HRF is both
        computationally expensive and physiologically mismatched.
    response_boxcar_width_sec : float, optional
        Width of the post-stimulus boxcar regressor when
        ``response_model='boxcar'``. Defaults to ``response_window_sec``.
    latency_method : {'hrf_peak', 'xcorr', 'fir'}, optional
        Method used to estimate the latency term T in the denominator.
        * 'hrf_peak' (default): uses the canonical HRF peak latency.
        * 'xcorr': cross-correlates each ROI with the stick function and
          averages the lag (in seconds) yielding maximum correlation magnitude.
        * 'fir': fits a finite impulse response (FIR) around each onset and
          takes the average time-to-peak of the event-related response.
    fir_window : float, optional
        Time window in seconds on either side of each onset for the FIR
        latency estimate (only used when ``latency_method='fir'``).
    xcorr_maxlag : int, optional
        Maximum lag (in TRs) explored on either side for the cross-correlation
        latency estimate (only used when ``latency_method='xcorr'``).
    quality_weights : tuple(float, float, float), optional
        Non-negative weights ``(w_G, w_F, w_U)`` for quality components
        Goal-alignment (G), Feedback-integration (F), and Adaptive-update (U).
    goal_pre_window_sec : float, optional
        Pre-stimulus window (seconds) used for implicit objective-state
        extraction.
    response_window_sec : float, optional
        Post-stimulus response window (seconds) for response-state extraction.
    goal_objective_window_sec : float, optional
        Post-goal-event window (seconds) used when explicit ``goal_onsets`` are
        available.
    feedback_window_sec : float, optional
        Symmetric feedback-locked window length (seconds) used for
        pre/post-feedback contrast.
    quality_ridge : float, optional
        Ridge regularization strength for canonical-correlation estimation in G.
    require_explicit_feedback : bool, optional
        When ``True`` (default), RAM is marked undefined unless explicit
        feedback events and feedback values are provided. This prevents
        synthesizing feedback-driven quality terms from proxy assumptions when
        environmental feedback is not directly annotated.
    return_details : bool, optional
        If ``True``, returns a dict with speed/quality sub-terms and component
        diagnostics instead of only the scalar RAM value.

    Returns
    -------
    float or dict
        Scalar RAM value (default) or a diagnostics dict when
        ``return_details=True``.

    Notes
    -----
    RAM is implemented as:
      RAM = (M / (T + epsilon)) * Q
    with quality term:
      Q = (G^w_G * F^w_F * U^w_U)^(1 / (w_G + w_F + w_U))
    where G, F, U are data-derived in [0,1] from event-locked neural activity.
    """
    # ensure ts has shape (n_regions, n_tp)
    if ts.ndim != 2:
        raise ValueError(f"ts should be 2D (n_regions × n_time), got shape {ts.shape}")
    if tr is None or tr <= 0:
        raise ValueError("tr must be a positive number for RAM")
    if magnitude_scale <= 0:
        raise ValueError("magnitude_scale must be > 0")
    response_model_norm = str(response_model).strip().lower()
    if response_model_norm not in {"hrf", "boxcar", "stick"}:
        raise ValueError("response_model must be one of {'hrf', 'boxcar', 'stick'}")
    if response_boxcar_width_sec is not None and float(response_boxcar_width_sec) <= 0:
        raise ValueError("response_boxcar_width_sec must be > 0 when provided")
    if epsilon is None:
        epsilon = float(tr)
    if epsilon < 0:
        raise ValueError("epsilon must be >= 0")
    w = np.asarray(quality_weights, dtype=float).reshape(-1)
    if w.size != 3:
        raise ValueError("quality_weights must be a 3-tuple: (w_G, w_F, w_U)")
    if np.any(w < 0):
        raise ValueError("quality_weights must be non-negative")
    if not np.any(w > 0):
        raise ValueError("At least one quality weight must be > 0")
    if goal_pre_window_sec < 0 or response_window_sec <= 0:
        raise ValueError("goal_pre_window_sec must be >= 0 and response_window_sec > 0")
    if goal_objective_window_sec <= 0 or feedback_window_sec <= 0:
        raise ValueError("goal_objective_window_sec and feedback_window_sec must be > 0")
    if quality_ridge <= 0:
        raise ValueError("quality_ridge must be > 0")

    backend = resolve_hardware_backend(hardware_backend)
    n_regions, n_tp = ts.shape
    event_bundle = _coerce_ram_event_bundle(stimulus_onsets)
    stim_onsets_s, stim_idx = _sanitize_onset_seconds(event_bundle["onsets"], tr=tr, n_tp=n_tp)
    stim_onsets = stim_onsets_s.tolist()
    if stim_idx.size == 0:
        if not return_details:
            return float("nan")
        return {
            "value": float("nan"),
            "undefined_reason": "missing_stimulus_events",
            "speed_term": float("nan"),
            "quality_term": float("nan"),
            "magnitude_term": float("nan"),
            "latency_term": float("nan"),
            "latency_seconds": float("nan"),
            "epsilon": float(epsilon),
            "components": {
                "goal_alignment": float("nan"),
                "feedback_integration": float("nan"),
                "adaptive_update": float("nan"),
            },
            "weights": {
                "goal_alignment": float(w[0]),
                "feedback_integration": float(w[1]),
                "adaptive_update": float(w[2]),
            },
            "n_stimulus_events": 0,
            "n_goal_events": 0,
            "n_feedback_events": 0,
        }
    log.debug(
        (
            "[compute_RAM] ts shape: %d regions x %d timepoints, tr=%.6f, "
            "n_onsets=%d, latency_method=%s"
        ),
        n_regions,
        n_tp,
        float(tr),
        len(stim_onsets) if stim_onsets is not None else 0,
        latency_method,
    )

    # 1) build stick regressor
    stick = np.zeros(n_tp)
    stick[stim_idx] = 1

    # 2) create stimulus-response regressor.
    hrf = None
    default_latency = 0.0
    if response_model_norm == "hrf":
        hrf = glover_hrf(tr)
        reg = np.convolve(stick, hrf)[:n_tp]
        default_latency = float(np.argmax(hrf) * tr)
    elif response_model_norm == "boxcar":
        width_sec = (
            float(response_window_sec)
            if response_boxcar_width_sec is None
            else float(response_boxcar_width_sec)
        )
        width_samples = max(1, int(round(width_sec / float(tr))))
        reg = np.zeros(n_tp, dtype=float)
        for idx in stim_idx.tolist():
            end = min(n_tp, int(idx) + width_samples)
            if end > int(idx):
                reg[int(idx):end] += 1.0
        reg /= float(width_samples)
        default_latency = 0.5 * float(width_samples) * float(tr)
    else:
        reg = stick.astype(float, copy=True)
        default_latency = 0.0
    # design matrix: stimulus regressor + intercept
    X = np.vstack([reg, np.ones(n_tp)]).T  # shape (n_tp × 2)

    # 3) solve for betas via pseudo-inverse: betas shape (2 × n_regions)
    # transpose ts so rows correspond to time samples
    betas = accelerated_pinv_dot(X, ts.T, backend=backend)
    stim_betas = betas[0, :]  # first row corresponds to stimulus regressor

    # use mean absolute β as amplitude (M), with optional scale factor
    abs_mean_beta = float(np.mean(np.abs(stim_betas))) * float(magnitude_scale)

    # 4) estimate latency T depending on chosen method
    T = 0.0
    if latency_method == "hrf_peak":
        # Peak of the selected response model. For fMRI this is the canonical
        # HRF peak; for non-HRF models it is the model's intrinsic latency.
        T = default_latency
    elif latency_method == "xcorr":
        # estimate per-ROI latency by cross-correlating z-scored ROI with stick
        roi_lags = []
        # pre-normalize the stick for correlation
        stick_mean = stick.mean()
        stick_std = stick.std(ddof=0)
        # if stick_std is zero (no onsets), fall back to HRF peak
        if stick_std == 0 or np.all(stick == 0):
            T = np.argmax(hrf) * tr
        else:
            stick_norm = (stick - stick_mean) / (stick_std + 1e-12)
            for r in range(n_regions):
                # z-score ROI
                roi = ts[r]
                roi_norm = (roi - roi.mean()) / (roi.std(ddof=0) + 1e-12)
                best_abs_corr = -np.inf
                best_lag = 0
                # explore lags in TR units
                for lag in range(-xcorr_maxlag, xcorr_maxlag + 1):
                    # shift ROI relative to stick
                    if lag > 0:
                        # roi shifted right; shorten both sequences
                        roi_shift = roi_norm[lag:]
                        stick_shift = stick_norm[: len(roi_shift)]
                    elif lag < 0:
                        roi_shift = roi_norm[: lag]
                        stick_shift = stick_norm[-lag:]
                    else:
                        roi_shift = roi_norm
                        stick_shift = stick_norm
                    # require at least two points to compute correlation
                    if roi_shift.size < 2:
                        continue
                    corr = np.corrcoef(roi_shift, stick_shift)[0, 1]
                    # use absolute correlation to find strongest alignment
                    abs_corr = abs(corr)
                    if abs_corr > best_abs_corr:
                        best_abs_corr = abs_corr
                        best_lag = lag
                roi_lags.append(best_lag * tr)
            # average positive latencies; if none, set to HRF peak
            if roi_lags:
                T = float(np.mean(roi_lags))
            else:
                T = default_latency
    elif latency_method == "fir":
        # estimate latency by fitting an FIR around each onset for each ROI
        if stim_onsets is None or len(stim_onsets) == 0:
            # no events -> revert to selected response-model latency
            T = default_latency
        else:
            # number of samples on either side of onset
            half_window = int(round(fir_window / tr))
            roi_latencies = []
            for r in range(n_regions):
                # collect per-event segments
                segments = []
                for onset in stim_onsets:
                    idx = int(round(onset / tr))
                    start = idx - half_window
                    end = idx + half_window + 1
                    # ensure indices within bounds
                    if start < 0 or end > n_tp:
                        continue
                    seg = ts[r, start:end]
                    segments.append(seg)
                if not segments:
                    continue
                # average across segments
                avg_resp = np.mean(segments, axis=0)
                # find index of peak relative to onset
                # consider absolute peak to account for undershoots
                peak_idx = int(np.argmax(np.abs(avg_resp)))
                # convert index to time lag relative to onset
                lag_tr = peak_idx - half_window
                roi_latencies.append(lag_tr * tr)
            if roi_latencies:
                T = float(np.mean(roi_latencies))
            else:
                T = default_latency
    else:
        raise ValueError(
            "latency_method must be one of 'hrf_peak', 'xcorr', or 'fir'"
        )

    # latency represents a response delay and should be non-negative
    T = max(float(T), 0.0)
    # avoid division by zero
    latency_term = T + float(epsilon)
    speed_term = abs_mean_beta / latency_term

    # 5) quality term Q = weighted geometric mean of G, F, U
    _, goal_idx = _sanitize_onset_seconds(event_bundle["goal_onsets"], tr=tr, n_tp=n_tp)
    _, feedback_idx = _sanitize_onset_seconds(event_bundle["feedback_onsets"], tr=tr, n_tp=n_tp)
    feedback_values_num = _coerce_numeric_feedback(event_bundle.get("feedback_values"))

    if require_explicit_feedback:
        if (feedback_idx.size == 0) or (feedback_values_num.size < 2):
            if not return_details:
                return float("nan")
            return {
                "value": float("nan"),
                "undefined_reason": "missing_explicit_feedback",
                "speed_term": float(speed_term),
                "quality_term": float("nan"),
                "magnitude_term": float(abs_mean_beta),
                "latency_term": float(latency_term),
                "latency_seconds": float(T),
                "epsilon": float(epsilon),
                "components": {
                    "goal_alignment": float("nan"),
                    "feedback_integration": float("nan"),
                    "adaptive_update": float("nan"),
                },
                "weights": {
                    "goal_alignment": float(w[0]),
                    "feedback_integration": float(w[1]),
                    "adaptive_update": float(w[2]),
                },
                "n_stimulus_events": int(stim_idx.size),
                "n_goal_events": int(goal_idx.size),
                "n_feedback_events": int(feedback_idx.size),
            }

    if feedback_idx.size == 0:
        feedback_idx = stim_idx

    ts_z = accelerated_zscore(ts, axis=1, backend=backend, eps=1e-12)

    goal_pre_samples = max(0, int(round(goal_pre_window_sec / float(tr))))
    response_samples = max(1, int(round(response_window_sec / float(tr))))
    goal_objective_samples = max(1, int(round(goal_objective_window_sec / float(tr))))
    feedback_samples = max(1, int(round(feedback_window_sec / float(tr))))

    g_score, stim_response_vecs = _goal_alignment_component(
        ts_z=ts_z,
        stim_idx=stim_idx,
        goal_idx=goal_idx,
        goal_pre_samples=goal_pre_samples,
        response_samples=response_samples,
        goal_objective_samples=goal_objective_samples,
        ridge=quality_ridge,
        hardware_backend=backend,
    )
    f_score, feedback_signal = _feedback_integration_component(
        ts_z=ts_z,
        feedback_idx=feedback_idx,
        feedback_values=feedback_values_num,
        stim_response_vecs=stim_response_vecs,
        feedback_samples=feedback_samples,
        hardware_backend=backend,
    )
    u_score = _adaptive_update_component(
        stim_response_vecs=stim_response_vecs,
        feedback_signal=feedback_signal,
        hardware_backend=backend,
    )

    components = np.asarray([g_score, f_score, u_score], dtype=float)
    components = np.nan_to_num(components, nan=0.0, posinf=0.0, neginf=0.0)
    components = np.clip(components, 0.0, 1.0)

    # Exact weighted geometric mean: any zero-valued weighted component yields Q=0.
    if np.any((components <= 0.0) & (w > 0.0)):
        quality_term = 0.0
    else:
        quality_term = float(np.exp(np.sum(w * np.log(components)) / np.sum(w)))
        quality_term = float(np.clip(quality_term, 0.0, 1.0))

    ram_value = float(speed_term * quality_term)

    if not return_details:
        return ram_value

    return {
        "value": ram_value,
        "speed_term": float(speed_term),
        "quality_term": float(quality_term),
        "magnitude_term": float(abs_mean_beta),
        "latency_term": float(latency_term),
        "latency_seconds": float(T),
        "epsilon": float(epsilon),
        "response_model": response_model_norm,
        "components": {
            "goal_alignment": float(components[0]),
            "feedback_integration": float(components[1]),
            "adaptive_update": float(components[2]),
        },
        "weights": {
            "goal_alignment": float(w[0]),
            "feedback_integration": float(w[1]),
            "adaptive_update": float(w[2]),
        },
        "n_stimulus_events": int(stim_idx.size),
        "n_goal_events": int(goal_idx.size),
        "n_feedback_events": int(feedback_idx.size),
    }

def compute_PDI(
    ts: np.ndarray,
    bins: int = 10,
    baseline_ts: np.ndarray = None,
    weighted: bool = True,
    normalize: bool = False,
    clip_negative: bool = True,
    stability_segments: int = 4,
    noise_penalty_kappa: float = 1.0,
    component_weights: tuple = (0.35, 0.25, 0.20, 0.20),
    ordinal_order: int = 3,
    multiscale_max_scale: int = 5,
    eps: float = 1e-12,
    hardware_backend=None,
) -> float:
    """
    Composite measurable Phenomenal Differentiation Index (PDI).

    PDI is operationalized from EEG/fMRI timeseries as a baseline-referenced
    composition of four observable dimensions:
      1) normalized excess repertoire differentiation (Xi_norm),
      2) spatial repertoire divergence across regions (Delta_spatial),
      3) effective representational dimensionality (D_eff_norm), and
      4) multiscale ordinal complexity (C_multi).

    Each dimension is measured for observed and baseline runs, baseline
    variability is explicitly attenuated per dimension, and the aggregate is
    corrected by temporal stability and differential noise penalties.

    Parameters
    ----------
    ts : ndarray, shape (n_regions, n_time)
        Task timeseries.
    bins : int, optional
        Number of histogram bins for discretization. The maximum entropy is
        ``log2(bins)``.
    baseline_ts : ndarray, optional
        Baseline (rest) timeseries. Can be provided as a 2D array of shape
        (n_regions, n_time) corresponding to a single run, or a 3D array of
        shape (n_runs, n_regions, n_time) corresponding to multiple baseline
        runs. If ``None``, the baseline entropy is approximated by shuffling
        each region of ``ts``.
    weighted : bool, optional
        Only used when ``baseline_ts`` is 3D. When ``True`` (default), the
        baseline component statistics from multiple runs are weighted by their
        number of timepoints. When ``False``, a simple mean across runs is used.
    normalize : bool, optional
        When ``True`` returns a bounded [0,1] representation. With
        ``clip_negative=False``, signed values are mapped by ``0.5*(x+1)``.
    clip_negative : bool, optional
        When ``True`` (default), negative gains are clipped to zero.
    stability_segments : int, optional
        Number of contiguous segments for split-stability estimation.
    noise_penalty_kappa : float, optional
        Strength of differential-noise attenuation.
    component_weights : tuple, optional
        Non-negative weights for (Xi_norm, Delta_spatial, D_eff_norm, C_multi).
    ordinal_order : int, optional
        Permutation order used for multiscale ordinal complexity.
    multiscale_max_scale : int, optional
        Maximum coarse-graining scale for multiscale ordinal complexity.
    eps : float, optional
        Numerical stability constant.

    Returns
    -------
    float
        The PDI value.

    Notes
    -----
    With ``clip_negative=True`` (default), the output is non-negative and
    increases when observed differentiation exceeds baseline with good temporal
    stability and low differential noise.
    """
    if ts.ndim != 2:
        raise ValueError(f"ts should be 2D (n_regions × n_time), got shape {ts.shape}")
    if bins < 2:
        raise ValueError("bins must be >= 2")
    if stability_segments < 2:
        raise ValueError("stability_segments must be >= 2")
    if noise_penalty_kappa < 0:
        raise ValueError("noise_penalty_kappa must be >= 0")
    if ordinal_order < 3:
        raise ValueError("ordinal_order must be >= 3")
    if multiscale_max_scale < 1:
        raise ValueError("multiscale_max_scale must be >= 1")

    comp_w = np.asarray(component_weights, dtype=float)
    if comp_w.shape != (4,):
        raise ValueError("component_weights must contain 4 values")
    if np.any(comp_w < 0) or float(np.sum(comp_w)) <= 0:
        raise ValueError("component_weights must be non-negative and not all zero")
    comp_w = comp_w / float(np.sum(comp_w))
    backend = resolve_hardware_backend(hardware_backend)

    # compute baseline reference set
    if baseline_ts is None:
        # fallback: shuffled surrogate destroys temporal structure while
        # preserving marginal amplitudes.
        n_regions = ts.shape[0]
        rng = np.random.RandomState(0)
        shuffled = np.stack([rng.permutation(ts[r]) for r in range(n_regions)])
        baseline_ts_use = [shuffled]
    else:
        # baseline_ts may be provided as a list/tuple of runs (each run is 2D)
        if isinstance(baseline_ts, (list, tuple)):
            baseline_ts_use = list(baseline_ts)
        elif isinstance(baseline_ts, np.ndarray):
            if baseline_ts.ndim == 2:
                baseline_ts_use = [baseline_ts]
            elif baseline_ts.ndim == 3:
                # split 3D array into list of 2D runs
                baseline_ts_use = [baseline_ts[i] for i in range(baseline_ts.shape[0])]
            else:
                raise ValueError(
                    f"baseline_ts must be 2D, 3D or a list/tuple of 2D arrays; got shape {baseline_ts.shape}"
                )
        else:
            raise ValueError(
                "baseline_ts must be a numpy array or a list/tuple of numpy arrays"
            )

    # Use shared bin edges across observed + baseline to keep response alphabet fixed.
    arrays_for_edges = [ts.ravel()] + [run.ravel() for run in baseline_ts_use]
    all_vals = np.concatenate(arrays_for_edges)
    finite_vals = all_vals[np.isfinite(all_vals)]
    if finite_vals.size == 0:
        return 0.0
    vmin = float(np.min(finite_vals))
    vmax = float(np.max(finite_vals))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-12
    edges = np.linspace(vmin, vmax, bins + 1)
    bins_inner = edges[1:-1]
    log2_bins = float(np.log2(bins))

    def _disc(run_ts: np.ndarray) -> np.ndarray:
        x = np.asarray(run_ts, dtype=float)
        x = np.nan_to_num(x, nan=vmin, posinf=vmax, neginf=vmin)
        z = np.digitize(x, bins_inner, right=False)
        return np.clip(z, 0, bins - 1).astype(np.int16)

    def _p_from_disc(disc: np.ndarray) -> np.ndarray:
        cnt = np.bincount(disc.ravel(), minlength=bins).astype(float)
        s = float(cnt.sum())
        if s <= 0:
            return np.ones(bins, dtype=float) / float(bins)
        return cnt / s

    def _jsd(p: np.ndarray, q: np.ndarray) -> float:
        p = p / max(float(np.sum(p)), eps)
        q = q / max(float(np.sum(q)), eps)
        m = 0.5 * (p + q)
        return float(0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2))

    def _conditional_entropy_1step(disc: np.ndarray) -> float:
        n_regions, n_time = disc.shape
        if n_time < 2:
            return 0.0
        cond_vals = []
        for r in range(n_regions):
            prev = disc[r, :-1].astype(np.int64, copy=False)
            nxt = disc[r, 1:].astype(np.int64, copy=False)
            joint = np.zeros((bins, bins), dtype=float)
            np.add.at(joint, (prev, nxt), 1.0)
            total = float(joint.sum())
            if total <= 0:
                cond_vals.append(0.0)
                continue
            p_joint = joint / total
            p_prev = p_joint.sum(axis=1)
            h_joint = float(entropy(p_joint.ravel(), base=2))
            h_prev = float(entropy(p_prev, base=2))
            cond_vals.append(max(h_joint - h_prev, 0.0))
        return float(np.mean(cond_vals)) if cond_vals else 0.0

    def _stability(disc: np.ndarray) -> float:
        n_time = disc.shape[1]
        if n_time < 4:
            return 1.0
        n_seg = int(min(stability_segments, max(2, n_time // 2)))
        idx_segments = np.array_split(np.arange(n_time), n_seg)
        ps = []
        for seg in idx_segments:
            if seg.size == 0:
                continue
            p = _p_from_disc(disc[:, seg])
            ps.append(p)
        if len(ps) < 2:
            return 1.0
        d = []
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                d.append(_jsd(ps[i], ps[j]))
        if not d:
            return 1.0
        return float(np.clip(1.0 - np.mean(d), 0.0, 1.0))

    def _region_probabilities(disc: np.ndarray) -> np.ndarray:
        n_regions = disc.shape[0]
        p = np.empty((n_regions, bins), dtype=float)
        for r in range(n_regions):
            cnt = np.bincount(disc[r], minlength=bins).astype(float)
            s = float(cnt.sum())
            if s <= 0:
                p[r] = 1.0 / float(bins)
            else:
                p[r] = cnt / s
        return p

    def _spatial_repertoire_divergence(disc: np.ndarray) -> float:
        n_regions = disc.shape[0]
        if n_regions < 2:
            return 0.0
        p_reg = _region_probabilities(disc)
        h_reg = entropy(p_reg, axis=1, base=2)
        acc = 0.0
        pairs = 0
        for i in range(n_regions - 1):
            m = 0.5 * (p_reg[i + 1:] + p_reg[i])
            h_m = entropy(m, axis=1, base=2)
            jsd = h_m - 0.5 * (h_reg[i + 1:] + h_reg[i])
            jsd = np.clip(jsd, 0.0, 1.0)
            acc += float(np.sum(jsd))
            pairs += int(jsd.size)
        if pairs <= 0:
            return 0.0
        return float(np.clip(acc / float(pairs), 0.0, 1.0))

    def _effective_dimensionality(run_ts: np.ndarray) -> float:
        x = accelerated_zscore(run_ts, axis=1, backend=backend, eps=eps)
        q = int(min(x.shape))
        if q <= 1:
            return 0.0
        try:
            svals = accelerated_svd_values(x, backend=backend)
        except np.linalg.LinAlgError:
            return 0.0
        var = np.square(svals)
        total = float(np.sum(var))
        if total <= 0:
            return 0.0
        denom = float(np.sum(np.square(var)))
        if denom <= 0:
            return 0.0
        d_eff = (total * total) / (denom + eps)
        d_eff = float(np.clip(d_eff, 1.0, float(q)))
        return float(np.clip((d_eff - 1.0) / (float(q) - 1.0), 0.0, 1.0))

    n_perm = int(math.factorial(ordinal_order))
    perm_uniform = np.ones(n_perm, dtype=float) / float(n_perm)
    perm_log_norm = float(np.log2(max(2, n_perm)))
    perm_base = np.power(
        ordinal_order,
        np.arange(ordinal_order - 1, -1, -1, dtype=np.int64),
        dtype=np.int64,
    )
    perm_lookup = np.full(int(ordinal_order ** ordinal_order), -1, dtype=np.int64)
    perms = np.asarray(list(itertools.permutations(range(ordinal_order))), dtype=np.int64)
    perm_codes = np.sum(perms * perm_base, axis=1, dtype=np.int64)
    perm_lookup[perm_codes] = np.arange(n_perm, dtype=np.int64)
    delta = np.zeros(n_perm, dtype=float)
    delta[0] = 1.0
    jsd_max_perm = max(_jsd(delta, perm_uniform), eps)

    def _ordinal_prob_1d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size < ordinal_order:
            return None
        win = np.lib.stride_tricks.sliding_window_view(x, ordinal_order)
        if win.shape[0] == 0:
            return None
        ranks = np.argsort(win, axis=1, kind="mergesort")
        codes = np.sum(ranks * perm_base, axis=1, dtype=np.int64)
        ids = perm_lookup[codes]
        ids = ids[ids >= 0]
        if ids.size == 0:
            return None
        cnt = np.bincount(ids, minlength=n_perm).astype(float)
        s = float(np.sum(cnt))
        if s <= 0:
            return None
        return cnt / s

    def _multiscale_ordinal_complexity(run_ts: np.ndarray) -> float:
        x = np.asarray(run_ts, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        n_regions, n_time = x.shape
        if n_time < ordinal_order + 1:
            return 0.0
        max_scale_eff = int(min(multiscale_max_scale, n_time // (ordinal_order + 1)))
        if max_scale_eff < 1:
            return 0.0
        vals = []
        for scale in range(1, max_scale_eff + 1):
            usable = (n_time // scale) * scale
            if usable < (ordinal_order + 1):
                continue
            coarse = x[:, :usable].reshape(n_regions, usable // scale, scale).mean(axis=2)
            for r in range(n_regions):
                p_ord = _ordinal_prob_1d(coarse[r])
                if p_ord is None:
                    continue
                h_norm = float(entropy(p_ord, base=2) / (perm_log_norm + eps))
                h_norm = float(np.clip(h_norm, 0.0, 1.0))
                q_jsd = _jsd(p_ord, perm_uniform) / jsd_max_perm
                q_jsd = float(np.clip(q_jsd, 0.0, 1.0))
                vals.append(h_norm * q_jsd)
        if not vals:
            return 0.0
        return float(np.clip(np.mean(vals), 0.0, 1.0))

    def _noise_index(run_ts: np.ndarray) -> float:
        x = np.asarray(run_ts, dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        mad_x = median_abs_deviation(x, axis=1, scale=1.0)
        if x.shape[1] < 2:
            mad_d = np.zeros(x.shape[0], dtype=float)
        else:
            dx = np.diff(x, axis=1)
            mad_d = median_abs_deviation(dx, axis=1, scale=1.0)
        ratio = mad_d / (mad_x + eps)
        ratio = ratio[np.isfinite(ratio)]
        if ratio.size == 0:
            return 0.0
        return float(np.median(ratio))

    def _run_stats(run_ts: np.ndarray) -> dict:
        if run_ts.ndim != 2:
            raise ValueError(
                f"each run should be 2D (n_regions×n_time), got shape {run_ts.shape}"
            )
        disc = _disc(run_ts)
        p = _p_from_disc(disc)
        H = float(entropy(p, base=2))
        h = _conditional_entropy_1step(disc)
        xi_norm = float(np.clip((H - h) / (log2_bins + eps), 0.0, 1.0))
        delta_spatial = _spatial_repertoire_divergence(disc)
        d_eff_norm = _effective_dimensionality(run_ts)
        c_multi = _multiscale_ordinal_complexity(run_ts)
        S = _stability(disc)
        nu = _noise_index(run_ts)
        return {
            "H": H,
            "h": h,
            "xi_norm": xi_norm,
            "delta_spatial": delta_spatial,
            "d_eff_norm": d_eff_norm,
            "c_multi": c_multi,
            "S": S,
            "nu": nu,
        }

    obs = _run_stats(ts)
    base_stats = [_run_stats(run) for run in baseline_ts_use]
    if not base_stats:
        base_stats = [
            {
                "xi_norm": 0.0,
                "delta_spatial": 0.0,
                "d_eff_norm": 0.0,
                "c_multi": 0.0,
                "S": 1.0,
                "nu": 0.0,
            }
        ]

    if weighted:
        w = np.asarray([run.size for run in baseline_ts_use], dtype=float)
    else:
        w = np.ones(len(base_stats), dtype=float)
    if np.sum(w) <= 0:
        w = np.ones(len(base_stats), dtype=float)
    w = w / float(np.sum(w))

    x_xi_arr = np.asarray([d["xi_norm"] for d in base_stats], dtype=float)
    x_delta_arr = np.asarray([d["delta_spatial"] for d in base_stats], dtype=float)
    x_deff_arr = np.asarray([d["d_eff_norm"] for d in base_stats], dtype=float)
    x_cmulti_arr = np.asarray([d["c_multi"] for d in base_stats], dtype=float)
    s_arr = np.asarray([d["S"] for d in base_stats], dtype=float)
    nu_arr = np.asarray([d["nu"] for d in base_stats], dtype=float)

    x_xi_base = float(np.sum(w * x_xi_arr))
    x_delta_base = float(np.sum(w * x_delta_arr))
    x_deff_base = float(np.sum(w * x_deff_arr))
    x_cmulti_base = float(np.sum(w * x_cmulti_arr))
    S_base = float(np.sum(w * s_arr))
    nu_base = float(np.sum(w * nu_arr))
    x_xi_sd = float(np.sqrt(np.sum(w * (x_xi_arr - x_xi_base) ** 2)))
    x_delta_sd = float(np.sqrt(np.sum(w * (x_delta_arr - x_delta_base) ** 2)))
    x_deff_sd = float(np.sqrt(np.sum(w * (x_deff_arr - x_deff_base) ** 2)))
    x_cmulti_sd = float(np.sqrt(np.sum(w * (x_cmulti_arr - x_cmulti_base) ** 2)))

    contrasts = np.asarray(
        [
            (obs["xi_norm"] - x_xi_base) / (1.0 + x_xi_sd),
            (obs["delta_spatial"] - x_delta_base) / (1.0 + x_delta_sd),
            (obs["d_eff_norm"] - x_deff_base) / (1.0 + x_deff_sd),
            (obs["c_multi"] - x_cmulti_base) / (1.0 + x_cmulti_sd),
        ],
        dtype=float,
    )
    contrasts = np.clip(contrasts, -1.0, 1.0)
    gain_pos = np.clip(contrasts, 0.0, 1.0)
    core_positive = float(np.sum(comp_w * gain_pos))
    core_signed = float(np.sum(comp_w * contrasts))

    # Stability correction: if task stability is below baseline stability,
    # down-weight the gain.
    c_stab = float(np.clip(obs["S"] / (S_base + eps), 0.0, 1.0))

    # Differential-noise correction from robust first-difference ratio.
    c_noise = float(np.exp(-noise_penalty_kappa * max(0.0, obs["nu"] - nu_base)))

    core = core_positive if clip_negative else core_signed
    pdi_raw = core * c_stab * c_noise
    if clip_negative:
        pdi_raw = max(float(pdi_raw), 0.0)

    if not normalize:
        return float(pdi_raw)

    # bounded normalization in [0,1]
    if clip_negative:
        return float(np.clip(pdi_raw, 0.0, 1.0))
    return float(np.clip(0.5 * (pdi_raw + 1.0), 0.0, 1.0))

def compute_NAS(
    ts: np.ndarray,
    tr: float = None,
    zthr: float = 1.0,
    eps: float = 0.2,
    tau: float = None,
    lambda_phase: float = 0.5,
    alpha: float = 0.20,
    beta: float = 0.16,
    gamma: float = 0.14,
    delta: float = 0.12,
    eta: float = 0.16,
    zeta: float = 0.12,
    rho: float = 0.10,
    bands: list = None,
    band_weights: list = None,
    window_len: int = None,
    step_len: int = None,
    max_triads: int = 5000,
    random_state: int = 0,
    workspace_nodes: np.ndarray = None,
    workspace_quantile: float = 0.2,
    workspace_min_size: int = 4,
    directed_lag: int = 1,
    reverberation_lags: tuple = (2, 3, 4),
    baseline_ts: np.ndarray = None,
    boost_against_baseline: bool = False,
    normalize: bool = True,
    hardware_backend=None,
) -> float:
    """
    Theory-aligned Network Activation Synchrony (NAS).

    NAS operationalizes GNW-like broadcast dynamics using seven measurable
    components from EEG/fMRI node×time series:
      1) L: intra-broadcast synchrony,
      2) B: broadcast reach to non-broadcast nodes,
      3) H: higher-order triadic closure in broadcast nodes,
      4) D: temporal stability of synchrony graphs,
      5) W: dedicated-workspace recruitment with ignition boost,
      6) E: directed broadcast efficacy (lagged asymmetry),
      7) R: workspace reverberatory persistence (reportability proxy).

    Components are computed per band and aggregated by weighted geometric mean.

    Parameters
    ----------
    ts : ndarray, shape (n_regions, n_time)
        Timeseries matrix.
    tr : float
        Sample interval in seconds. Must be explicit and > 0.
    zthr : float, optional
        Legacy parameter kept for backward compatibility.
    eps : float, optional
        Legacy compatibility parameter (not used in current strict estimator).
    tau : float
        Broadcast-node quantile tolerance in [0,1). Broadcast set is
        strength >= quantile(1-tau).
    lambda_phase : float, optional
        Mixing factor between PLV and envelope coupling in [0,1].
    alpha, beta, gamma, delta, eta, zeta, rho : float, optional
        Exponents for (L,B,H,D,W,E,R). Automatically normalized to sum to 1.
    bands : list[(lo,hi)]
        Explicit frequency bands in Hz.
    band_weights : list[float]
        Non-negative weights for each band. Normalized internally.
    window_len : int
        Window length in samples.
    step_len : int
        Window step in samples.
    max_triads : int, optional
        Maximum number of triads sampled per window for H computation.
    random_state : int, optional
        Seed for triad sampling.
    workspace_nodes : array-like[int], optional
        Optional predefined dedicated-workspace node indices. If None,
        workspace is inferred from global synchrony-centrality quantile.
    workspace_quantile : float, optional
        Fraction of top-central nodes used when inferring workspace.
    workspace_min_size : int, optional
        Minimum number of nodes in inferred workspace.
    directed_lag : int, optional
        Lag (samples) for directed broadcast asymmetry.
    reverberation_lags : tuple[int], optional
        Positive lags used for workspace reverberation persistence.
    baseline_ts : ndarray, optional
        Baseline timeseries for optional boost computation.
    boost_against_baseline : bool, optional
        If True and ``baseline_ts`` is provided, returns max(NAS - NAS_base, 0).
    normalize : bool, optional
        If True, clip final value to [0,1].

    Returns
    -------
    float
        NAS value in [0,1] when ``normalize=True``.
    """
    if ts.ndim != 2:
        raise ValueError(f"ts should be 2D (n_regions × n_time), got shape {ts.shape}")
    _ = (zthr, eps)  # retained only for backward compatibility
    n_regions, n_time = ts.shape
    if n_regions < 2 or n_time < 4:
        return 0.0
    backend = resolve_hardware_backend(hardware_backend)

    if tr is None or (not np.isfinite(tr)) or float(tr) <= 0:
        raise ValueError("compute_NAS requires explicit positive tr")
    if tau is None:
        raise ValueError("compute_NAS requires explicit tau")
    if not (0.0 <= tau < 1.0):
        raise ValueError("tau must be in [0, 1)")
    if not (0.0 <= lambda_phase <= 1.0):
        raise ValueError("lambda_phase must be in [0,1]")
    if max_triads < 1:
        raise ValueError("max_triads must be >= 1")
    if directed_lag < 1:
        raise ValueError("directed_lag must be >= 1")
    if not (0.0 < workspace_quantile <= 1.0):
        raise ValueError("workspace_quantile must be in (0,1]")
    if workspace_min_size < 2:
        raise ValueError("workspace_min_size must be >= 2")

    comp_w = np.array([alpha, beta, gamma, delta, eta, zeta, rho], dtype=float)
    if np.any(comp_w < 0):
        raise ValueError("alpha/beta/gamma/delta/eta/zeta/rho must be non-negative")
    if float(comp_w.sum()) <= 0:
        raise ValueError("sum of NAS component exponents must be > 0")
    comp_w = comp_w / float(comp_w.sum())
    alpha, beta, gamma, delta, eta, zeta, rho = [float(v) for v in comp_w]

    # Node-wise z-scoring for scale robustness.
    x = np.asarray(ts, dtype=float)
    means = np.nanmean(x, axis=1, keepdims=True)
    stds = np.nanstd(x, axis=1, ddof=0, keepdims=True) + 1e-12
    x = (x - means) / stds
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    fs = 1.0 / float(tr)

    if bands is None:
        raise ValueError("compute_NAS requires explicit frequency bands")
    band_specs = []
    for b in bands:
        if b is None:
            raise ValueError("compute_NAS does not allow None band entries")
        if (not isinstance(b, (tuple, list))) or len(b) != 2:
            raise ValueError("bands must be a list of (low, high) tuples")
        lo, hi = b
        lo = None if lo is None else float(lo)
        hi = None if hi is None else float(hi)
        if lo is None or hi is None:
            raise ValueError("compute_NAS requires finite low/high for all bands")
        band_specs.append((lo, hi))
    if not band_specs:
        raise ValueError("compute_NAS requires a non-empty bands list")

    if band_weights is None:
        raise ValueError("compute_NAS requires explicit band_weights")
    bw = np.asarray(band_weights, dtype=float)
    if bw.shape[0] != len(band_specs):
        raise ValueError("band_weights length must match number of bands")
    if np.any(bw < 0) or float(bw.sum()) <= 0:
        raise ValueError("band_weights must be non-negative and sum > 0")
    bw = bw / float(bw.sum())

    if window_len is None:
        raise ValueError("compute_NAS requires explicit window_len")
    wlen = int(window_len)
    if wlen < 4:
        raise ValueError("window_len must be >= 4")
    if wlen > n_time:
        raise ValueError(f"window_len={wlen} exceeds run length n_time={n_time}")
    if step_len is None:
        raise ValueError("compute_NAS requires explicit step_len")
    step = int(step_len)
    if step < 1:
        raise ValueError("step_len must be >= 1")
    if step > wlen:
        raise ValueError("step_len must be <= window_len")

    starts = list(range(0, max(1, n_time - wlen + 1), step))
    if not starts:
        starts = [0]
    if starts[-1] != max(0, n_time - wlen):
        starts.append(max(0, n_time - wlen))
    starts = sorted(set(starts))

    rng_master = np.random.RandomState(int(random_state))

    def _global_workspace_indices(x_full: np.ndarray) -> np.ndarray:
        if workspace_nodes is not None:
            idx = np.asarray(workspace_nodes, dtype=int).reshape(-1)
            idx = idx[(idx >= 0) & (idx < n_regions)]
            idx = np.unique(idx)
            if idx.size >= 2:
                return idx
        A0 = np.abs(accelerated_corrcoef(x_full, backend=backend))
        if A0.ndim != 2:
            A0 = np.zeros((n_regions, n_regions), dtype=float)
        A0 = np.nan_to_num(A0, nan=0.0, posinf=1.0, neginf=0.0)
        A0 = np.clip(A0, 0.0, 1.0)
        np.fill_diagonal(A0, 0.0)
        strength0 = A0.sum(axis=1) / float(max(1, n_regions - 1))
        k = int(np.ceil(float(workspace_quantile) * float(n_regions)))
        k = max(int(workspace_min_size), k)
        k = min(max(2, k), n_regions)
        idx = np.argpartition(strength0, -k)[-k:]
        return np.sort(np.asarray(idx, dtype=int))

    G = _global_workspace_indices(x)
    G_set = set(G.tolist())

    def _band_filter(data, lo, hi, fs_local):
        if lo is None or hi is None or fs_local is None:
            raise ValueError("compute_NAS requires valid band edges and tr/fs")
        nyq = 0.5 * fs_local
        if lo <= 0 or hi <= lo or hi >= nyq:
            raise ValueError(
                f"compute_NAS invalid band ({lo}, {hi}) for Nyquist {nyq}"
            )
        wn = [lo / nyq, hi / nyq]
        try:
            sos = butter(2, wn, btype="band", output="sos")
            fil = sosfiltfilt(sos, data, axis=1)
            return fil, True
        except ValueError:
            raise ValueError(f"compute_NAS bandpass failed for band ({lo}, {hi})")

    def _window_adjacency(data_w, use_phase, lambda_mix):
        if use_phase:
            analytic = hilbert(data_w, axis=1)
            phase = np.angle(analytic)
            env = np.abs(analytic)

            zc = np.exp(1j * phase)
            plv = np.abs(
                accelerated_dot(zc, zc.conj().T, backend=backend) / float(data_w.shape[1])
            ).astype(float)
            ecoh = np.abs(accelerated_corrcoef(env, backend=backend))
            A = lambda_mix * plv + (1.0 - lambda_mix) * ecoh
        else:
            A = np.abs(accelerated_corrcoef(data_w, backend=backend))

        if A.ndim != 2:
            A = np.zeros((n_regions, n_regions), dtype=float)
        A = np.nan_to_num(A, nan=0.0, posinf=1.0, neginf=0.0)
        A = np.clip(A, 0.0, 1.0)
        np.fill_diagonal(A, 0.0)
        return A

    def _triadic_intensity(Avv, max_n, rng_local):
        nV = Avv.shape[0]
        if nV < 3:
            return 0.0
        total = nV * (nV - 1) * (nV - 2) // 6
        vals = []
        if total <= max_n:
            for i, j, k in itertools.combinations(range(nV), 3):
                vals.append((Avv[i, j] * Avv[i, k] * Avv[j, k]) ** (1.0 / 3.0))
        else:
            seen = set()
            max_attempts = int(max_n * 25)
            attempts = 0
            while len(vals) < max_n and attempts < max_attempts:
                tri = tuple(sorted(rng_local.choice(nV, size=3, replace=False).tolist()))
                attempts += 1
                if tri in seen:
                    continue
                seen.add(tri)
                i, j, k = tri
                vals.append((Avv[i, j] * Avv[i, k] * Avv[j, k]) ** (1.0 / 3.0))
        if not vals:
            return 0.0
        return float(np.mean(vals))

    def _directed_broadcast_advantage(data_w: np.ndarray, V: np.ndarray) -> float:
        if V.size == 0:
            return 0.0
        lag = int(directed_lag)
        if data_w.shape[1] <= (lag + 2):
            return 0.0
        x0 = data_w[:, :-lag]
        x1 = data_w[:, lag:]
        x0 = x0 - np.mean(x0, axis=1, keepdims=True)
        x1 = x1 - np.mean(x1, axis=1, keepdims=True)
        x0 = x0 / (np.std(x0, axis=1, keepdims=True) + 1e-12)
        x1 = x1 / (np.std(x1, axis=1, keepdims=True) + 1e-12)
        C_lag = accelerated_dot(x0, x1.T, backend=backend) / float(max(1, x0.shape[1]))
        C_lag = np.nan_to_num(C_lag, nan=0.0, posinf=0.0, neginf=0.0)
        A_dir = np.maximum(C_lag - C_lag.T, 0.0)
        np.fill_diagonal(A_dir, 0.0)

        Vbar = np.setdiff1d(np.arange(n_regions), V, assume_unique=True)
        if Vbar.size == 0:
            return 0.0
        out = float(np.mean(A_dir[np.ix_(V, Vbar)]))
        back = float(np.mean(A_dir[np.ix_(Vbar, V)]))
        return float(np.clip((out - back) / (out + back + 1e-12), 0.0, 1.0))

    rev_lags = np.asarray([int(l) for l in reverberation_lags], dtype=int)
    rev_lags = rev_lags[rev_lags > 0]

    nas_bands = []
    for b_idx, (flo, fhi) in enumerate(band_specs):
        xb, can_phase = _band_filter(x, flo, fhi, fs)
        use_phase = bool(can_phase and xb.shape[1] >= 8)
        if not use_phase:
            raise ValueError(
                "compute_NAS requires phase/envelope synchrony; "
                "bandpass unavailable or window too short (<8 samples)"
            )

        L_vals, B_vals, H_vals = [], [], []
        W_vals, E_vals, R_vals = [], [], []
        A_seq = []
        for w_idx, st in enumerate(starts):
            en = min(st + wlen, n_time)
            data_w = xb[:, st:en]
            if data_w.shape[1] < 4:
                continue

            A = _window_adjacency(data_w, use_phase=use_phase, lambda_mix=lambda_phase)
            A_seq.append(A)

            strength = A.sum(axis=1) / float(max(1, n_regions - 1))
            q = float(np.quantile(strength, 1.0 - tau))
            V = np.where(strength >= q)[0]
            if V.size < 2:
                L_vals.append(0.0)
                B_vals.append(0.0)
                H_vals.append(0.0)
                W_vals.append(0.0)
                E_vals.append(0.0)
                R_vals.append(0.0)
                continue

            Avv = A[np.ix_(V, V)]
            iu = np.triu_indices(V.size, k=1)
            L = float(Avv[iu].mean()) if iu[0].size > 0 else 0.0

            Vbar = np.setdiff1d(np.arange(n_regions), V, assume_unique=True)
            if Vbar.size > 0:
                B = float(A[np.ix_(V, Vbar)].mean())
            else:
                B = 0.0

            rng_local = np.random.RandomState(
                int(rng_master.randint(0, 2**31 - 1) + 31 * b_idx + 7 * w_idx)
            )
            H = _triadic_intensity(Avv, max_n=max_triads, rng_local=rng_local)

            Vbar = np.setdiff1d(np.arange(n_regions), V, assume_unique=True)
            n_hit = int(sum(1 for i in V if i in G_set))
            recruit = float(n_hit / float(max(1, G.size)))
            if Vbar.size >= 2:
                A_nn = A[np.ix_(Vbar, Vbar)]
                iu_nn = np.triu_indices(Vbar.size, k=1)
                non_sync = float(A_nn[iu_nn].mean()) if iu_nn[0].size > 0 else 0.0
            else:
                non_sync = 0.0
            ignition = float(np.clip((L - non_sync) / (1.0 - non_sync + 1e-12), 0.0, 1.0))
            W = float(np.clip(recruit * ignition, 0.0, 1.0))

            E = _directed_broadcast_advantage(data_w, V)

            g_sig = np.mean(data_w[G, :], axis=0) if G.size > 0 else np.mean(data_w, axis=0)
            g_sig = g_sig - float(np.mean(g_sig))
            g_den = float(np.linalg.norm(g_sig))
            rev_vals = []
            if g_den > 0 and rev_lags.size > 0:
                for lag in rev_lags.tolist():
                    if lag >= g_sig.shape[0] - 1:
                        continue
                    a = g_sig[:-lag]
                    b = g_sig[lag:]
                    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                    if den <= 0:
                        continue
                    rev_vals.append(max(float(np.dot(a, b) / den), 0.0))
            rev = float(np.mean(rev_vals)) if rev_vals else 0.0
            R = float(np.clip(recruit * rev, 0.0, 1.0))

            L_vals.append(L)
            B_vals.append(B)
            H_vals.append(H)
            W_vals.append(W)
            E_vals.append(E)
            R_vals.append(R)

        if not L_vals:
            nas_bands.append(0.0)
            continue

        L_bar = float(np.mean(L_vals))
        B_bar = float(np.mean(B_vals))
        H_bar = float(np.mean(H_vals))
        W_bar = float(np.mean(W_vals))
        E_bar = float(np.mean(E_vals))
        R_bar = float(np.mean(R_vals))

        if len(A_seq) <= 1:
            D = 1.0
        else:
            dvals = []
            for A0, A1 in zip(A_seq[:-1], A_seq[1:]):
                num = float(np.linalg.norm(A1 - A0, ord="fro"))
                den = float(np.linalg.norm(A1, ord="fro") + np.linalg.norm(A0, ord="fro") + 1e-12)
                dvals.append(num / den)
            D = float(np.clip(1.0 - np.mean(dvals), 0.0, 1.0))

        comps = np.asarray(
            [
                max(L_bar, 0.0),
                max(B_bar, 0.0),
                max(H_bar, 0.0),
                max(D, 0.0),
                max(W_bar, 0.0),
                max(E_bar, 0.0),
                max(R_bar, 0.0),
            ],
            dtype=float,
        )
        if np.any((comps <= 0.0) & (comp_w > 0.0)):
            nas_f = 0.0
        else:
            nas_f = float(np.exp(np.sum(comp_w * np.log(comps + 1e-12))))
        nas_bands.append(float(nas_f))

    nas_val = float(np.dot(bw, np.asarray(nas_bands, dtype=float)))
    if boost_against_baseline and baseline_ts is not None:
        nas_base = compute_NAS(
            baseline_ts,
            tr=tr,
            zthr=zthr,
            eps=eps,
            tau=tau,
            lambda_phase=lambda_phase,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            eta=eta,
            zeta=zeta,
            rho=rho,
            bands=band_specs,
            band_weights=bw.tolist(),
            window_len=wlen,
            step_len=step,
            max_triads=max_triads,
            random_state=random_state,
            workspace_nodes=G,
            workspace_quantile=workspace_quantile,
            workspace_min_size=workspace_min_size,
            directed_lag=directed_lag,
            reverberation_lags=tuple(int(v) for v in rev_lags.tolist()),
            baseline_ts=None,
            boost_against_baseline=False,
            normalize=False,
            hardware_backend=backend,
        )
        nas_val = max(nas_val - nas_base, 0.0)

    if normalize:
        return float(np.clip(nas_val, 0.0, 1.0))
    return float(nas_val)


def compute_IIM(
    ts: np.ndarray,
    bins: int = 3,
    lag_trs: int = 1,
    n_parts: int | None = None,
    rng: int = 0,
    method: str = "causal",
    partition_mode: str = "all",
    clamp: bool = True,
    scale: float = 1.0,
    max_nodes: int | None = None,
    max_mechanism_size: int | None = None,
    max_purview_size: int | None = None,
    tpm_alpha: float = 1e-3,
    max_state_space: int = 1500,
    return_details: bool = False,
    checkpoint_path: str | None = None,
    resume_from_checkpoint: bool = True,
    checkpoint_every_cuts: int = 1,
    progress_log_every_cuts: int = 1,
    progress_label: str | None = None,
    phase1_parallel_workers: int | None = None,
    phase1_chunk_size: int = 8,
    phase1_shared_memory: bool = True,
    use_induced_partition_cache: bool = True,
    kernel_cache_path: str | None = None,
    kernel_cache_memory_entries: int = 300_000,
    kernel_cache_flush_batch: int = 5_000,
    hardware_backend=None,
) -> float:
    """
    IIT-leaning Integrated Information Metric (IIM) from an empirical causal TPM.

    The method estimates:
      1) a discrete transition probability matrix (TPM) over a reduced node set,
      2) mechanism-level irreducibility via cause/effect repertoires, and
      3) system-level irreducibility by Minimum Information Partition (MIP)
         over bipartition cuts.

    Let Ψ be the weighted sum of mechanism irreducibility terms in the intact
    TPM and Ψ^κ the same quantity under cut κ. We evaluate:

        IIM_raw = (Ψ - max_k Ψ^κ) / (Ψ + eps)

    where max_k Ψ^κ corresponds to the cut that preserves the most causal
    structure (MIP under this preserved-structure form). Canonical mapping:

        IIM_can = clip(IIM_raw, 0, 1) # CI-facing bounded integration score

    keeps the CI-facing term in [0, 1] while retaining signed raw diagnostics.

    Parameters
    ----------
    ts : ndarray, shape (n_regions, n_time)
        Timeseries matrix.
    bins : int, optional
        Target number of discrete levels per node. May be reduced internally
        to satisfy ``max_state_space``.
    lag_trs : int, optional
        Lag (in samples) for transition construction.
    n_parts : int or None, optional
        Number of system cuts evaluated for MIP search. If ``None`` (default),
        all possible unique bipartitions are evaluated (exhaustive MIP search).
    rng : int or np.random.RandomState, optional
        Seed or RandomState used for sampled cuts.
    method : {'causal', 'random'}, optional
        Causal TPM method selector. ``'random'`` is accepted as legacy alias
        for ``'causal'``.
    partition_mode : {'all', 'balanced'}, optional
        Cut candidate regime. ``'balanced'`` restricts to near-equal cuts.
    clamp : bool, optional
        If True (default), return canonical IIM in [0,1]. If False, return raw.
    scale : float, optional
        Multiplicative factor applied to the returned metric value.
    max_nodes : int or None, optional
        Max number of nodes retained for causal TPM estimation. If ``None``,
        no explicit node-count cap is applied before the state-space budget
        enforcement.
    max_mechanism_size : int or None, optional
        Largest mechanism size included in Ψ. If ``None`` (default), all
        mechanism sizes up to the selected subsystem size are included.
    max_purview_size : int or None, optional
        Largest purview size included in Ψ. If ``None`` (default), all purview
        sizes up to the selected subsystem size are included.
    tpm_alpha : float, optional
        Laplace smoothing for TPM row estimation.
    max_state_space : int, optional
        Upper bound on number of discrete system states used in TPM.
    return_details : bool, optional
        When True, return diagnostics dictionary.
    checkpoint_path : str or None, optional
        Path to JSON checkpoint file. When provided, per-cut progress is
        persisted and can be resumed on a later run.
    resume_from_checkpoint : bool, optional
        If True (default), reuse compatible checkpoint progress when
        ``checkpoint_path`` exists.
    checkpoint_every_cuts : int, optional
        Persist checkpoint after every N newly completed cuts (default 1).
    progress_log_every_cuts : int, optional
        Emit cut-level progress logs every N completed cuts (default 1).
    progress_label : str or None, optional
        Optional short label included in progress logs (e.g. run/file id).
    phase1_parallel_workers : int or None, optional
        Number of worker processes used for mechanism-chunk parallelization
        inside each Ψ computation (phase 1 and per-cut Ψ in phase 2). ``None``
        or values <= 1 disable intra-task parallelization.
    phase1_chunk_size : int, optional
        Number of mechanisms per submitted worker chunk when intra-task
        parallelization is enabled.
    phase1_shared_memory : bool, optional
        If True (default), stage read-only TPM/state arrays in shared memory
        for intra-task workers to minimize RAM duplication.
    use_induced_partition_cache : bool, optional
        Deprecated opt-out flag. Induced-partition caching is always enabled
        by default in this implementation and this flag is ignored.
    kernel_cache_path : str or None, optional
        SQLite path for disk-backed induced-partition cache. If omitted, a
        checkpoint-adjacent path (when checkpointing) or temp file is used.
    kernel_cache_memory_entries : int, optional
        In-memory LRU front-cache size for the kernel cache.
    kernel_cache_flush_batch : int, optional
        Number of pending inserts before batched SQLite flush.

    Returns
    -------
    float or dict
        Scaled scalar value (default), or details when requested.
    """
    backend = resolve_hardware_backend(hardware_backend)

    # ---------- local functions ----------
    def _encode_vals(vals, base):
        key = 0
        for v in vals:
            key = key * base + int(v)
        return int(key)

    def _decode_key(key, k, base):
        out = [0] * k
        v = int(key)
        for i in range(k - 1, -1, -1):
            out[i] = v % base
            v //= base
        return tuple(out)

    def _subset_key_matrix(states, subset, base):
        if len(subset) == 0:
            return np.zeros(states.shape[0], dtype=np.int64)
        cols = states[:, subset].astype(np.int64, copy=False)
        mult = (base ** np.arange(len(subset) - 1, -1, -1, dtype=np.int64))
        return (cols * mult).sum(axis=1).astype(np.int64)

    def _enumerate_subsets(nodes, max_size):
        out = []
        for r in range(1, min(max_size, len(nodes)) + 1):
            out.extend(tuple(c) for c in itertools.combinations(nodes, r))
        return out

    def _enumerate_bipartitions(nodes):
        n = len(nodes)
        if n < 2:
            return []
        out = []
        node_set = tuple(nodes)
        for k in range(1, (n // 2) + 1):
            for A in itertools.combinations(node_set, k):
                A = tuple(A)
                B = tuple(x for x in node_set if x not in A)
                if k == n - k and A[0] > B[0]:
                    continue
                out.append((A, B))
        return out

    def _jsd(p, q):
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        p = p / max(p.sum(), 1e-12)
        q = q / max(q.sum(), 1e-12)
        m = 0.5 * (p + q)
        return float(0.5 * entropy(p, m, base=2) + 0.5 * entropy(q, m, base=2))

    def _discretize_per_node(arr, base_bins):
        n, T = arr.shape
        out = np.zeros((n, T), dtype=np.int16)
        for i in range(n):
            x = arr[i]
            finite = x[np.isfinite(x)]
            if finite.size == 0:
                continue
            vmin = float(np.min(finite))
            vmax = float(np.max(finite))
            if np.isclose(vmin, vmax):
                continue
            edges = np.quantile(finite, np.linspace(0.0, 1.0, base_bins + 1))
            if np.unique(edges).size != base_bins + 1:
                edges = np.linspace(vmin, vmax, base_bins + 1)
            bins_inner = edges[1:-1]
            z = np.digitize(x, bins_inner, right=False)
            z = np.clip(z, 0, base_bins - 1)
            z[~np.isfinite(x)] = 0
            out[i] = z.astype(np.int16)
        return out

    def _select_nodes(arr, max_n):
        n = arr.shape[0]
        if n <= max_n:
            return np.arange(n, dtype=int)
        var = np.nanvar(arr, axis=1)
        var = np.where(np.isfinite(var), var, -np.inf)
        idx = np.argsort(var)[::-1][:max_n]
        return np.sort(idx.astype(int))

    def _build_states_and_tpm(disc, base, lag, alpha):
        n, T = disc.shape
        T_eff = T - lag
        curr = disc[:, :T_eff].T.astype(np.int16, copy=False)
        nxt = disc[:, lag:].T.astype(np.int16, copy=False)

        full_subset = tuple(range(n))
        curr_keys = _subset_key_matrix(curr, full_subset, base)
        nxt_keys = _subset_key_matrix(nxt, full_subset, base)

        n_states = int(base ** n)
        C = np.zeros((n_states, n_states), dtype=float)
        np.add.at(C, (curr_keys, nxt_keys), 1.0)
        row_sum = C.sum(axis=1, keepdims=True)
        tpm = (C + alpha) / (row_sum + alpha * n_states)

        sid = np.arange(n_states, dtype=np.int64)[:, None]
        powv = (base ** np.arange(n - 1, -1, -1, dtype=np.int64))[None, :]
        states_full = ((sid // powv) % base).astype(np.int16)
        return curr, tpm, states_full

    def _marginalize(dist_full, keys, n_keys):
        p = np.bincount(keys, weights=dist_full, minlength=n_keys).astype(float)
        s = p.sum()
        if s <= 0:
            return np.ones(n_keys, dtype=float) / float(n_keys)
        return p / s

    def _build_phase1_chunks_adaptive(
        remaining_mechanisms,
        max_chunk_size,
        workers,
    ):
        """
        Build mechanism chunks with adaptive sizes.

        Strategy:
        - keep chunks as large as possible (to reduce scheduling overhead),
        - keep worker utilization high.

        Chunks are contiguous in mechanism order (resume-safe with prefix-based
        checkpoint progress). Piecewise policy requested for phase-1 throughput:
        - first 50% of remaining mechanisms: target 4 mechanisms per worker task,
        - next 30%: target 2 mechanisms per worker task,
        - final 20% (last 1/5): target 1 mechanism per worker task.

        We still keep enough tasks to occupy available workers whenever possible.
        """
        mechs = tuple(remaining_mechanisms)
        if not mechs:
            return []
        max_chunk = max(1, int(max_chunk_size))
        workers_eff = max(1, int(workers))
        if workers_eff <= 1:
            return [
                tuple(mechs[i:i + max_chunk])
                for i in range(0, len(mechs), max_chunk)
            ]

        chunks = []
        idx = 0
        n_total = int(len(mechs))
        while idx < n_total:
            rem = int(n_total - idx)
            active_workers = max(1, min(workers_eff, rem))

            progress = float(idx) / float(max(1, n_total))
            if progress >= 0.80:
                stage_target = 1
            elif progress >= 0.50:
                stage_target = 2
            else:
                stage_target = 4

            stage_target = min(int(stage_target), int(max_chunk))
            # Keep enough chunks for all active workers when possible.
            c = min(stage_target, max(1, rem // active_workers))
            c = max(1, min(int(c), rem))
            chunks.append(tuple(mechs[idx:idx + c]))
            idx += c
        return chunks

    phase_parallel_runtime_enabled = True

    def _compute_psi(
        tpm,
        curr_obs,
        states_full,
        base,
        mech_size,
        purv_size,
        resume_done_mechanisms=0,
        resume_psi_partial=np.nan,
        progress_cb=None,
        phase1_parallel_workers=None,
        phase1_chunk_size=8,
        phase1_shared_memory=True,
        mechanisms=None,
        purviews=None,
        static_cache=None,
        obs_state_cache=None,
        parallel_runtime=None,
        kernel_cache=None,
        cut_mask_a=None,
        use_induced_partition_cache=False,
        kernel_cache_lookup_only=False,
    ):
        nonlocal phase_parallel_runtime_enabled
        _, n_nodes = states_full.shape
        if mechanisms is None:
            all_nodes = tuple(range(n_nodes))
            mechanisms = tuple(_enumerate_subsets(all_nodes, mech_size))
        else:
            mechanisms = tuple(tuple(sorted(m)) for m in mechanisms)
        if purviews is None:
            all_nodes = tuple(range(n_nodes))
            purviews = tuple(_enumerate_subsets(all_nodes, purv_size))
        else:
            purviews = tuple(tuple(sorted(z)) for z in purviews)
        if not mechanisms or not purviews:
            return 0.0

        # empirical mechanism-state frequencies from observed current states
        try:
            done_start = int(resume_done_mechanisms)
        except (TypeError, ValueError):
            done_start = 0
        done_start = max(0, min(done_start, int(len(mechanisms))))
        try:
            psi = float(resume_psi_partial)
        except (TypeError, ValueError):
            psi = np.nan
        if not np.isfinite(psi):
            psi = 0.0
        n_mechanisms = int(len(mechanisms))
        if done_start >= n_mechanisms:
            if progress_cb is not None:
                progress_cb(int(n_mechanisms), int(n_mechanisms), float(psi))
            return float(psi)

        parallel_workers_eff = 1
        if phase1_parallel_workers is not None:
            try:
                parallel_workers_eff = max(1, int(phase1_parallel_workers))
            except (TypeError, ValueError):
                parallel_workers_eff = 1
        chunk_size_eff = max(1, int(phase1_chunk_size))
        remaining_mechanisms = mechanisms[done_start:]
        chunks = _build_phase1_chunks_adaptive(
            remaining_mechanisms,
            max_chunk_size=chunk_size_eff,
            workers=parallel_workers_eff,
        )
        worker_cache_spec = None
        if bool(use_induced_partition_cache) and (kernel_cache is not None):
            cache_path = str(getattr(kernel_cache, "path", "")).strip()
            if cache_path:
                cache_mem_total = max(10_000, int(getattr(kernel_cache, "memory_entries", 100_000)))
                cache_flush_batch = max(100, int(getattr(kernel_cache, "flush_batch", 5_000)))
                per_worker_mem_entries = max(
                    10_000,
                    int(cache_mem_total // max(1, parallel_workers_eff)),
                )
                worker_cache_spec = {
                    "enabled": True,
                    "path": cache_path,
                    "memory_entries": int(per_worker_mem_entries),
                    "flush_batch": int(cache_flush_batch),
                }
        use_parallel = (
            phase_parallel_runtime_enabled
            and
            parallel_workers_eff > 1
            and len(remaining_mechanisms) > 1
        )
        if use_parallel and (parallel_runtime is not None):
            try:
                return float(
                    parallel_runtime.run_chunks(
                        tpm=tpm,
                        chunks=chunks,
                        psi_start=float(psi),
                        done_start=int(done_start),
                        total_mechanisms=int(n_mechanisms),
                        progress_cb=progress_cb,
                        cache_spec=worker_cache_spec,
                        cut_mask_a=(None if cut_mask_a is None else int(cut_mask_a)),
                        use_induced_partition_cache=bool(use_induced_partition_cache),
                        kernel_cache_lookup_only=bool(kernel_cache_lookup_only),
                    )
                )
            except Exception as exc:
                phase_parallel_runtime_enabled = False
                log.warning(
                    "[%s] IIM reusable phase workers failed (%s); falling back to per-call execution.",
                    run_label,
                    exc,
                )

        if use_parallel:
            parallel_workers_eff = min(parallel_workers_eff, len(remaining_mechanisms))
            owner_shms = []
            owner_tmp_dir = None
            owner_tmp_files = []
            try:
                def _mk_memmap_spec(label, arr):
                    nonlocal owner_tmp_dir
                    if owner_tmp_dir is None:
                        owner_tmp_dir = tempfile.mkdtemp(prefix="iim_phase1_")
                    path = os.path.join(owner_tmp_dir, f"{label}.npy")
                    np.save(path, np.ascontiguousarray(arr), allow_pickle=False)
                    owner_tmp_files.append(path)
                    return {
                        "mode": "memmap",
                        "path": str(path),
                    }

                def _mk_shm_spec(arr):
                    arr_c = np.ascontiguousarray(arr)
                    shm = shared_memory.SharedMemory(create=True, size=arr_c.nbytes)
                    shm_arr = np.ndarray(arr_c.shape, dtype=arr_c.dtype, buffer=shm.buf)
                    shm_arr[...] = arr_c
                    owner_shms.append(shm)
                    return {
                        "mode": "shared_memory",
                        "name": str(shm.name),
                        "shape": list(arr_c.shape),
                        "dtype": str(arr_c.dtype),
                    }

                try:
                    if not bool(phase1_shared_memory):
                        raise RuntimeError("phase1_shared_memory_disabled")
                    spec_tpm = _mk_shm_spec(tpm)
                    spec_curr = _mk_shm_spec(curr_obs)
                    spec_states = _mk_shm_spec(states_full)
                except Exception as shm_exc:
                    for shm in owner_shms:
                        try:
                            shm.close()
                        except Exception:
                            pass
                        try:
                            shm.unlink()
                        except Exception:
                            pass
                    owner_shms = []
                    log.warning(
                        "[%s] IIM phase intra-task shared-memory unavailable (%s); using read-only memmap fallback.",
                        run_label,
                        shm_exc,
                    )
                    spec_tpm = _mk_memmap_spec("tpm", tpm)
                    spec_curr = _mk_memmap_spec("curr_obs", curr_obs)
                    spec_states = _mk_memmap_spec("states_full", states_full)

                future_to_idx = {}
                completed = {}
                next_idx = 0
                done_abs = int(done_start)
                total_abs = int(n_mechanisms)
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=int(parallel_workers_eff),
                    initializer=_iim_phase_worker_init_static,
                    initargs=(spec_curr, spec_states, int(base), tuple(purviews)),
                ) as ex:
                    max_in_flight = max(1, int(parallel_workers_eff) * 2)
                    next_submit_idx = 0
                    n_chunks = int(len(chunks))

                    while next_submit_idx < min(n_chunks, max_in_flight):
                        idx = int(next_submit_idx)
                        chunk = chunks[idx]
                        fut = ex.submit(
                            _iim_phase_worker_run_chunk_for_tpm,
                            spec_tpm,
                            chunk,
                            worker_cache_spec,
                            (None if cut_mask_a is None else int(cut_mask_a)),
                            bool(use_induced_partition_cache),
                            bool(kernel_cache_lookup_only),
                        )
                        future_to_idx[fut] = idx
                        next_submit_idx += 1

                    while future_to_idx:
                        done_set, _ = concurrent.futures.wait(
                            tuple(future_to_idx.keys()),
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for fut in done_set:
                            idx = int(future_to_idx.pop(fut))
                            chunk_psi, chunk_len = fut.result()
                            completed[idx] = (float(chunk_psi), int(chunk_len))
                        while (
                            next_submit_idx < n_chunks
                            and len(future_to_idx) < max_in_flight
                        ):
                            idx = int(next_submit_idx)
                            chunk = chunks[idx]
                            fut = ex.submit(
                                _iim_phase_worker_run_chunk_for_tpm,
                                spec_tpm,
                                chunk,
                                worker_cache_spec,
                                (None if cut_mask_a is None else int(cut_mask_a)),
                                bool(use_induced_partition_cache),
                                bool(kernel_cache_lookup_only),
                            )
                            future_to_idx[fut] = idx
                            next_submit_idx += 1
                        while next_idx in completed:
                            cpsi, clen = completed.pop(next_idx)
                            psi = float(math.fsum((float(psi), float(cpsi))))
                            done_abs += int(clen)
                            if progress_cb is not None:
                                progress_cb(int(done_abs), int(total_abs), float(psi))
                            next_idx += 1
                return float(psi)
            except Exception as exc:
                phase_parallel_runtime_enabled = False
                log.warning(
                    "[%s] IIM intra-task phase parallelization failed (%s); falling back to sequential mechanisms.",
                    run_label,
                    exc,
                )
            finally:
                for shm in owner_shms:
                    try:
                        shm.close()
                    except Exception:
                        pass
                    try:
                        shm.unlink()
                    except Exception:
                        pass
                for p in owner_tmp_files:
                    try:
                        os.remove(p)
                    except Exception:
                        pass
                if owner_tmp_dir and os.path.isdir(owner_tmp_dir):
                    try:
                        shutil.rmtree(owner_tmp_dir, ignore_errors=True)
                    except Exception:
                        pass

        done_abs = int(done_start)
        total_abs = int(n_mechanisms)
        for chunk in chunks:
            chunk_psi = _iim_phase1_chunk_contribution(
                chunk,
                purviews,
                int(base),
                tpm,
                curr_obs,
                states_full,
                static_cache=static_cache,
                obs_state_cache=obs_state_cache,
                kernel_cache=kernel_cache,
                cut_mask_a=cut_mask_a,
                use_induced_partition_cache=bool(use_induced_partition_cache),
                kernel_cache_lookup_only=bool(kernel_cache_lookup_only),
            )
            chunk_len = int(len(chunk))
            psi = float(math.fsum((float(psi), float(chunk_psi))))
            done_abs += chunk_len
            if progress_cb is not None:
                progress_cb(int(done_abs), int(total_abs), float(psi))
        return float(psi)

    def _build_cut_tpm(tpm, states_full, base, A, B):
        return _iim_build_cut_tpm(
            tpm,
            states_full,
            base,
            A,
            B,
            hardware_backend=backend,
        )

    def _all_system_cuts(n_nodes_sys, part_mode):
        nodes = tuple(range(n_nodes_sys))
        all_cuts = _enumerate_bipartitions(nodes)
        if part_mode == "balanced":
            out = []
            for A, B in all_cuts:
                if abs(len(A) - len(B)) <= 1:
                    out.append((A, B))
            return out
        return all_cuts

    def _cut_to_key(A, B):
        return f"{','.join(map(str, A))}|{','.join(map(str, B))}"

    def _cut_from_payload(payload):
        if (
            isinstance(payload, (list, tuple))
            and len(payload) == 2
            and isinstance(payload[0], (list, tuple))
            and isinstance(payload[1], (list, tuple))
        ):
            return (tuple(int(x) for x in payload[0]), tuple(int(x) for x in payload[1]))
        return None

    # ---------- input checks ----------
    if ts.ndim != 2:
        raise ValueError(f"ts should be 2D (n_regions × n_time), got shape {ts.shape}")
    n_regions, n_time = ts.shape
    if method not in {"causal", "random"}:
        raise ValueError("method must be 'causal' (or legacy alias 'random')")
    if partition_mode not in {"all", "balanced"}:
        raise ValueError("partition_mode must be 'all' or 'balanced'")
    if scale <= 0:
        raise ValueError("scale must be > 0")
    if bins < 2:
        raise ValueError("bins must be >= 2")
    if n_parts is not None and int(n_parts) < 1:
        raise ValueError("n_parts must be >= 1 or None for exhaustive search")
    if max_nodes is not None and int(max_nodes) < 2:
        raise ValueError("max_nodes must be >= 2 or None")
    if max_mechanism_size is not None and int(max_mechanism_size) < 1:
        raise ValueError("max_mechanism_size must be >= 1 or None")
    if max_purview_size is not None and int(max_purview_size) < 1:
        raise ValueError("max_purview_size must be >= 1 or None")
    if tpm_alpha <= 0:
        raise ValueError("tpm_alpha must be > 0")
    if max_state_space < 16:
        raise ValueError("max_state_space must be >= 16")
    if checkpoint_every_cuts < 1:
        raise ValueError("checkpoint_every_cuts must be >= 1")
    if progress_log_every_cuts < 1:
        raise ValueError("progress_log_every_cuts must be >= 1")
    if phase1_parallel_workers is not None and int(phase1_parallel_workers) < 1:
        raise ValueError("phase1_parallel_workers must be >= 1 or None")
    if int(phase1_chunk_size) < 1:
        raise ValueError("phase1_chunk_size must be >= 1")

    def _undefined_payload(reason, psi_full=np.nan, psi_mip=np.nan, extra=None):
        payload_extra = extra or {}
        if return_details:
            out = {
                "value": np.nan,
                "raw": np.nan,
                "canonical": np.nan,
                "clipped": np.nan,   # legacy alias
                "iim_plus": np.nan,  # legacy alias
                "scale": scale,
                "I_full": float(psi_full),          # legacy field name
                "min_partition_sum": float(psi_mip),# legacy field name
                "Psi_full": float(psi_full),
                "Psi_mip_preserved": float(psi_mip),
                "defined": False,
                "undefined_reason": reason,
            }
            out.update(payload_extra)
            return out
        return np.nan

    run_label = str(progress_label).strip() if progress_label is not None else ""
    if not run_label:
        run_label = "IIM"
    if not bool(use_induced_partition_cache):
        log.info(
            "[%s] IIM induced-partition cache opt-out is deprecated; keeping cache enabled.",
            run_label,
        )
    use_induced_partition_cache = True

    # not enough regions or timepoints
    if n_regions < 2 or n_time - lag_trs < 1:
        return _undefined_payload(
            "insufficient_shape",
            extra={"checkpoint_path": checkpoint_path, "checkpoint_resumed": False},
        )

    # set up random generator
    if isinstance(rng, np.random.RandomState):
        rand_state = rng
    else:
        rand_state = np.random.RandomState(rng)

    # legacy alias
    _ = method

    prep = prepare_iim_problem(
        ts,
        bins=int(bins),
        lag_trs=int(lag_trs),
        n_parts=n_parts,
        rng=rand_state,
        partition_mode=str(partition_mode),
        max_nodes=max_nodes,
        max_mechanism_size=max_mechanism_size,
        max_purview_size=max_purview_size,
        tpm_alpha=float(tpm_alpha),
        max_state_space=int(max_state_space),
        hardware_backend=backend,
    )
    if not bool(prep.get("defined", False)):
        return _undefined_payload(
            str(prep.get("undefined_reason", "iim_prepare_failed")),
            extra={
                "n_nodes_used": prep.get("n_nodes_used"),
                "bins_used": prep.get("bins_used"),
            },
        )

    selected = np.asarray(prep["selected_nodes"], dtype=int)
    ts_sel = np.asarray(prep["ts_selected"], dtype=float)
    disc = np.asarray(prep["disc"], dtype=np.int16)
    curr_obs = np.asarray(prep["curr_obs"], dtype=np.int16)
    tpm_full = np.asarray(prep["tpm_full"], dtype=float)
    states_full = np.asarray(prep["states_full"], dtype=np.int16)
    eff_bins = int(prep["bins_used"])
    n_sel = int(prep["n_nodes_used"])
    mech_size_eff = int(prep["max_mechanism_size_used"])
    purv_size_eff = int(prep["max_purview_size_used"])
    mechanisms_all = tuple(tuple(m) for m in prep["mechanisms_all"])
    purviews_all = tuple(tuple(z) for z in prep["purviews_all"])
    # Static structures reused for all Ψ evaluations in this run (phase-1 + all cuts).
    psi_static_cache = {}
    psi_obs_state_cache = {}

    cuts_eval = [tuple(x) for x in prep["cuts_eval"]]
    cut_key_to_cut = {_cut_to_key(A, B): (A, B) for A, B in cuts_eval}
    cuts_payload = [list(x) for x in prep["cuts_payload"]]

    class _ReusablePhaseParallelRuntime:
        def __init__(
            self,
            workers,
            base,
            purviews,
            curr_obs,
            states_full,
            use_shared_memory,
            run_label,
        ):
            self.workers = int(max(1, workers))
            self.base = int(base)
            self.purviews = tuple(tuple(z) for z in purviews)
            self.use_shared_memory = bool(use_shared_memory)
            self.run_label = str(run_label)
            self.static_shms = []
            self.static_files = []
            self.tmp_dir = tempfile.mkdtemp(prefix="iim_phase_runtime_")
            self.executor = None

            spec_curr = self._mk_static_spec("curr_obs", curr_obs)
            spec_states = self._mk_static_spec("states_full", states_full)
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.workers,
                initializer=_iim_phase_worker_init_static,
                initargs=(spec_curr, spec_states, self.base, self.purviews),
            )

        def _mk_memmap_spec(self, label, arr, remember_static):
            path = os.path.join(self.tmp_dir, f"{label}_{os.getpid()}_{time.time_ns()}.npy")
            np.save(path, np.ascontiguousarray(arr), allow_pickle=False)
            if remember_static:
                self.static_files.append(path)
            return {"mode": "memmap", "path": str(path)}, None, path

        def _mk_shm_spec(self, arr, remember_static):
            arr_c = np.ascontiguousarray(arr)
            shm = shared_memory.SharedMemory(create=True, size=int(arr_c.nbytes))
            shm_arr = np.ndarray(arr_c.shape, dtype=arr_c.dtype, buffer=shm.buf)
            shm_arr[...] = arr_c
            if remember_static:
                self.static_shms.append(shm)
            return {
                "mode": "shared_memory",
                "name": str(shm.name),
                "shape": list(arr_c.shape),
                "dtype": str(arr_c.dtype),
            }, shm, None

        def _mk_static_spec(self, label, arr):
            if self.use_shared_memory:
                try:
                    spec, _, _ = self._mk_shm_spec(arr, remember_static=True)
                    return spec
                except Exception as exc:
                    log.warning(
                        "[%s] IIM reusable workers: shared-memory unavailable for %s (%s); using memmap.",
                        self.run_label,
                        label,
                        exc,
                    )
            spec, _, _ = self._mk_memmap_spec(label, arr, remember_static=True)
            return spec

        def _mk_tpm_spec(self, tpm):
            if self.use_shared_memory:
                try:
                    return self._mk_shm_spec(tpm, remember_static=False)
                except Exception as exc:
                    log.warning(
                        "[%s] IIM reusable workers: shared-memory unavailable for cut TPM (%s); using memmap.",
                        self.run_label,
                        exc,
                    )
            return self._mk_memmap_spec("tpm_cut", tpm, remember_static=False)

        @staticmethod
        def _cleanup_tpm_spec(spec_shm, spec_path):
            if spec_shm is not None:
                try:
                    spec_shm.close()
                except Exception:
                    pass
                try:
                    spec_shm.unlink()
                except Exception:
                    pass
            if spec_path and os.path.exists(spec_path):
                try:
                    os.remove(spec_path)
                except Exception:
                    pass

        def run_chunks(
            self,
            tpm,
            chunks,
            psi_start,
            done_start,
            total_mechanisms,
            progress_cb,
            cache_spec=None,
            cut_mask_a=None,
            use_induced_partition_cache=False,
            kernel_cache_lookup_only=False,
        ):
            if self.executor is None:
                raise RuntimeError("Reusable phase runtime is not initialized.")
            if not chunks:
                if progress_cb is not None:
                    progress_cb(int(total_mechanisms), int(total_mechanisms), float(psi_start))
                return float(psi_start)

            spec_tpm, tpm_shm, tpm_path = self._mk_tpm_spec(tpm)
            psi = float(psi_start)
            done_abs = int(done_start)
            total_abs = int(total_mechanisms)
            future_to_idx = {}
            completed = {}
            next_idx = 0
            try:
                max_in_flight = max(1, int(self.workers) * 2)
                next_submit_idx = 0
                n_chunks = int(len(chunks))

                while next_submit_idx < min(n_chunks, max_in_flight):
                    idx = int(next_submit_idx)
                    chunk = chunks[idx]
                    fut = self.executor.submit(
                        _iim_phase_worker_run_chunk_for_tpm,
                        spec_tpm,
                        chunk,
                        cache_spec,
                        cut_mask_a,
                        bool(use_induced_partition_cache),
                        bool(kernel_cache_lookup_only),
                    )
                    future_to_idx[fut] = idx
                    next_submit_idx += 1

                while future_to_idx:
                    done_set, _ = concurrent.futures.wait(
                        tuple(future_to_idx.keys()),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for fut in done_set:
                        idx = int(future_to_idx.pop(fut))
                        chunk_psi, chunk_len = fut.result()
                        completed[idx] = (float(chunk_psi), int(chunk_len))
                    while (
                        next_submit_idx < n_chunks
                        and len(future_to_idx) < max_in_flight
                    ):
                        idx = int(next_submit_idx)
                        chunk = chunks[idx]
                        fut = self.executor.submit(
                            _iim_phase_worker_run_chunk_for_tpm,
                            spec_tpm,
                            chunk,
                            cache_spec,
                            cut_mask_a,
                            bool(use_induced_partition_cache),
                            bool(kernel_cache_lookup_only),
                        )
                        future_to_idx[fut] = idx
                        next_submit_idx += 1
                    while next_idx in completed:
                        cpsi, clen = completed.pop(next_idx)
                        psi = float(math.fsum((float(psi), float(cpsi))))
                        done_abs += int(clen)
                        if progress_cb is not None:
                            progress_cb(int(done_abs), int(total_abs), float(psi))
                        next_idx += 1
                return float(psi)
            finally:
                self._cleanup_tpm_spec(tpm_shm, tpm_path)

        def close(self):
            if self.executor is not None:
                try:
                    self.executor.shutdown(wait=True, cancel_futures=False)
                except Exception:
                    pass
                self.executor = None
            for shm in self.static_shms:
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass
            self.static_shms = []
            for path in self.static_files:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass
            self.static_files = []
            if self.tmp_dir and os.path.isdir(self.tmp_dir):
                try:
                    shutil.rmtree(self.tmp_dir, ignore_errors=True)
                except Exception:
                    pass
                self.tmp_dir = None

    phase_parallel_runtime = None
    try:
        phase1_parallel_workers_eff = 1
        if phase1_parallel_workers is not None:
            try:
                phase1_parallel_workers_eff = max(1, int(phase1_parallel_workers))
            except (TypeError, ValueError):
                phase1_parallel_workers_eff = 1
        if phase1_parallel_workers_eff > 1 and len(mechanisms_all) > 1:
            try:
                phase_parallel_runtime = _ReusablePhaseParallelRuntime(
                    workers=phase1_parallel_workers_eff,
                    base=eff_bins,
                    purviews=purviews_all,
                    curr_obs=curr_obs,
                    states_full=states_full,
                    use_shared_memory=bool(phase1_shared_memory),
                    run_label=run_label,
                )
            except Exception as exc:
                phase_parallel_runtime = None
                log.warning(
                    "[%s] IIM reusable phase-worker pool init failed (%s); using per-call parallelization.",
                    run_label,
                    exc,
                )
    except Exception:
        phase_parallel_runtime = None

    def _close_phase_parallel_runtime():
        nonlocal phase_parallel_runtime
        if phase_parallel_runtime is not None:
            phase_parallel_runtime.close()
            phase_parallel_runtime = None

    checkpoint_resumed = False
    checkpoint_reused_cuts = 0
    checkpoint_used_psi_full = False
    phase1_mechanisms_done = 0
    phase1_total_mechanisms = None
    phase1_eta_seconds = None
    phase1_psi_partial = np.nan
    phase1_resumed_partial = False
    iim_phase = "init"
    phase2_mode = "unknown"
    materialization_done_mechanisms = 0
    materialization_total_mechanisms = None
    materialization_est_scale = None
    materialization_key_checks = 0
    materialization_missing_groups = 0
    materialization_computed_groups = 0
    completed_cut_scores = {}
    psi_preserved_max = -np.inf
    mip_cut = None
    psi_full = np.nan

    signature = None
    if checkpoint_path:
        try:
            disc_view = memoryview(np.ascontiguousarray(disc))
            disc_hash = hashlib.sha1(disc_view).hexdigest()
        except Exception:
            disc_hash = None
        cuts_hash = hashlib.sha1(
            json.dumps(cuts_payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest()
        signature = {
            "n_regions_input": int(n_regions),
            "n_time_input": int(n_time),
            "n_nodes_used": int(n_sel),
            "bins_used": int(eff_bins),
            "lag_trs": int(lag_trs),
            "partition_mode": str(partition_mode),
            "n_parts_requested": (None if n_parts is None else int(n_parts)),
            "max_nodes_requested": (None if max_nodes is None else int(max_nodes)),
            "max_mechanism_size_used": int(mech_size_eff),
            "max_purview_size_used": int(purv_size_eff),
            "n_cuts_total": int(len(cuts_eval)),
            "cuts_hash": cuts_hash,
            "disc_hash": disc_hash,
        }

        checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

        if resume_from_checkpoint and os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, "r", encoding="utf-8") as f:
                    ckpt = json.load(f)
                if ckpt.get("signature") == signature:
                    checkpoint_resumed = True
                    psi_ck = ckpt.get("psi_full")
                    if psi_ck is not None:
                        try:
                            psi_ck = float(psi_ck)
                        except (TypeError, ValueError):
                            psi_ck = np.nan
                        if np.isfinite(psi_ck):
                            psi_full = psi_ck
                            checkpoint_used_psi_full = True

                    ck_completed = ckpt.get("completed_cuts", {})
                    if isinstance(ck_completed, dict):
                        for k, v in ck_completed.items():
                            if k not in cut_key_to_cut:
                                continue
                            try:
                                v = float(v)
                            except (TypeError, ValueError):
                                continue
                            if not np.isfinite(v):
                                continue
                            completed_cut_scores[k] = v
                        checkpoint_reused_cuts = int(len(completed_cut_scores))

                    ph1_done = ckpt.get("phase1_mechanisms_done")
                    ph1_total = ckpt.get("phase1_total_mechanisms")
                    ph1_psi = ckpt.get("phase1_psi_partial")
                    try:
                        ph1_done = int(ph1_done) if ph1_done is not None else 0
                    except (TypeError, ValueError):
                        ph1_done = 0
                    try:
                        ph1_total = int(ph1_total) if ph1_total is not None else None
                    except (TypeError, ValueError):
                        ph1_total = None
                    try:
                        ph1_psi = float(ph1_psi) if ph1_psi is not None else np.nan
                    except (TypeError, ValueError):
                        ph1_psi = np.nan
                    if ph1_total is not None and ph1_total > 0:
                        phase1_mechanisms_done = max(0, min(ph1_done, ph1_total))
                        phase1_total_mechanisms = ph1_total
                    if np.isfinite(ph1_psi):
                        phase1_psi_partial = float(ph1_psi)
                    if checkpoint_used_psi_full and np.isfinite(psi_full):
                        phase1_psi_partial = float(psi_full)

                    best_block = ckpt.get("best", {})
                    if isinstance(best_block, dict):
                        best_v = best_block.get("psi_preserved_max")
                        try:
                            best_v = float(best_v)
                        except (TypeError, ValueError):
                            best_v = -np.inf
                        if np.isfinite(best_v):
                            psi_preserved_max = float(best_v)
                            cut_payload = _cut_from_payload(best_block.get("mip_cut"))
                            if cut_payload is not None:
                                mip_cut = cut_payload

                    if not np.isfinite(psi_preserved_max):
                        psi_preserved_max = -np.inf
                    if (mip_cut is None) or (not np.isfinite(psi_preserved_max)):
                        for ck, cv in completed_cut_scores.items():
                            if cv > psi_preserved_max:
                                psi_preserved_max = cv
                                mip_cut = cut_key_to_cut.get(ck)

                    log.info(
                        "[%s] IIM checkpoint resume: reused_cuts=%d/%d, reused_psi_full=%s, file=%s",
                        run_label,
                        checkpoint_reused_cuts,
                        len(cuts_eval),
                        checkpoint_used_psi_full,
                        checkpoint_path,
                    )
                    if (
                        (not checkpoint_used_psi_full)
                        and (phase1_total_mechanisms is not None)
                        and (phase1_mechanisms_done > 0)
                    ):
                        if np.isfinite(phase1_psi_partial):
                            phase1_resumed_partial = True
                            log.info(
                                "[%s] IIM checkpoint partial phase-1 resume prepared: %d/%d mechanisms, psi_partial=%.6f.",
                                run_label,
                                int(phase1_mechanisms_done),
                                int(phase1_total_mechanisms),
                                float(phase1_psi_partial),
                            )
                        else:
                            log.info(
                                "[%s] IIM checkpoint has partial phase-1 progress %d/%d but no psi_partial; restarting phase-1 from scratch.",
                                run_label,
                                int(phase1_mechanisms_done),
                                int(phase1_total_mechanisms),
                            )
                            phase1_mechanisms_done = 0
                            phase1_total_mechanisms = None
                else:
                    log.warning(
                        "[%s] IIM checkpoint signature mismatch, starting fresh: %s",
                        run_label,
                        checkpoint_path,
                    )
            except Exception as exc:
                log.warning(
                    "[%s] IIM checkpoint load failed (%s), starting fresh: %s",
                    run_label,
                    exc,
                    checkpoint_path,
                )

    induced_kernel_cache = None
    induced_kernel_cache_path_eff = None
    if kernel_cache_path is not None:
        induced_kernel_cache_path_eff = str(kernel_cache_path)
    elif checkpoint_path:
        induced_kernel_cache_path_eff = f"{checkpoint_path}.kernel.sqlite3"
    else:
        induced_kernel_cache_path_eff = os.path.join(
            tempfile.gettempdir(),
            f"iim_kernel_cache_{os.getpid()}_{time.time_ns()}.sqlite3",
        )
    try:
        induced_kernel_cache = _IIMDiskKernelCache(
            induced_kernel_cache_path_eff,
            signature=signature,
            memory_entries=int(kernel_cache_memory_entries),
            flush_batch=int(kernel_cache_flush_batch),
        )
        log.info(
            "[%s] IIM induced-partition cache: enabled path=%s mem_entries=%d flush_batch=%d",
            run_label,
            induced_kernel_cache_path_eff,
            int(kernel_cache_memory_entries),
            int(kernel_cache_flush_batch),
        )
    except Exception as exc:
        induced_kernel_cache = None
        log.warning(
            "[%s] IIM induced-partition cache init failed (%s); continuing without disk-backed cache.",
            run_label,
            exc,
        )

    def _close_induced_cache():
        nonlocal induced_kernel_cache
        if induced_kernel_cache is not None:
            try:
                induced_kernel_cache.close()
            except Exception:
                pass
            induced_kernel_cache = None

    def _write_checkpoint(status, undefined_reason=None):
        if not checkpoint_path:
            return
        payload = {
            "version": 3,
            "status": status,
            "updated_unix": float(time.time()),
            "signature": signature,
            "iim_phase": str(iim_phase),
            "phase2_mode": str(phase2_mode),
            "psi_full": (None if not np.isfinite(psi_full) else float(psi_full)),
            "completed_cuts": {k: float(v) for k, v in completed_cut_scores.items()},
            "best": {
                "psi_preserved_max": (
                    None if not np.isfinite(psi_preserved_max) else float(psi_preserved_max)
                ),
                "mip_cut": (
                    None
                    if mip_cut is None
                    else [list(mip_cut[0]), list(mip_cut[1])]
                ),
            },
            "n_cuts_total": int(len(cuts_eval)),
            "undefined_reason": undefined_reason,
            "phase1_mechanisms_done": int(phase1_mechanisms_done),
            "phase1_total_mechanisms": (
                None
                if phase1_total_mechanisms is None
                else int(phase1_total_mechanisms)
            ),
            "phase1_eta_seconds": (
                None if phase1_eta_seconds is None else float(phase1_eta_seconds)
            ),
            "phase1_psi_partial": (
                None if not np.isfinite(phase1_psi_partial) else float(phase1_psi_partial)
            ),
            "phase1_resumed_partial": bool(phase1_resumed_partial),
            "materialization_done_mechanisms": int(materialization_done_mechanisms),
            "materialization_total_mechanisms": (
                None
                if materialization_total_mechanisms is None
                else int(materialization_total_mechanisms)
            ),
            "materialization_est_scale": (
                None if materialization_est_scale is None else int(materialization_est_scale)
            ),
            "materialization_key_checks": int(materialization_key_checks),
            "materialization_missing_groups": int(materialization_missing_groups),
            "materialization_computed_groups": int(materialization_computed_groups),
        }
        tmp_path = f"{checkpoint_path}.tmp.{os.getpid()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=True, sort_keys=True)
            os.replace(tmp_path, checkpoint_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    if not np.isfinite(psi_full):
        iim_phase = "phase1_psi"
        resume_done_mechanisms = 0
        resume_psi_for_compute = np.nan
        if (
            phase1_resumed_partial
            and (phase1_total_mechanisms is not None)
            and np.isfinite(phase1_psi_partial)
            and (phase1_mechanisms_done > 0)
        ):
            resume_done_mechanisms = int(phase1_mechanisms_done)
            resume_psi_for_compute = float(phase1_psi_partial)
        else:
            phase1_mechanisms_done = 0
            phase1_total_mechanisms = None
            phase1_eta_seconds = None
            phase1_psi_partial = np.nan

        log.info(
            "[%s] IIM phase 1/2: computing Psi_full (nodes=%d, bins=%d, cuts=%d, resume_done=%d)",
            run_label,
            int(n_sel),
            int(eff_bins),
            int(len(cuts_eval)),
            int(resume_done_mechanisms),
        )
        phase1_log_step_pct = 5.0
        if phase1_total_mechanisms is not None and int(phase1_total_mechanisms) > 0:
            phase1_next_log_pct = (
                100.0 * float(phase1_mechanisms_done) / float(int(phase1_total_mechanisms))
            )
        else:
            phase1_next_log_pct = 0.0
        phase1_last_logged_done = -1
        phase1_started_at = time.time()
        phase1_start_done = int(resume_done_mechanisms)

        def _phase1_progress_cb(done, total, psi_partial):
            nonlocal phase1_mechanisms_done
            nonlocal phase1_total_mechanisms
            nonlocal phase1_eta_seconds
            nonlocal phase1_psi_partial
            nonlocal phase1_next_log_pct
            nonlocal phase1_last_logged_done
            phase1_mechanisms_done = int(done)
            phase1_total_mechanisms = int(total) if total is not None else None
            phase1_psi_partial = float(psi_partial) if np.isfinite(psi_partial) else np.nan
            if total is None or int(total) <= 0:
                return
            total_i = int(total)
            done_i = int(done)
            pct = 100.0 * float(done_i) / float(total_i)
            should_log = (done_i == total_i) or (pct >= phase1_next_log_pct)
            if should_log and done_i != phase1_last_logged_done:
                elapsed = max(float(time.time() - phase1_started_at), 1e-9)
                done_delta = max(0, int(done_i - phase1_start_done))
                rate = float(done_delta) / elapsed
                if rate > 0:
                    eta = max(float(total_i - done_i) / rate, 0.0)
                    phase1_eta_seconds = float(eta)
                else:
                    eta = np.nan
                    phase1_eta_seconds = None
                eta_txt = "na" if not np.isfinite(eta) else f"{eta:.1f}s"
                log.info(
                    "[%s] IIM phase 1/2 progress: mechanisms=%d/%d (%.1f%%), eta=%s, psi_partial=%.6f",
                    run_label,
                    done_i,
                    total_i,
                    pct,
                    eta_txt,
                    float(psi_partial),
                )
                if checkpoint_path:
                    _write_checkpoint("running")
                phase1_last_logged_done = done_i
                while phase1_next_log_pct <= pct:
                    phase1_next_log_pct += phase1_log_step_pct

        psi_full = _compute_psi(
            tpm_full,
            curr_obs,
            states_full,
            eff_bins,
            mech_size_eff,
            purv_size_eff,
            resume_done_mechanisms=resume_done_mechanisms,
            resume_psi_partial=resume_psi_for_compute,
            progress_cb=_phase1_progress_cb,
            phase1_parallel_workers=phase1_parallel_workers,
            phase1_chunk_size=phase1_chunk_size,
            phase1_shared_memory=phase1_shared_memory,
            mechanisms=mechanisms_all,
            purviews=purviews_all,
            static_cache=psi_static_cache,
            obs_state_cache=psi_obs_state_cache,
            parallel_runtime=phase_parallel_runtime,
            kernel_cache=induced_kernel_cache,
            cut_mask_a=None,
            use_induced_partition_cache=bool(induced_kernel_cache is not None),
        )
        phase1_psi_partial = float(psi_full) if np.isfinite(psi_full) else np.nan
        _write_checkpoint("running")
    else:
        iim_phase = "phase1_done"
        log.info(
            "[%s] IIM phase 1/2: Psi_full loaded from checkpoint (nodes=%d, bins=%d, cuts=%d)",
            run_label,
            int(n_sel),
            int(eff_bins),
            int(len(cuts_eval)),
        )

    if not np.isfinite(psi_full) or psi_full <= 0:
        iim_phase = "undefined"
        _write_checkpoint("undefined", undefined_reason="nonpositive_psi_full")
        _close_phase_parallel_runtime()
        _close_induced_cache()
        return _undefined_payload(
            "nonpositive_psi_full",
            psi_full=float(psi_full) if np.isfinite(psi_full) else np.nan,
            extra={
                "n_nodes_used": int(n_sel),
                "bins_used": int(eff_bins),
                "checkpoint_path": checkpoint_path,
                "checkpoint_resumed": bool(checkpoint_resumed),
                "checkpoint_reused_cuts": int(checkpoint_reused_cuts),
                "checkpoint_used_psi_full": bool(checkpoint_used_psi_full),
                "phase1_resumed_partial": bool(phase1_resumed_partial),
                "phase1_psi_partial": (
                    float(phase1_psi_partial) if np.isfinite(phase1_psi_partial) else np.nan
                ),
            },
        )

    def _materialize_induced_kernel_cases():
        """
        Precompute unique kernel values keyed by induced unlabeled partition (pi),
        then allow cut-stage aggregation to run in lookup-only mode.

        Returns
        -------
        bool
            True when materialization ran successfully and lookup-only aggregation
            should be used; False to keep legacy per-cut recomputation behavior.
        """
        if induced_kernel_cache is None:
            return False
        if len(cuts_eval) <= 1:
            return False

        n_nodes_sys = int(states_full.shape[1])
        all_nodes_sys = tuple(range(n_nodes_sys))

        def _subset_mask_fast(subset):
            m = 0
            for nn in subset:
                m |= (1 << int(nn))
            return int(m)

        def _induced_pi_from_masks(m_mask, z_mask, cut_mask):
            u_mask = int(m_mask) | int(z_mask)
            part_a = int(u_mask & int(cut_mask))
            part_b = int(u_mask ^ part_a)
            if part_a == 0 or part_b == 0:
                return 0
            return int(part_a if part_a < part_b else part_b)

        def _mask_to_cut(mask):
            A = tuple(nn for nn in all_nodes_sys if ((int(mask) >> int(nn)) & 1))
            if len(A) == 0 or len(A) == n_nodes_sys:
                return None
            B = tuple(nn for nn in all_nodes_sys if not ((int(mask) >> int(nn)) & 1))
            return (A, B)

        cut_masks_eval = []
        for A, _B in cuts_eval:
            cm = 0
            for nn in A:
                cm |= (1 << int(nn))
            cut_masks_eval.append(int(cm))

        mechanism_entries = []
        for M in mechanisms_all:
            M = tuple(sorted(M))
            if len(M) < 2:
                continue
            obs_keys = _subset_key_matrix(curr_obs, M, eff_bins)
            if obs_keys.size == 0:
                continue
            uk = np.unique(obs_keys).astype(np.int64, copy=False)
            if uk.size == 0:
                continue
            mechanism_entries.append((M, _subset_mask_fast(M), uk))

        purview_entries = []
        for Z in purviews_all:
            Z = tuple(sorted(Z))
            if len(Z) < 2:
                continue
            purview_entries.append((Z, _subset_mask_fast(Z)))

        if not mechanism_entries or not purview_entries:
            return False

        nonlocal iim_phase
        nonlocal phase2_mode
        nonlocal materialization_done_mechanisms
        nonlocal materialization_total_mechanisms
        nonlocal materialization_est_scale
        nonlocal materialization_key_checks
        nonlocal materialization_missing_groups
        nonlocal materialization_computed_groups

        est_scale = int(len(mechanism_entries)) * int(len(purview_entries)) * int(len(cuts_eval))
        iim_phase = "phase2_materialize"
        phase2_mode = "lookup_only"
        materialization_done_mechanisms = 0
        materialization_total_mechanisms = int(len(mechanism_entries))
        materialization_est_scale = int(est_scale)
        materialization_key_checks = 0
        materialization_missing_groups = 0
        materialization_computed_groups = 0

        log.info(
            "[%s] IIM phase 2 prep: materializing induced kernel cases (mechanisms=%d, purviews=%d, cuts=%d, est_scale=%d).",
            run_label,
            int(len(mechanism_entries)),
            int(len(purview_entries)),
            int(len(cuts_eval)),
            int(est_scale),
        )

        t0 = time.time()
        tpm_by_pi = OrderedDict()
        tpm_by_pi[0] = tpm_full
        max_cached_pi_tpms = 4

        total_directional_keys = 0
        missing_groups = 0
        computed_groups = 0

        for m_idx, (M, m_mask, uk) in enumerate(mechanism_entries, start=1):
            materialization_done_mechanisms = int(m_idx)
            for Z, z_mask in purview_entries:
                pis = set()
                for cut_mask in cut_masks_eval:
                    pis.add(_induced_pi_from_masks(m_mask, z_mask, cut_mask))
                if not pis:
                    continue

                for pi in pis:
                    pi = int(pi)
                    if pi == 0:
                        tpm_ref = tpm_full
                    else:
                        if pi in tpm_by_pi:
                            tpm_ref = tpm_by_pi[pi]
                            tpm_by_pi.move_to_end(pi, last=True)
                        else:
                            cut_pair = _mask_to_cut(pi)
                            if cut_pair is None:
                                tpm_ref = tpm_full
                            else:
                                tpm_ref = _build_cut_tpm(
                                    tpm_full,
                                    states_full,
                                    eff_bins,
                                    cut_pair[0],
                                    cut_pair[1],
                                )
                            tpm_by_pi[pi] = tpm_ref
                            while (len(tpm_by_pi) - (1 if 0 in tpm_by_pi else 0)) > int(max_cached_pi_tpms):
                                old_nonzero = None
                                for k_tmp in tpm_by_pi.keys():
                                    if int(k_tmp) != 0:
                                        old_nonzero = int(k_tmp)
                                        break
                                if old_nonzero is None:
                                    break
                                tpm_by_pi.pop(old_nonzero, None)

                    static_cache = {}
                    cut_mask_for_key = int(pi)

                    for m_key in uk:
                        mk = int(m_key)
                        total_directional_keys += 2
                        materialization_key_checks = int(total_directional_keys)
                        key_e = (0, int(m_mask), int(mk), int(z_mask), int(pi))
                        key_c = (1, int(m_mask), int(mk), int(z_mask), int(pi))
                        v_e = induced_kernel_cache.get(key_e)
                        v_c = induced_kernel_cache.get(key_c)
                        if (v_e is not None) and (v_c is not None):
                            continue

                        missing_groups += 1
                        materialization_missing_groups = int(missing_groups)
                        obs_row = np.zeros((1, n_nodes_sys), dtype=np.int16)
                        m_vals = _decode_key(int(mk), len(M), eff_bins)
                        for p, nn in enumerate(M):
                            obs_row[0, int(nn)] = int(m_vals[p])

                        _iim_phase1_chunk_contribution(
                            (M,),
                            (Z,),
                            int(eff_bins),
                            tpm_ref,
                            obs_row,
                            states_full,
                            static_cache=static_cache,
                            obs_state_cache=None,
                            kernel_cache=induced_kernel_cache,
                            cut_mask_a=int(cut_mask_for_key),
                            use_induced_partition_cache=True,
                            kernel_cache_lookup_only=False,
                        )
                        computed_groups += 1
                        materialization_computed_groups = int(computed_groups)

            if (
                m_idx == int(len(mechanism_entries))
                or (m_idx % 10 == 0)
            ):
                elapsed = max(float(time.time() - t0), 1e-9)
                rate = float(computed_groups) / elapsed
                log.info(
                    "[%s] IIM materialization progress: mechanisms=%d/%d, computed_groups=%d, missing_groups=%d, key_checks=%d, rate=%.2f/s",
                    run_label,
                    int(m_idx),
                    int(len(mechanism_entries)),
                    int(computed_groups),
                    int(missing_groups),
                    int(total_directional_keys),
                    float(rate),
                )
                if checkpoint_path:
                    _write_checkpoint("running")

        try:
            induced_kernel_cache.flush()
        except Exception:
            pass

        elapsed = max(float(time.time() - t0), 1e-9)
        materialization_done_mechanisms = int(len(mechanism_entries))
        materialization_key_checks = int(total_directional_keys)
        materialization_missing_groups = int(missing_groups)
        materialization_computed_groups = int(computed_groups)
        log.info(
            "[%s] IIM materialization complete: computed_groups=%d, missing_groups=%d, key_checks=%d, elapsed=%.1fs",
            run_label,
            int(computed_groups),
            int(missing_groups),
            int(total_directional_keys),
            float(elapsed),
        )
        return True

    lookup_only_cut_aggregation = False
    try:
        lookup_only_cut_aggregation = bool(_materialize_induced_kernel_cases())
    except Exception as exc:
        lookup_only_cut_aggregation = False
        phase2_mode = "recompute"
        iim_phase = "phase2_cuts_recompute"
        log.warning(
            "[%s] IIM induced-kernel materialization failed (%s); using legacy per-cut recomputation.",
            run_label,
            exc,
        )
    if lookup_only_cut_aggregation:
        phase2_mode = "lookup_only"
        iim_phase = "phase2_cuts_lookup"
    else:
        phase2_mode = "recompute"
        if iim_phase != "phase2_cuts_recompute":
            iim_phase = "phase2_cuts_recompute"

    log.info(
        "[%s] IIM phase 2/2: cut search start (%d total, %d already done, lookup_only=%s)",
        run_label,
        int(len(cuts_eval)),
        int(len(completed_cut_scores)),
        bool(lookup_only_cut_aggregation),
    )
    cuts_done = int(len(completed_cut_scores))
    last_logged_done = -1
    for A, B in cuts_eval:
        cut_key = _cut_to_key(A, B)
        if cut_key in completed_cut_scores:
            psi_cut = float(completed_cut_scores[cut_key])
        else:
            cut_mask_a = 0
            for nn in A:
                cut_mask_a |= (1 << int(nn))
            if bool(lookup_only_cut_aggregation) and (induced_kernel_cache is not None):
                try:
                    psi_cut = _compute_psi(
                        tpm_full,
                        curr_obs,
                        states_full,
                        eff_bins,
                        mech_size_eff,
                        purv_size_eff,
                        phase1_parallel_workers=phase1_parallel_workers,
                        phase1_chunk_size=phase1_chunk_size,
                        phase1_shared_memory=phase1_shared_memory,
                        mechanisms=mechanisms_all,
                        purviews=purviews_all,
                        static_cache=psi_static_cache,
                        obs_state_cache=psi_obs_state_cache,
                        parallel_runtime=phase_parallel_runtime,
                        kernel_cache=induced_kernel_cache,
                        cut_mask_a=int(cut_mask_a),
                        use_induced_partition_cache=True,
                        kernel_cache_lookup_only=True,
                    )
                except _IIMKernelCacheMissError:
                    lookup_only_cut_aggregation = False
                    phase2_mode = "recompute"
                    iim_phase = "phase2_cuts_recompute"
                    log.warning(
                        "[%s] IIM lookup-only aggregation cache miss; falling back to per-cut recomputation.",
                        run_label,
                    )
                    tpm_cut = _build_cut_tpm(tpm_full, states_full, eff_bins, A, B)
                    psi_cut = _compute_psi(
                        tpm_cut,
                        curr_obs,
                        states_full,
                        eff_bins,
                        mech_size_eff,
                        purv_size_eff,
                        phase1_parallel_workers=phase1_parallel_workers,
                        phase1_chunk_size=phase1_chunk_size,
                        phase1_shared_memory=phase1_shared_memory,
                        mechanisms=mechanisms_all,
                        purviews=purviews_all,
                        static_cache=psi_static_cache,
                        obs_state_cache=psi_obs_state_cache,
                        parallel_runtime=phase_parallel_runtime,
                        kernel_cache=induced_kernel_cache,
                        cut_mask_a=int(cut_mask_a),
                        use_induced_partition_cache=bool(induced_kernel_cache is not None),
                        kernel_cache_lookup_only=False,
                    )
            else:
                tpm_cut = _build_cut_tpm(tpm_full, states_full, eff_bins, A, B)
                psi_cut = _compute_psi(
                    tpm_cut,
                    curr_obs,
                    states_full,
                    eff_bins,
                    mech_size_eff,
                    purv_size_eff,
                    phase1_parallel_workers=phase1_parallel_workers,
                    phase1_chunk_size=phase1_chunk_size,
                    phase1_shared_memory=phase1_shared_memory,
                    mechanisms=mechanisms_all,
                    purviews=purviews_all,
                    static_cache=psi_static_cache,
                    obs_state_cache=psi_obs_state_cache,
                    parallel_runtime=phase_parallel_runtime,
                    kernel_cache=induced_kernel_cache,
                    cut_mask_a=int(cut_mask_a),
                    use_induced_partition_cache=bool(induced_kernel_cache is not None),
                    kernel_cache_lookup_only=False,
                )
            completed_cut_scores[cut_key] = float(psi_cut)
            cuts_done += 1
            if (cuts_done % int(checkpoint_every_cuts) == 0) or (cuts_done == len(cuts_eval)):
                _write_checkpoint("running")

        if psi_cut > psi_preserved_max:
            psi_preserved_max = psi_cut
            mip_cut = (A, B)

        if (
            cuts_done != last_logged_done
            and ((cuts_done % int(progress_log_every_cuts) == 0) or (cuts_done == len(cuts_eval)))
        ):
            pct = 100.0 * float(cuts_done) / float(max(1, len(cuts_eval)))
            raw_so_far = float((psi_full - psi_preserved_max) / (psi_full + 1e-12))
            log.info(
                "[%s] IIM cut progress: %d/%d (%.1f%%), best_raw_so_far=%.6f",
                run_label,
                int(cuts_done),
                int(len(cuts_eval)),
                pct,
                raw_so_far,
            )
            last_logged_done = cuts_done

    if not np.isfinite(psi_preserved_max):
        iim_phase = "undefined"
        _write_checkpoint("undefined", undefined_reason="mip_not_found")
        _close_phase_parallel_runtime()
        _close_induced_cache()
        return _undefined_payload(
            "mip_not_found",
            psi_full=float(psi_full),
            psi_mip=np.nan,
            extra={
                "n_nodes_used": int(n_sel),
                "bins_used": int(eff_bins),
                "checkpoint_path": checkpoint_path,
                "checkpoint_resumed": bool(checkpoint_resumed),
                "checkpoint_reused_cuts": int(checkpoint_reused_cuts),
                "checkpoint_used_psi_full": bool(checkpoint_used_psi_full),
                "phase1_resumed_partial": bool(phase1_resumed_partial),
                "phase1_psi_partial": (
                    float(phase1_psi_partial) if np.isfinite(phase1_psi_partial) else np.nan
                ),
            },
        )

    raw = float((psi_full - psi_preserved_max) / (psi_full + 1e-12))
    canonical = float(np.clip(raw, 0.0, 1.0))
    # Legacy positive-only decomposition term retained for diagnostics.
    iim_plus = float(np.clip(raw, 0.0, 1.0))

    selected = canonical if clamp else raw
    value = float(selected * scale)

    if return_details:
        iim_phase = "complete"
        _write_checkpoint("complete")
        induced_cache_enabled = bool(induced_kernel_cache is not None)
        induced_cache_stats = (
            None if induced_kernel_cache is None else induced_kernel_cache.stats()
        )
        _close_phase_parallel_runtime()
        _close_induced_cache()
        return {
            "value": value,
            "raw": raw,
            "canonical": canonical,
            "clipped": canonical,  # legacy alias
            "iim_plus": iim_plus,  # legacy alias
            "scale": scale,
            "I_full": float(psi_full),  # legacy field name
            "min_partition_sum": float(psi_preserved_max),  # legacy field name
            "Psi_full": float(psi_full),
            "Psi_mip_preserved": float(psi_preserved_max),
            "n_nodes_used": int(n_sel),
            "bins_used": int(eff_bins),
            "max_nodes_requested": (None if max_nodes is None else int(max_nodes)),
            "max_mechanism_size_used": int(mech_size_eff),
            "max_purview_size_used": int(purv_size_eff),
            "n_parts_requested": (None if n_parts is None else int(n_parts)),
            "n_cuts_evaluated": int(len(cuts_eval)),
            "mip_cut": mip_cut,
            "checkpoint_path": checkpoint_path,
            "checkpoint_resumed": bool(checkpoint_resumed),
            "checkpoint_reused_cuts": int(checkpoint_reused_cuts),
            "checkpoint_used_psi_full": bool(checkpoint_used_psi_full),
            "phase1_resumed_partial": bool(phase1_resumed_partial),
            "phase1_psi_partial": (
                float(phase1_psi_partial) if np.isfinite(phase1_psi_partial) else np.nan
            ),
            "phase1_mechanisms_done": int(phase1_mechanisms_done),
            "phase1_total_mechanisms": (
                None if phase1_total_mechanisms is None else int(phase1_total_mechanisms)
            ),
            "phase1_eta_seconds": (
                None if phase1_eta_seconds is None else float(phase1_eta_seconds)
            ),
            "phase1_parallel_workers": (
                None if phase1_parallel_workers is None else int(phase1_parallel_workers)
            ),
            "phase1_chunk_size": int(phase1_chunk_size),
            "phase1_shared_memory": bool(phase1_shared_memory),
            "induced_partition_cache_enabled": bool(induced_cache_enabled),
            "induced_partition_cache_path": induced_kernel_cache_path_eff,
            "induced_partition_cache_stats": induced_cache_stats,
            "defined": True,
            "undefined_reason": None,
        }
    iim_phase = "complete"
    _write_checkpoint("complete")
    _close_phase_parallel_runtime()
    _close_induced_cache()
    return value


def compute_CI(
    ram: float,
    pdi: float,
    nas: float,
    iim: float,
    srpi: float,
    references: dict = None,
    weights: dict = None,
    defined: dict = None,
    eps: float = 1e-12,
    return_details: bool = False,
):
    """
    Human-normalized weighted geometric Consciousness Index (CI).

    CI = (RAM*^alpha) (PDI+*^beta) (NAS*^gamma) (IIM_can*^delta) (SRPI*^rho)

    where each * component is normalized by a human reference mean.
    If any weighted component is undefined, CI is set to 0 (hard criterion).
    """
    comp_keys = ("RAM", "PDI", "NAS", "IIM", "SRPI")
    comp_vals = {
        "RAM": float(ram),
        "PDI": float(pdi),
        "NAS": float(nas),
        "IIM": float(iim),
        "SRPI": float(srpi),
    }

    if references is None:
        references = {k: 1.0 for k in comp_keys}
    else:
        references = {k: float(references.get(k, 1.0)) for k in comp_keys}

    if weights is None:
        weights = {k: 1.0 / len(comp_keys) for k in comp_keys}
    else:
        weights = {k: float(weights.get(k, 0.0)) for k in comp_keys}
        wsum = sum(weights.values())
        if wsum <= 0:
            raise ValueError("weights must sum to a positive value")
        weights = {k: v / wsum for k, v in weights.items()}

    if defined is None:
        defined = {k: True for k in comp_keys}
    else:
        defined = {k: bool(defined.get(k, True)) for k in comp_keys}
    # Non-finite component values are treated as undefined.
    for k in comp_keys:
        if not np.isfinite(comp_vals[k]):
            defined[k] = False

    norm = {}
    undefined_weighted = []
    for k in comp_keys:
        if not defined[k]:
            norm[k] = np.nan
            if weights[k] > 0:
                undefined_weighted.append(k)
            continue
        ref = references[k]
        if ref <= 0:
            raise ValueError(f"reference for {k} must be > 0")
        # Metrics are defined as non-negative components in CI.
        norm[k] = max(comp_vals[k] / ref, 0.0)

    # Hard criterion: any undefined weighted component forces CI to 0.
    if undefined_weighted:
        ci_val = 0.0
    # Weighted geometric mean. Any zero component with non-zero weight zeros CI.
    elif any((norm[k] <= 0.0 and weights[k] > 0.0) for k in comp_keys):
        ci_val = 0.0
    else:
        log_ci = 0.0
        for k in comp_keys:
            log_ci += weights[k] * np.log(norm[k] + eps)
        ci_val = float(np.exp(log_ci))

    if return_details:
        return {
            "value": ci_val,
            "normalized_components": norm,
            "weights": weights,
            "references": references,
            "defined_components": defined,
            "undefined_weighted_components": undefined_weighted,
        }
    return ci_val


def _srpi_undefined_result(
    reason,
    return_details,
    counts,
    modality,
    windows_sec,
    eps,
):
    if not return_details:
        return float("nan")
    return {
        "value": float("nan"),
        "undefined_reason": str(reason),
        "components": {
            "reactivity_bias": float("nan"),
            "representational_separability": float("nan"),
            "self_pattern_stability": float("nan"),
            "internal_state_coupling": float("nan"),
        },
        "components_raw": {
            "reactivity_bias": float("nan"),
            "representational_separability": float("nan"),
            "self_pattern_stability": float("nan"),
            "internal_state_coupling": float("nan"),
        },
        "signed_reactivity_bias": float("nan"),
        "reliability": float("nan"),
        "weights": {
            "reactivity_bias": float("nan"),
            "representational_separability": float("nan"),
            "self_pattern_stability": float("nan"),
            "internal_state_coupling": float("nan"),
        },
        "counts": dict(counts),
        "modality": str(modality),
        "windows_sec": dict(windows_sec),
        "eps": float(eps),
    }


def _event_locked_state_deltas(ts_z, event_idx, lag_samples, pre_samples, post_samples):
    shifted_idx = np.asarray(event_idx, dtype=np.int64).reshape(-1) + int(lag_samples)
    pre_vecs, post_vecs, used_shifted = _window_mean_vectors(
        ts_z,
        shifted_idx,
        pre_samples=pre_samples,
        post_samples=post_samples,
    )
    if used_shifted.size == 0:
        n_regions = int(ts_z.shape[0])
        empty = np.empty((0, n_regions), dtype=float)
        return empty, empty, np.empty(0, dtype=np.int64)
    delta_vecs = post_vecs - pre_vecs
    used_original = used_shifted - int(lag_samples)
    return pre_vecs, delta_vecs, used_original


def _regularized_mahalanobis_distance_sq(x, y, ridge, hardware_backend=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2:
        return np.nan
    if x.shape[1] != y.shape[1]:
        return np.nan
    if x.shape[0] < 2 or y.shape[0] < 2:
        return np.nan

    mu_x = x.mean(axis=0)
    mu_y = y.mean(axis=0)
    dx = x - mu_x
    dy = y - mu_y

    cov_x = (dx.T @ dx) / float(max(1, x.shape[0] - 1))
    cov_y = (dy.T @ dy) / float(max(1, y.shape[0] - 1))
    cov_pool = 0.5 * (cov_x + cov_y)
    cov_pool = 0.5 * (cov_pool + cov_pool.T)

    d = int(cov_pool.shape[0])
    if d == 0:
        return np.nan
    tr_cov = float(np.trace(cov_pool))
    scale = max(tr_cov / float(max(1, d)), 1e-12)
    lam = float(max(ridge, 1e-12)) * scale
    reg_cov = cov_pool + lam * np.eye(d, dtype=float)

    delta = (mu_x - mu_y).astype(float)
    try:
        sol = accelerated_solve(reg_cov, delta, backend=hardware_backend)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(reg_cov).dot(delta)
    d2 = float(delta.dot(sol))
    if not np.isfinite(d2):
        return np.nan
    return float(max(d2, 0.0))


def _mean_pairwise_pattern_similarity(mat, hardware_backend=None):
    x = np.asarray(mat, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        return np.nan
    x = x - x.mean(axis=1, keepdims=True)
    norms = accelerated_row_norm(x, axis=1, backend=hardware_backend)
    keep = norms > 1e-12
    x = x[keep]
    norms = norms[keep]
    if x.shape[0] < 2:
        return np.nan
    x = x / norms[:, None]
    sim = accelerated_dot(x, x.T, backend=hardware_backend)
    iu = np.triu_indices(sim.shape[0], k=1)
    vals = sim[iu]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.clip(vals.mean(), -1.0, 1.0))


def _leading_latent_axis(pre_a, pre_b, hardware_backend=None):
    x = np.vstack([np.asarray(pre_a, dtype=float), np.asarray(pre_b, dtype=float)])
    if x.ndim != 2 or x.shape[0] < 2:
        return None
    x = x - x.mean(axis=0, keepdims=True)
    try:
        backend = resolve_hardware_backend(hardware_backend)
        if backend.accelerator:
            xp = get_array_module(backend)
            _, _, vt_d = xp.linalg.svd(xp.asarray(x), full_matrices=False)
            vt = to_numpy(vt_d)
        else:
            _, _, vt = np.linalg.svd(x, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if vt.ndim != 2 or vt.shape[0] == 0:
        return None
    axis = np.asarray(vt[0], dtype=float).reshape(-1)
    nrm = float(np.linalg.norm(axis))
    if nrm <= 1e-12:
        return None
    return axis / nrm


def compute_SRPI(
    ts: np.ndarray,
    tr: float = None,
    self_onsets: list = None,
    nonself_onsets: list = None,
    directional: bool = False,
    eps: float = 1e-8,
    modality: str = "fmri",
    pre_window_sec: float = 2.0,
    response_lag_sec: float = 4.0,
    response_window_sec: float = 6.0,
    covariance_ridge: float = 1e-3,
    component_weights=(0.35, 0.25, 0.20, 0.20),
    min_events_per_class: int = 3,
    sample_reliability_tau: float = 4.0,
    return_details: bool = False,
    hardware_backend=None,
) -> float:
    """
    Self-Referential Processing Index (SRPI).

    SRPI quantifies a measurable minimal self-model from self vs non-self
    event-locked activity using four components:
      1) self-reactivity bias,
      2) self/non-self representational separability,
      3) excess within-self pattern stability, and
      4) excess coupling of self responses to pre-event internal state.

    The final SRPI is the weighted geometric mean of these components after an
    event-count reliability attenuation.

    Parameters
    ----------
    ts : ndarray, shape (n_regions, n_time)
        Timeseries matrix.
    tr : float
        Sampling interval (seconds).
    self_onsets : list of float
        Self-related event onsets (seconds).
    nonself_onsets : list of float
        Non-self event onsets (seconds).
    directional : bool, optional
        If ``True``, maps the signed self-vs-nonself reactivity contrast from
        [-1,1] into [0,1] for the reactivity component; if ``False`` (default),
        only positive self-preference contributes.
    eps : float, optional
        Numerical stabilizer.
    modality : {'fmri', 'eeg'}, optional
        Modality label used for traceability in diagnostics.
    pre_window_sec : float, optional
        Pre-event window (seconds) for internal-state estimation.
    response_lag_sec : float, optional
        Post-onset lag (seconds) before response window begins.
    response_window_sec : float, optional
        Response window length (seconds).
    covariance_ridge : float, optional
        Ridge regularization strength for separability estimation.
    component_weights : tuple(float, float, float, float), optional
        Weights for (reactivity, separability, stability, internal coupling).
    min_events_per_class : int, optional
        Minimum required self and non-self events after windowing.
    sample_reliability_tau : float, optional
        Saturation constant for event-count reliability attenuation.
    return_details : bool, optional
        If ``True``, returns diagnostics dict.
    """
    if ts.ndim != 2:
        raise ValueError(f"ts should be 2D (n_regions × n_time), got shape {ts.shape}")
    if tr is None or (not np.isfinite(tr)) or float(tr) <= 0:
        raise ValueError("tr must be provided and > 0 for SRPI")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if pre_window_sec <= 0 or response_window_sec <= 0:
        raise ValueError("pre_window_sec and response_window_sec must be > 0")
    if response_lag_sec < 0:
        raise ValueError("response_lag_sec must be >= 0")
    if covariance_ridge <= 0:
        raise ValueError("covariance_ridge must be > 0")
    if min_events_per_class < 2:
        raise ValueError("min_events_per_class must be >= 2")
    if sample_reliability_tau <= 0:
        raise ValueError("sample_reliability_tau must be > 0")

    weights = np.asarray(component_weights, dtype=float).reshape(-1)
    if weights.size != 4:
        raise ValueError(
            "component_weights must be a 4-tuple: "
            "(reactivity, separability, stability, internal_coupling)"
        )
    if np.any(weights < 0):
        raise ValueError("component_weights must be non-negative")
    if not np.any(weights > 0):
        raise ValueError("At least one SRPI component weight must be > 0")

    backend = resolve_hardware_backend(hardware_backend)
    n_regions, n_tp = ts.shape
    _, self_idx = _sanitize_onset_seconds(self_onsets, tr=tr, n_tp=n_tp)
    _, non_idx = _sanitize_onset_seconds(nonself_onsets, tr=tr, n_tp=n_tp)

    counts = {
        "n_self_events_raw": int(self_idx.size),
        "n_nonself_events_raw": int(non_idx.size),
        "n_self_events_used": 0,
        "n_nonself_events_used": 0,
    }
    windows_sec = {
        "pre_window_sec": float(pre_window_sec),
        "response_lag_sec": float(response_lag_sec),
        "response_window_sec": float(response_window_sec),
    }
    if self_idx.size == 0 and non_idx.size == 0:
        return _srpi_undefined_result(
            reason="missing_self_and_nonself_events",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )
    if self_idx.size == 0:
        return _srpi_undefined_result(
            reason="missing_self_events",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )
    if non_idx.size == 0:
        return _srpi_undefined_result(
            reason="missing_nonself_events",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )

    ts_z = accelerated_zscore(ts, axis=1, backend=backend, eps=1e-12)
    pre_samples = max(1, int(round(float(pre_window_sec) / float(tr))))
    lag_samples = int(round(float(response_lag_sec) / float(tr)))
    post_samples = max(1, int(round(float(response_window_sec) / float(tr))))

    pre_self, delta_self, used_self = _event_locked_state_deltas(
        ts_z=ts_z,
        event_idx=self_idx,
        lag_samples=lag_samples,
        pre_samples=pre_samples,
        post_samples=post_samples,
    )
    pre_non, delta_non, used_non = _event_locked_state_deltas(
        ts_z=ts_z,
        event_idx=non_idx,
        lag_samples=lag_samples,
        pre_samples=pre_samples,
        post_samples=post_samples,
    )
    counts["n_self_events_used"] = int(used_self.size)
    counts["n_nonself_events_used"] = int(used_non.size)

    if used_self.size < int(min_events_per_class):
        return _srpi_undefined_result(
            reason="insufficient_self_events_after_windowing",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )
    if used_non.size < int(min_events_per_class):
        return _srpi_undefined_result(
            reason="insufficient_nonself_events_after_windowing",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )

    mag_self = accelerated_row_norm(delta_self, axis=1, backend=backend)
    mag_non = accelerated_row_norm(delta_non, axis=1, backend=backend)
    gamma_self = float(np.mean(mag_self))
    gamma_non = float(np.mean(mag_non))
    signed_bias = float((gamma_self - gamma_non) / (gamma_self + gamma_non + float(eps)))
    signed_bias = float(np.clip(signed_bias, -1.0, 1.0))
    if directional:
        c_reactivity = 0.5 * (signed_bias + 1.0)
    else:
        c_reactivity = max(signed_bias, 0.0)
    c_reactivity = float(np.clip(c_reactivity, 0.0, 1.0))

    d2 = _regularized_mahalanobis_distance_sq(
        x=delta_self,
        y=delta_non,
        ridge=float(covariance_ridge),
        hardware_backend=backend,
    )
    if not np.isfinite(d2):
        return _srpi_undefined_result(
            reason="separability_undefined",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )
    c_sep = float(np.clip(1.0 - np.exp(-0.5 * float(d2)), 0.0, 1.0))

    rho_self = _mean_pairwise_pattern_similarity(delta_self, hardware_backend=backend)
    rho_non = _mean_pairwise_pattern_similarity(delta_non, hardware_backend=backend)
    if (not np.isfinite(rho_self)) or (not np.isfinite(rho_non)):
        return _srpi_undefined_result(
            reason="stability_undefined",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )
    c_stability = float(np.clip((float(rho_self) - float(rho_non)) / 2.0, 0.0, 1.0))

    axis = _leading_latent_axis(pre_self, pre_non, hardware_backend=backend)
    if axis is None:
        return _srpi_undefined_result(
            reason="internal_state_axis_undefined",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )
    state_self = pre_self.dot(axis)
    state_non = pre_non.dot(axis)
    corr_self = _safe_abs_corr(state_self, mag_self)
    corr_non = _safe_abs_corr(state_non, mag_non)
    if (not np.isfinite(corr_self)) or (not np.isfinite(corr_non)):
        return _srpi_undefined_result(
            reason="internal_state_coupling_undefined",
            return_details=return_details,
            counts=counts,
            modality=modality,
            windows_sec=windows_sec,
            eps=eps,
        )
    c_internal = float(np.clip(float(corr_self) - float(corr_non), 0.0, 1.0))

    reliability = _sample_reliability(
        min(int(used_self.size), int(used_non.size)),
        tau=float(sample_reliability_tau),
    )
    comp_raw = np.asarray(
        [c_reactivity, c_sep, c_stability, c_internal],
        dtype=float,
    )
    comp = np.clip(comp_raw * float(reliability), 0.0, 1.0)

    if np.any((comp <= 0.0) & (weights > 0.0)):
        srpi_value = 0.0
    else:
        srpi_value = float(
            np.exp(np.sum(weights * np.log(comp)) / np.sum(weights))
        )
        srpi_value = float(np.clip(srpi_value, 0.0, 1.0))

    if not return_details:
        return srpi_value

    return {
        "value": float(srpi_value),
        "undefined_reason": None,
        "components": {
            "reactivity_bias": float(comp[0]),
            "representational_separability": float(comp[1]),
            "self_pattern_stability": float(comp[2]),
            "internal_state_coupling": float(comp[3]),
        },
        "components_raw": {
            "reactivity_bias": float(comp_raw[0]),
            "representational_separability": float(comp_raw[1]),
            "self_pattern_stability": float(comp_raw[2]),
            "internal_state_coupling": float(comp_raw[3]),
        },
        "signed_reactivity_bias": float(signed_bias),
        "reliability": float(reliability),
        "weights": {
            "reactivity_bias": float(weights[0]),
            "representational_separability": float(weights[1]),
            "self_pattern_stability": float(weights[2]),
            "internal_state_coupling": float(weights[3]),
        },
        "counts": dict(counts),
        "modality": str(modality),
        "windows_sec": dict(windows_sec),
        "eps": float(eps),
    }
