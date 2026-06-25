import numpy as np
import pandas as pd
from scipy import stats
from impact_pipeline.synergy_ci import compute_synergy_ci


def _paired_summary(agg: pd.DataFrame, metric: str, sessions: tuple[str, ...]) -> dict:
    out = {
        "n": 0,
        "mean_a": np.nan,
        "mean_b": np.nan,
        "delta_a_minus_b": np.nan,
        "t": np.nan,
        "p": np.nan,
        "d_paired": np.nan,
    }
    if metric not in agg.columns or len(sessions) < 2:
        return out
    s0, s1 = str(sessions[0]), str(sessions[1])
    piv = agg.pivot(index="subject", columns="session", values=metric)
    if (s0 not in piv.columns) or (s1 not in piv.columns):
        return out
    paired = piv[[s0, s1]].dropna()
    n = int(len(paired))
    out["n"] = n
    if n == 0:
        return out
    a = paired[s0].to_numpy(dtype=float)
    b = paired[s1].to_numpy(dtype=float)
    diff = a - b
    out["mean_a"] = float(np.nanmean(a))
    out["mean_b"] = float(np.nanmean(b))
    out["delta_a_minus_b"] = float(np.nanmean(diff))
    if n < 2:
        return out
    try:
        t_val, p_val = stats.ttest_rel(a, b, nan_policy="omit")
        out["t"] = float(t_val)
        out["p"] = float(p_val)
    except Exception:
        pass
    sd = float(np.nanstd(diff, ddof=1))
    if np.isfinite(sd) and sd > 0:
        out["d_paired"] = float(np.nanmean(diff) / sd)
    return out


def atlas_check(
    data_dir,
    atlases=('aal90', 'shen268'),
    sessions=('awake', 'deep'),
    thetas=None,
    tr=None,
    stimulus_onsets=None,
    mpc_metrics=None,
    compute_ci=False,
    pdi_params=None,
    pdi_require_explicit_params=False,
    pdi_require_strict_baseline=False,
    pdi_primary_endpoint="anchor",
    nas_params=None,
    srpi_params=None,
    srpi_require_explicit_params=True,
    subjects=None,
):
    out = {}
    thresholds = thetas if thetas is not None else [i * 0.1 for i in range(1, 10)]
    valid_mpc = ("RAM", "PDI", "NAS", "IIM", "SRPI")
    if mpc_metrics is None:
        # Default robustness set for event-sparse datasets.
        mpc_eff = ("PDI", "NAS", "IIM")
    else:
        mpc_eff = tuple(dict.fromkeys([m for m in mpc_metrics if m in valid_mpc]))

    # RAM/SRPI require event timing to be meaningful.
    if stimulus_onsets is None:
        mpc_eff = tuple(m for m in mpc_eff if m not in {"RAM", "SRPI"})
    ci_enabled = bool(compute_ci and set(("RAM", "PDI", "NAS", "IIM", "SRPI")).issubset(set(mpc_eff)))

    for atlas in atlases:
        df = compute_synergy_ci(
            data_dir,
            atlas=atlas,
            thetas=thresholds,
            sessions=sessions,
            tr=tr,
            stimulus_onsets=stimulus_onsets,
            compute_mpc=bool(len(mpc_eff) > 0),
            mpc_metrics=(None if len(mpc_eff) == 0 else mpc_eff),
            compute_ci=ci_enabled,
            pdi_params=pdi_params,
            pdi_require_explicit_params=pdi_require_explicit_params,
            pdi_require_strict_baseline=pdi_require_strict_baseline,
            pdi_primary_endpoint=pdi_primary_endpoint,
            nas_params=nas_params,
            srpi_params=srpi_params,
            srpi_require_explicit_params=srpi_require_explicit_params,
            subjects=subjects,
        )
        if not isinstance(df, pd.DataFrame):
            raise TypeError("compute_synergy_ci must return DataFrame")
        rough = {}
        for ses in sessions:
            sub = df[df.session == ses]
            s = sub.sort_values('theta')['S'].values
            rough[ses] = np.sqrt(np.mean(np.diff(s) ** 2)) if len(s) > 1 else np.nan
        agg_map = {"S": ("S", "mean")}
        for metric in ("CI", "RAM", "PDI", "NAS", "IIM", "SRPI", "IIM_raw", "IIM_raw_scaled"):
            if metric in df.columns:
                agg_map[metric] = (metric, "mean")
        agg = df.groupby(["subject", "session"]).agg(**agg_map).reset_index()

        metric_stats = {
            "S": _paired_summary(agg, "S", sessions),
        }
        for metric in ("PDI", "NAS", "IIM", "IIM_raw", "RAM", "SRPI", "CI"):
            if metric in agg.columns:
                metric_stats[metric] = _paired_summary(agg, metric, sessions)

        notes = []
        if stimulus_onsets is None and any(m in ("RAM", "SRPI") for m in (mpc_metrics or ())):
            notes.append("RAM/SRPI omitted in atlas robustness because no event timings were provided.")

        payload = {
            # Legacy top-level fields retained for backward compatibility.
            str(sessions[0]) if len(sessions) > 0 else "session0": rough.get(sessions[0], np.nan) if len(sessions) > 0 else np.nan,
            str(sessions[1]) if len(sessions) > 1 else "session1": rough.get(sessions[1], np.nan) if len(sessions) > 1 else np.nan,
            "roughness": rough,
            "metrics": metric_stats,
            "mpc_metrics_used": list(mpc_eff),
            "ci_enabled": ci_enabled,
            "n_subjects": int(agg["subject"].nunique()) if not agg.empty else 0,
            "n_rows": int(len(df)),
            "notes": notes,
        }
        out[atlas] = payload
    return out
