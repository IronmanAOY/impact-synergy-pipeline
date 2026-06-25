import os
import glob
import numpy as np
import pandas as pd
import statsmodels.api as sm

def _read_fd_file(path: str) -> float:
    with open(path, 'r') as f:
        return float(f.read().strip())

def _weighted_session_fd(data_dir: str, subject: str, session: str, atlas: str) -> float:
    """
    Weighted mean FD across all conditions/runs in a session.
    Weights are the number of timepoints from the TS used in CI.
    Looks under: {data_dir}/{subject}/{session}/**/mean_fd.txt
    """
    root = os.path.join(data_dir, subject, session)
    fd_paths = sorted(glob.glob(os.path.join(root, "**", "mean_fd.txt"), recursive=True))
    if not fd_paths:
        raise FileNotFoundError(f"No mean_fd.txt under {root}")

    fds, weights = [], []
    for fd_path in fd_paths:
        fd = _read_fd_file(fd_path)
        cond_dir = os.path.dirname(fd_path)

        # Match the TS file(s) for this condition/run
        ts_candidates = sorted(glob.glob(os.path.join(
            cond_dir, f"{subject}_run-*_{atlas}_ts.npy"
        )))
        if not ts_candidates:
            # fallback (handles slightly different naming styles)
            ts_candidates = sorted(glob.glob(os.path.join(
                cond_dir, f"{subject}_*_{atlas}_ts.npy"
            )))

        if ts_candidates:
            try:
                ts = np.load(ts_candidates[0], mmap_mode='r')
                # TS shape is (n_regions, n_time); weight by timepoints.
                n_tp = int(ts.shape[1])
            except Exception:
                n_tp = 1
        else:
            n_tp = 1  # conservative fallback if TS is missing

        fds.append(fd)
        weights.append(n_tp)

    return float(np.average(fds, weights=weights))

def motion_covariate_analysis(df_agg, data_dir: str, atlas: str = "schaefer400"):
    """
    df_agg should be one row per (subject, session), e.g., df_mean from run_s_ci.
    Computes FD per (subject, session) as timepoint-weighted mean across conditions,
    then fits CI ~ FD separately within each session. Returns a one-row DataFrame
    with coef_{session} and p_{session}.
    """
    # Make a copy so we don't mutate the input
    dfc = df_agg.copy()

    # Pre-compute FD once per subject/session
    keys = dfc[['subject','session']].drop_duplicates()
    fd_map = {}
    for _, row in keys.iterrows():
        sub, ses = row['subject'], row['session']
        fd_map[(sub, ses)] = _weighted_session_fd(data_dir, sub, ses, atlas)

    dfc['FD'] = dfc.apply(lambda r: fd_map[(r.subject, r.session)], axis=1)

    out = {}
    # Regress within each session separately (awake vs deep)
    for ses, grp in dfc.groupby('session'):
        coef, pval = np.nan, np.nan
        if len(grp) >= 2:
            X = sm.add_constant(grp['FD'])
            y = grp['CI']
            res = sm.OLS(y, X).fit()
            coef = res.params.get('FD', np.nan)
            pval = res.pvalues.get('FD', np.nan)
        out[f'coef_{ses}'] = coef
        out[f'p_{ses}']    = pval

    return pd.DataFrame([out])
