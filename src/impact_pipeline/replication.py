import os
import argparse
import glob
import json
from bids import BIDSLayout
from impact_pipeline.preprocessing import run_preprocessing
from impact_pipeline.synergy_ci import compute_synergy_ci
from impact_pipeline.analysis_bootstrap import bootstrap_ci


def _infer_tr_from_bold_sidecar(bids_root):
    sidecars = sorted(glob.glob(os.path.join(bids_root, "sub-*", "func", "*_bold.json")))
    if not sidecars:
        raise ValueError(
            "No *_bold.json sidecar found; explicit TR is required for NAS computation."
        )
    with open(sidecars[0], "r", encoding="utf-8") as f:
        sidecar = json.load(f)
    tr = sidecar.get("RepetitionTime")
    if tr is None:
        raise ValueError(
            f"Missing RepetitionTime in sidecar '{sidecars[0]}'; explicit TR is required."
        )
    tr = float(tr)
    if tr <= 0:
        raise ValueError(f"Invalid RepetitionTime={tr} in '{sidecars[0]}'.")
    return tr


def _replication_nas_params():
    return {
        "zthr": 1.0,
        "eps": 0.2,
        "tau": 0.2,
        "lambda_phase": 0.5,
        "alpha": 0.20,
        "beta": 0.16,
        "gamma": 0.14,
        "delta": 0.12,
        "eta": 0.16,
        "zeta": 0.12,
        "rho": 0.10,
        "bands": ((0.01, 0.10),),
        "band_weights": (1.0,),
        "window_len": 30,
        "step_len": 15,
        "max_triads": 5000,
        "random_state": 0,
        "workspace_nodes": None,
        "workspace_quantile": 0.2,
        "workspace_min_size": 4,
        "directed_lag": 1,
        "reverberation_lags": (2, 3, 4),
        "baseline_ts": None,
        "boost_against_baseline": False,
        "normalize": True,
    }


def run_replication(data_root, out_dir, atlas, sessions):
    layout = BIDSLayout(data_root, validate=False)
    if not layout.get(suffix='desc-preproc_bold'):
        raise RuntimeError(
            "No fMRIPrep outputs in data/scratch/melbourne/derivatives—"
            "run scripts/fetch_fmriprep.sh"
        )
    prep = os.path.join(out_dir, 'preprocessed')
    run_preprocessing(data_root, prep)

    s0, s1 = sessions
    tr = _infer_tr_from_bold_sidecar(data_root)
    df = compute_synergy_ci(
        prep,
        atlas,
        thetas=[0.5],
        sessions=sessions,
        tr=tr,
        nas_params=_replication_nas_params(),
    )
    m0, m1 = df[df.session==s0]['S'].mean(), df[df.session==s1]['S'].mean()
    delta = m0 - m1
    lo, hi = bootstrap_ci(df, 'S')
    coh = delta / df.S.std()
    return {'delta_S': delta, 'ci': (lo, hi), 'cohend': coh}

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('data_root')
    p.add_argument('--out-dir', default='outputs/scratch/melbourne')
    p.add_argument('--atlas', default='schaefer400')
    p.add_argument('--sessions', nargs=2, default=['awake','deep'])
    args = p.parse_args()
    print(run_replication(args.data_root,
                          args.out_dir,
                          args.atlas,
                          tuple(args.sessions)))
