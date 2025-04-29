import os
import argparse
from bids import BIDSLayout
from preprocessing import run_preprocessing
from synergy_ci import compute_synergy_ci
from analysis_bootstrap import bootstrap_ci

def run_replication(data_root, out_dir, atlas, sessions):
    layout = BIDSLayout(data_root, validate=False)
    if not layout.get(suffix='desc-preproc_bold'):
        raise RuntimeError("No fMRIPrep outputs—run scripts/fetch_fmriprep.sh")
    prep = os.path.join(out_dir, 'preprocessed')
    run_preprocessing(data_root, prep)

    s0, s1 = sessions
    df = compute_synergy_ci(prep, atlas, thetas=[0.5], sessions=sessions)
    m0, m1 = df[df.session==s0]['S'].mean(), df[df.session==s1]['S'].mean()
    delta = m0 - m1
    lo, hi = bootstrap_ci(df, 'S')
    coh = delta / df.S.std()
    return {'delta_S': delta, 'ci': (lo, hi), 'cohend': coh}

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('data_root')
    p.add_argument('--out-dir', default='outputs/melbourne')
    p.add_argument('--atlas', default='schaefer400')
    p.add_argument('--sessions', nargs=2, default=['awake','deep'])
    args = p.parse_args()
    print(run_replication(args.data_root,
                          args.out_dir,
                          args.atlas,
                          tuple(args.sessions)))
