#!/usr/bin/env python
# ensure project root is on pytest’s PYTHONPATH
import sys
from pathlib import Path
import argparse
import logging
import random
import numpy as np
import subprocess, os
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import run_preprocessing
from synergy_ci import compute_synergy_ci
from baseline_metrics import compute_baseline_metrics
from analysis_bootstrap import bootstrap_ci, permutation_test_auc
from motion_model import motion_covariate_analysis
from atlas_robustness import atlas_check
from replication import run_replication
from model_comparison import compare_models
from scripts.generate_word_doc import create_doc
from bids import BIDSLayout

# ---------------------------------------------------------------------
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

random.seed(42)
np.random.seed(42)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pipeline")

# Name for the DOCX report
REPORT_DOC = "Synergy_Empirical_Validation.docx"

def ensure_fmriprep(bids_dir, fmriprep_out, work_dir, fs_license, freesurf_out):

    layout = BIDSLayout(bids_dir, validate=False)
    subjects = sorted(layout.get(return_type='id', target='subject'))
 

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(fmriprep_out, exist_ok=True)
    os.makedirs(freesurf_out, exist_ok=True)
    
    for sub in subjects:
        log.info("Running fMRIPrep on %s …", sub)
        cache_host = Path(work_dir) / 'bids_db'
        cache_cont = '/bids_db'
        cmd = [
          'docker','run','--rm',
          '-v', f'{bids_dir}:/data',
          '-v', f'{fmriprep_out}:/out',
          '-v', f'{work_dir}:/work',
          '-v', f'{freesurf_out}:/out_freesurfer',
          '-v', f'{Path(fs_license).parent.resolve()}:/licenses:ro',
          '-v', f'{cache_host}:{cache_cont}',
          'nipreps/fmriprep:25.1.3',
          '/data','/out','participant',
          '--participant-label', sub,
          '--fs-license-file',
          '/licenses/fs_license.txt', # make sure your freesurfer license is named and placed correctly
          '--fs-subjects-dir', '/out_freesurfer',
          '--bids-database-dir', cache_cont,
          '--work-dir','/work',
 #         '--clean-workdir',   # During development or iterative runs → leave out. I recommend to only enable for final clean runs on the full dataset to finalize and free space
          '--skip-bids-validation',
          '--nthreads', '16', # or however many logical CPUs you have
          '--omp-nthreads', '8', # ~ half of total
          '--mem', '96000', # adjust to your RAM




        ]
        subprocess.run(cmd, check=True)

        
def main(out_dir):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    figdir = out / 'pipe_figures'
    figdir.mkdir(exist_ok=True)
    
    fmriprep_out = out / 'fmriprep'
    prep_out     = out / 'preprocessed'
    freesurf_out = out / 'freesurfer'
    workdir      = out / 'work'
    bids_root    = root / 'data' / 'ds003171'
    
    # 0. RUN FMRIPrep ON MISSING SUBJECTS
#    ensure_fmriprep(
#        bids_dir=str(bids_root),
#        fmriprep_out=str(fmriprep_out),
#        work_dir=str(workdir),
#        fs_license=str(root/'licenses'/'fs_license.txt'),
#        freesurf_out=str(freesurf_out)
#    )

    # 1. PREPROCESSING
#    log.info("1/9 Preprocessing ds003171 (Cambridge Propofol)")
#    run_preprocessing(
#        bids_root=str(bids_root),
#        fmriprep_deriv=str(out/'fmriprep'),
#        out_root=str(prep_out)
#    )

    # 2. SYNERGY & CI
    log.info("2/9 Computing Synergy & Consciousness Index (CI)")
    thetas = np.arange(0.1, 1.0, 0.1)
    df = compute_synergy_ci(
        str(prep_out),
        atlas='schaefer400',
        thetas=thetas,
        sessions=('awake','deep'),
    )
    df_mean = (
        df
        .groupby(['subject','session'])
        .agg(S=('S','mean'), CI=('CI','mean'))
        .reset_index()
    )
    df_S = df_mean.pivot(index='subject', columns='session', values='S')
    df_CI = df_mean.pivot(index='subject', columns='session', values='CI')
    # 2b. SYNERGY & CI (fine grid for supplement)
    thetas_fine = np.arange(0.4, 0.81, 0.02)
    df_fine = compute_synergy_ci(str(prep_out),
                                atlas='schaefer400',
                                thetas=thetas_fine,
                                sessions=('awake','deep'))
    
    # compute mean±SEM of the difference at each θ
    means, sems = [], []
    for θ, subdf in df_fine.groupby('theta'):
        diff = (subdf.loc[subdf.session=='awake','S'].values
            - subdf.loc[subdf.session=='deep' , 'S'].values)
        means.append(diff.mean())
        sems.append(diff.std(ddof=1)/np.sqrt(len(diff)))
    means, sems = np.array(means), np.array(sems)
    
    # save a supplemental figure
    fig, ax = plt.subplots()
    ax.errorbar(thetas_fine, means, yerr=sems,
                marker='o', linestyle='-')
    ax.axvline(0.6, linestyle='--')
    ax.set(xlabel='θ', ylabel='Mean S_awake – S_deep',
            title='Supplementary: Synergy difference across θ')
    fig.tight_layout()
    fig.savefig(figdir / 'supp_theta_curve.png')
    
    # extract paired samples
    S_awake = df_S['awake'].values
    S_deep  = df_S['deep'].values
    CI_awake = df_CI['awake'].values
    CI_deep  = df_CI['deep'].values
    
    df_S_wide  = df.pivot(index='subject', columns=['theta','session'], values='S')
    df_CI_wide = df.pivot(index='subject', columns=['theta','session'], values='CI')
    # Per‐θ paired t‐tests and Cohen’s d for S
    rows = []
    for theta, subdf in df.groupby('theta'):
        awake = subdf.loc[subdf.session=='awake','S'].values
        deep  = subdf.loc[subdf.session=='deep', 'S'].values
        t, p  = stats.ttest_rel(awake, deep)
        d     = (awake - deep).mean() / (awake - deep).std(ddof=1)
        rows.append({'theta': theta, 't_S': t, 'p_S': p, 'd_S': d})
    
    df_stats_by_theta = pd.DataFrame(rows).set_index('theta')

    # number of subjects
    n = len(S_awake)
    df_deg = n - 1

    # 1) means & SDs
    mean_S_awake, sd_S_awake = S_awake.mean(), S_awake.std(ddof=1)
    mean_S_deep,  sd_S_deep  = S_deep.mean(),  S_deep.std(ddof=1)
    mean_CI_awake, sd_CI_awake = CI_awake.mean(), CI_awake.std(ddof=1)
    mean_CI_deep,  sd_CI_deep  = CI_deep.mean(),  CI_deep.std(ddof=1)

    # paired t‐tests
    t_S, p_S   = stats.ttest_rel(S_awake,  S_deep)
    t_CI, p_CI = stats.ttest_rel(CI_awake, CI_deep)

    # Cohen's d for paired samples:
    #    d = mean(diff) / sd(diff)
    d_S  = (S_awake  - S_deep).mean()  / (S_awake  - S_deep).std(ddof=1)
    d_CI = (CI_awake - CI_deep).mean() / (CI_awake - CI_deep).std(ddof=1)

    # pack into df_stats
    df_stats = {
        'n_subjects':    n,
        'df':            df_deg,
        'mean_S_awake':  mean_S_awake,
        'sd_S_awake':    sd_S_awake,
        'mean_S_deep':   mean_S_deep,
        'sd_S_deep':     sd_S_deep,
        't_S':           t_S,
        'p_S':           p_S,
        'd_S':           d_S,
        'mean_CI_awake': mean_CI_awake,
        'sd_CI_awake':   sd_CI_awake,
        'mean_CI_deep':  mean_CI_deep,
        'sd_CI_deep':    sd_CI_deep,
        't_CI':          t_CI,
        'p_CI':          p_CI,
        'd_CI':          d_CI,
    }

    # 3. BASELINE METRICS (mean connectivity, etc.)
    log.info("3/9 Computing baseline graph metrics")
    df_baseline = compute_baseline_metrics(
        df_mean,
        data_dir=str(prep_out),
        atlas='schaefer400'
    )
    print(">>> df_baseline columns:", df_baseline.columns.tolist())

    # 4. BOOTSTRAP & PERMUTATION TESTS
    log.info("4/9 Bootstrapping CIs and permutation tests")
    boot_S  = bootstrap_ci(df_mean, 'S')
    boot_CI = bootstrap_ci(df_mean, 'CI')

    auc_S, p_S = permutation_test_auc(df_mean, 'S') or (None, None)
    auc_CI, p_CI = permutation_test_auc(df_mean, 'CI') or (None, None)
    theta_results = {}
    for theta in sorted(df['theta'].unique()):
        subdf = df[df['theta'] == theta]
        df_theta_mean = (
            subdf.groupby(['subject','session'])
                 .agg(S=('S', 'mean'),
                      CI=('CI', 'mean'))
                 .reset_index()
        )
        auc_S_theta, p_S_theta   = permutation_test_auc(df_theta_mean, 'S') or (None, None)
        auc_CI_theta, p_CI_theta = permutation_test_auc(df_theta_mean, 'CI') or (None, None)
        theta_results[theta] = {
            'auc_S': auc_S_theta, 'p_S': p_S_theta,
            'auc_CI': auc_CI_theta, 'p_CI': p_CI_theta
        }

    # 5. MOTION COVARIATE ANALYSIS
    log.info("5/9 Motion covariate analysis")
    motion = motion_covariate_analysis(df, str(prep_out))

    # 6. ATLAS ROBUSTNESS
    log.info("6/9 Testing robustness across atlases")
    atlas_res = atlas_check(
        str(prep_out),
        atlases=('aal90','shen268')
    )
    atlas_res = {
    name: {'awake': v['awake'], 'deep': v['sedation']}
    for name, v in atlas_res.items()
}

    # 7. REPLICATION on an independent dataset
#    log.info("7/9 Replication (Melbourne Propofol)")
#    melb_root = root / 'data' / 'melbourne_propofol'  # <- point to a separate folder
#    melb_out  = out / 'melbourne' / 'preprocessed'
#    repl = run_replication(
#        data_root=str(melb_root),
#        out_dir=str(melb_out),
#        atlas='schaefer400',
#        sessions=('awake','deep')
#    )




    # 8. MODEL COMPARISONS
    log.info("8/9 Comparing to baseline and PCI models")
    mc = compare_models(
        df_baseline,
        metrics=('mean_conn', 'modularity', 'pci_fmri'),
        sessions=('awake', 'deep')
    )
    print(df_baseline['session'].value_counts())

    # 9. GENERATE FINAL REPORT
    log.info("9/9 Generating Word report")
    create_doc(
        str(out / REPORT_DOC),  # 1) path to save
        df_stats,               # 2) your stats dict
        df_stats_by_theta,
        motion,                 # 3) motion covariate results
        atlas_res,              # 4) atlas robustness results
        None,                   # 5) repl (None, since you’re not doing replication)
        mc,                      # 6) model‐comparison results
        theta_results,
        fig_dir=str(figdir)
    )

    log.info("Pipeline complete! Outputs in %s", out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Impact-Synergy pipeline")
    parser.add_argument('--out-dir', default='outputs',
                        help="Root folder for all outputs")
    args = parser.parse_args()
    main(args.out_dir)
