#!/usr/bin/env python
# ensure project root is on pytest’s PYTHONPATH
import sys
from pathlib import Path

import argparse
import logging
import random
import numpy as np

from preprocessing import run_preprocessing
from synergy_ci import compute_synergy_ci
from baseline_metrics import compute_baseline_metrics
from analysis_bootstrap import bootstrap_ci, permutation_test_auc
from motion_model import motion_covariate_analysis
from atlas_robustness import atlas_check
from replication import run_replication
from model_comparison import compare_models
from scripts.generate_word_doc import create_doc

root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

random.seed(42)
np.random.seed(42)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("pipeline")

REPORT_DOC = "Parameter_Sensitivity_and_Pilot_Simulation.docx"

def main(out_dir):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)

    log.info("1/9 Preprocessing ds003171")
    run_preprocessing('data/ds003171', str(out/'cambridge'/'preprocessed'))

    log.info("2/9 Synergy & CI")
    thetas = np.arange(0.1, 1.0, 0.1)
    df = compute_synergy_ci(
        str(out/'cambridge'/'preprocessed'),
        atlas='schaefer400',
        thetas=thetas,
        sessions=('awake','sedation')
    )

    log.info("3/9 Baseline metrics")
    df = compute_baseline_metrics(
        df,
        data_dir=str(out/'cambridge'/'preprocessed'),
        atlas='schaefer400'
    )

    log.info("4/9 Bootstrap & Permutation")
    boot_S  = bootstrap_ci(df, 'S')
    boot_CI = bootstrap_ci(df, 'CI')

    # initialize to None in case permutation_test_auc returns None
    auc_S, p_S   = None, None
    auc_CI, p_CI = None, None

    res = permutation_test_auc(df, 'S')
    if res is not None:
        auc_S, p_S = res

    res = permutation_test_auc(df, 'CI')
    if res is not None:
        auc_CI, p_CI = res

    log.info("5/9 Motion model")
    motion = motion_covariate_analysis(df, str(out/'cambridge'/'preprocessed'))

    log.info("6/9 Atlas robustness")
    atlas_res = atlas_check(
        str(out/'cambridge'/'preprocessed'),
        ('aal90','shen268')
    )

    log.info("7/9 Replication (Melbourne)")
    repl = run_replication(
        data_root='data/melbourne',
        out_dir=str(out/'melbourne'),
        atlas='schaefer400',
        sessions=('awake','deep')
    )

    log.info("8/9 Model comparison")
    mc = compare_models(df, metrics=('mean_conn','modularity','pci_fmri'))

    log.info("9/9 Generate report")
    create_doc(
        path=str(out/REPORT_DOC),
        boot_S=boot_S, p_S=p_S,
        boot_CI=boot_CI, p_CI=p_CI,
        auc_S=auc_S, auc_CI=auc_CI,
        motion=motion, atlas=atlas_res,
        repl=repl, mc=mc
    )

    log.info("Done. Outputs in %s", out)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', default='outputs')
    args = p.parse_args()
    main(args.out_dir)
