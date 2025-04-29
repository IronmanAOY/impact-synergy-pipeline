import run_pipeline

def test_smoke(tmp_path,monkeypatch):
    for fn in ("run_preprocessing","compute_synergy_ci",
               "compute_baseline_metrics","bootstrap_ci",
               "permutation_test_auc","motion_covariate_analysis",
               "atlas_check","run_replication","compare_models",
               "create_doc"):
        monkeypatch.setattr(run_pipeline,fn,lambda *a,**k: None)
    run_pipeline.main(str(tmp_path))
