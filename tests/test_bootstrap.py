import pandas as pd
from analysis_bootstrap import bootstrap_ci,permutation_test_auc

def test_bootstrap_and_perm():
    df=pd.DataFrame({'session':['awake','sedation'],'S':[0.1,0.9]})
    ci=bootstrap_ci(df,'S',n_boot=10)
    auc,p=permutation_test_auc(df,'S',n_perm=10)
    assert len(ci)==2
    assert 0<=p<=1
