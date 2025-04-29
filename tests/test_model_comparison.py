import pandas as pd
from model_comparison import compare_models

def test_compare_models():
    df=pd.DataFrame({
        'session':['awake','sedation'],
        'S':[0.2,0.8],
        'mean_conn':[0.5,0.6],
        'modularity':[0.1,0.2],
        'pci_fmri':[0.3,0.4]
    })
    out=compare_models(df)
    assert 'mean_conn' in out
