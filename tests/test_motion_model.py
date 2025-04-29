import pandas as pd
from motion_model import motion_covariate_analysis

def test_motion_model(tmp_path):
    data_dir=tmp_path/"data"
    row={'subject':'s1','session':'awake','S':0.5,'CI':0.5}
    df=pd.DataFrame([row])
    d=data_dir/"s1"/"awake"
    d.mkdir(parents=True)
    (d/"mean_fd.txt").write_text("0.2")
    res=motion_covariate_analysis(df,str(data_dir))
    assert 'coef_awake' in res and 'p_awake' in res
