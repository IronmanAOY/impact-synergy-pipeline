import os
import statsmodels.formula.api as smf

def motion_covariate_analysis(df, data_dir):
    df2 = df.copy()
    fds = []
    for _, r in df2.iterrows():
        fn = os.path.join(data_dir, r.subject, r.session, 'mean_fd.txt')
        if not os.path.exists(fn):
            raise FileNotFoundError(f"Missing confound file {fn}")
        with open(fn) as f:
            fds.append(float(f.read()))
    df2['mean_fd'] = fds
    df2['is_awake'] = (df2.session == 'awake').astype(int)
    model = smf.mixedlm("S ~ is_awake + mean_fd", df2, groups=df2.subject)
    res = model.fit()
    return {'coef_awake': res.params['is_awake'],
            'p_awake': res.pvalues['is_awake']}
