import os
import pandas as pd
import numpy as np
import statsmodels.api as sm

def motion_covariate_analysis(df, data_dir):
    """
    For each session in df, read the mean FD from
        {data_dir}/{subject}/{session}/mean_fd.txt
    then regress CI ~ FD by OLS and return a one‐row DataFrame
    with columns coef_{session} and p_{session}.
    """
    # Copy to avoid mutating the input
    df_copy = df.copy()
    # Load FD
    df_copy['FD'] = df_copy.apply(
        lambda r: float(
            open(os.path.join(data_dir, r.subject, r.session, 'mean_fd.txt'))
            .read().strip()
        ),
        axis=1
    )

    out = {}
    # Fit a separate regression for each session
    for session, grp in df_copy.groupby('session'):
        # Default to NaN
        coef, pval = np.nan, np.nan

        # Only fit if we have at least 2 points
        if len(grp) >= 2:
            try:
                # Simple OLS of CI on FD
                model = sm.OLS(grp['CI'], sm.add_constant(grp['FD']))
                res = model.fit()
                coef = res.params.get('FD', np.nan)
                pval = res.pvalues.get('FD', np.nan)
            except Exception:
                # In the unlikely event OLS itself fails
                coef, pval = np.nan, np.nan

        out[f'coef_{session}'] = coef
        out[f'p_{session}']    = pval

    # Return as a one‐row DataFrame
    return pd.DataFrame([out])

