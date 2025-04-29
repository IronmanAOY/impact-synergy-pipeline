import os, glob, logging
import numpy as np, pandas as pd
from bids import BIDSLayout
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.signal import clean

log = logging.getLogger(__name__)

ATLAS_GLOBS = {
    'schaefer400': 'atlases/Schaefer*400Parcels*.nii*',
    'aal90':        'atlases/AAL/*.nii*',
    'shen268':      'atlases/shen*268*.nii*'
}

def collect_confounds(layout, subj, ses):
    try:
        cf = layout.get(subject=subj, session=ses,
                        suffix='desc-confounds_regressors',
                        extension='tsv')[0].path
    except IndexError:
        log.warning(f"No confounds for {subj} {ses}")
        raise
    df = pd.read_csv(cf, sep='\t')
    return df.filter(like='a_comp_cor_').values, df['framewise_displacement'].values

def find_atlas(key):
    files = glob.glob(ATLAS_GLOBS[key])
    if not files:
        log.warning(f"No atlas file found (pattern {ATLAS_GLOBS[key]})")
        raise FileNotFoundError(f"No atlas for {key}")
    return files[0]

def preprocess_subject(bids_root, subj, ses, out_dir):
    layout = BIDSLayout(bids_root, validate=False)
    try:
        bold_path = layout.get(subject=subj, session=ses,
                               suffix='desc-preproc_bold',
                               extension='nii.gz')[0].path
    except IndexError:
        log.warning(f"No preprocessed BOLD for {subj} {ses}")
        return

    compcor, fd = collect_confounds(layout, subj, ses)
    img = image.load_img(bold_path)
    zooms = img.header.get_zooms()
    tr = float(zooms[-1])
    if tr > 10:
        log.warning(f"Suspicious TR={tr}, falling back to 2.0s")
        tr = 2.0

    data = img.get_fdata().reshape(-1, img.shape[-1]).T
    cleaned = clean(signals=data, confounds=compcor,
                    standardize=True, detrend=True,
                    low_pass=0.1, high_pass=0.01, t_r=tr).T
    cleaned_img = cleaned.reshape(img.shape)

    os.makedirs(out_dir, exist_ok=True)
    cleaned_fn = os.path.join(out_dir, f"{subj}_{ses}_cleaned_bold.nii.gz")
    image.new_img_like(img, cleaned_img).to_filename(cleaned_fn)
    with open(os.path.join(out_dir, 'mean_fd.txt'), 'w') as f:
        f.write(str(np.mean(fd)))

    for key in ATLAS_GLOBS:
        atlas_file = find_atlas(key)
        masker = NiftiLabelsMasker(labels_img=atlas_file, standardize=True)
        ts = masker.fit_transform(cleaned_fn)
        np.save(os.path.join(out_dir, f"{subj}_{ses}_{key}_ts.npy"), ts)

def run_preprocessing(bids_root, out_root):
    layout = BIDSLayout(bids_root, validate=False)
    subs = sorted({f.subject for f in layout.get(suffix='desc-preproc_bold')})
    for subj in subs:
        sessions = sorted({f.session for f in layout.get(subject=subj,
                                                         suffix='desc-preproc_bold')})
        for ses in sessions:
            preprocess_subject(bids_root, subj, ses,
                               os.path.join(out_root, subj, ses))
