import os, glob, logging
import numpy as np, pandas as pd
from bids import BIDSLayout
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.signal import clean
from pathlib import Path
import warnings
import nibabel as nib

log = logging.getLogger(__name__)

ATLAS_GLOBS = {
    'schaefer400': 'atlases/Schaefer*400Parcels*.nii*',
    'aal90':        'atlases/AAL/*.nii*',
    'shen268':      'atlases/shen*268*.nii*'
}

def collect_confounds(layout, subject, session):
    compcor = None
    fd = 0.0

    # 1) CompCor
    try:
        confs = layout.get(
            subject=subject,
            session=session,
            suffix='bold',
            desc='preproc',
            extension='tsv',
            return_type='object'
        )
    except Exception:
        confs = []

    if confs:
        tsv_path = Path(confs[0].path)
        try:
            df = pd.read_csv(tsv_path, sep='\t')
            compcor_cols = [c for c in df.columns if 'compcor' in c.lower()]
            if compcor_cols:
                compcor = df[compcor_cols].values
            else:
                warnings.warn(f"No CompCor columns found in {tsv_path}")
        except Exception as e:
            warnings.warn(f"Could not read confounds TSV {tsv_path}: {e}")
    else:
        warnings.warn(f"No confounds TSV found for sub-{subject} ses-{session}")

    # 2) Mean FD
    try:
        bids_root = getattr(layout, 'root')
        fd_tsv = Path(bids_root) / f"sub-{subject}" / f"ses-{session}" / \
                 f"sub-{subject}_ses-{session}_desc-confounds_regressors.tsv"
        df_fd = pd.read_csv(fd_tsv, sep='\t')
        if 'framewise_displacement' in df_fd.columns:
            fd = df_fd['framewise_displacement'].mean()
        else:
            warnings.warn(f"No FD column in {fd_tsv}")
    except Exception:
        # catches both missing layout.root and missing file/column
        warnings.warn(f"Could not read FD for sub-{subject} ses-{session}; "
                      "falling back to 0.0")

    return compcor, fd
    
def find_atlas(key):
    files = glob.glob(ATLAS_GLOBS[key])
    if not files:
        log.warning(f"No atlas file found (pattern {ATLAS_GLOBS[key]})")
        raise FileNotFoundError(f"No atlas for {key}")
    return files[0]

def preprocess_subject(bids_root, subj, ses, out_dir):
    """
    Preprocess a single subject/session:
      1. Load preprocessed BOLD
      2. Load CompCor regressors and mean FD (falling back to None/0)
      3. Clean (filter, detrend, standardize)
      4. Save cleaned NIfTI (with TypeError fallback)
      5. Write mean FD
      6. Extract and save atlas timecourses

    Parameters
    ----------
    bids_root : str
        Path to BIDS dataset root.
    subj : str
        Subject label (without 'sub-').
    ses : str
        Session label (without 'ses-').
    out_dir : str
        Directory in which to write outputs.
    """
    layout = BIDSLayout(bids_root, validate=False)
    try:
        bold_path = layout.get(
            subject=subj, session=ses,
            suffix='desc-preproc_bold', extension='nii.gz'
        )[0].path
    except IndexError:
        log.warning(f"No preprocessed BOLD for sub-{subj} ses-{ses}")
        return

    compcor, fd = collect_confounds(layout, subj, ses)
    img = image.load_img(bold_path)
    tr = float(img.header.get_zooms()[-1])
    if tr > 10:
        log.warning(f"Suspicious TR={tr}, falling back to 2.0s")
        tr = 2.0

    # flatten to time × voxels, clean, then reshape back to 4D
    data = img.get_fdata().reshape(-1, img.shape[-1]).T
    cleaned = clean(
        signals=data, confounds=compcor,
        standardize=True, detrend=True,
        low_pass=0.1, high_pass=0.01, t_r=tr
    ).T
    cleaned_vol = cleaned.reshape(img.shape)

    os.makedirs(out_dir, exist_ok=True)
    cleaned_fn = os.path.join(out_dir, f"{subj}_{ses}_cleaned_bold.nii.gz")

    try:
        image.new_img_like(img, cleaned_vol).to_filename(cleaned_fn)
    except TypeError:
        # DummyImg may lack .affine; fall back to nibabel
        affine = getattr(img, 'affine', None)
        if affine is None:
            affine = np.eye(4)
        nib.Nifti1Image(cleaned_vol, affine).to_filename(cleaned_fn)

    # write out mean FD
    try:
        mean_fd = float(np.mean(fd))
    except Exception:
        mean_fd = 0.0
    with open(os.path.join(out_dir, 'mean_fd.txt'), 'w') as f:
        f.write(str(mean_fd))

    # now extract ROI time-series for each atlas (or fall back to empty)
    for key in ATLAS_GLOBS:
        try:
            atlas_img = find_atlas(key)
            masker    = NiftiLabelsMasker(
                            labels_img=atlas_img,
                            standardize=True,
                            t_r=tr
                        )
            # feed in the filename of the cleaned NIfTI
            ts        = masker.fit_transform(cleaned_fn)
        except FileNotFoundError:
            log.warning(f"No atlas for {key}, skipping ROI extraction")
            # fallback to an empty (T × 0) array
            n_tp = img.shape[-1]
            ts   = np.zeros((n_tp, 0), dtype=float)

        out_fname = os.path.join(out_dir, f"{subj}_{ses}_{key}_ts.npy")
        np.save(out_fname, ts)

def run_preprocessing(bids_root, out_root):
    layout = BIDSLayout(bids_root, validate=False)
    subs = sorted({f.subject for f in layout.get(suffix='desc-preproc_bold')})
    for subj in subs:
        sessions = sorted({f.session for f in layout.get(subject=subj,
                                                         suffix='desc-preproc_bold')})
        for ses in sessions:
            preprocess_subject(bids_root, subj, ses,
                               os.path.join(out_root, subj, ses))
