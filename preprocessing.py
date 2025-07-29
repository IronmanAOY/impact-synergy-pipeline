import os
import glob
import logging
import warnings
import re

import numpy as np
import pandas as pd
from bids import BIDSLayout
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from nilearn.signal import clean
import nibabel as nib
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_atlas_aal

log = logging.getLogger(__name__)

# Download once and save into atlases/s
atlas_sch = fetch_atlas_schaefer_2018(
    n_rois=400,
    yeo_networks=7,
    data_dir='atlases',
    resume=True
)
atlas_aal = fetch_atlas_aal(
    data_dir='atlases',
    resume=True
)

# point to your manually downloaded Shen-268 map:
atlas_shen_map = os.path.abspath('atlases/shen_1mm_268_parcellation.nii.gz')

ATLAS_GLOBS = {
    "schaefer400": atlas_sch["maps"],
    "aal90":       atlas_aal.maps,
    "shen268":     atlas_shen_map,
}


def find_atlas(key):
    try:
        return ATLAS_GLOBS[key]
    except KeyError:
        raise FileNotFoundError(f"No atlas entry for {key}")

def collect_confounds(bids_root, fmriprep_deriv, subj, run_no, task):
    func_dir = os.path.join(fmriprep_deriv, f"sub-{subj}", "func")
    pattern = os.path.join(
        func_dir,
        f"sub-{subj}_task-{task}_run-*_desc-confounds*timeseries.tsv"
    )
    candidates = glob.glob(pattern)
    if not candidates:
        warnings.warn(f"No confounds files found at {pattern}")
        return None, 0.0

    # filter to the one whose run-index == run_no
    sel = []
    for f in candidates:
        m = re.search(r"_run-0*(\d+)_desc", os.path.basename(f))
        if m and int(m.group(1)) == run_no:
            sel.append(f)

    if len(sel) > 1:
        warnings.warn(f"Multiple confound files for sub-{subj} run-{run_no}: {sel}\n  Picking the first.")
    if not sel:
        warnings.warn(f"No confounds TSV for sub-{subj} task-{task} run-{run_no}")
        return None, 0.0

    conf_file = sel[0]
    df = pd.read_csv(conf_file, sep="\t")

    compcor_cols = [c for c in df.columns if "compcor" in c.lower()]
    compcor = df[compcor_cols].values if compcor_cols else None
    fd = df["framewise_displacement"].mean() if "framewise_displacement" in df.columns else 0.0
    return compcor, fd


def preprocess_subject(bids_root, fmriprep_deriv, subj, bf, out_dir):
    """
    Preprocess one bold run (bf is a BIDSLayoutFile object).
    Outputs cleaned NIfTI, mean_fd.txt, and ROI‐TS .npy files.
    """
    # 1) Bold path & run number
    bold_path = bf.path
    run_no    = int(bf.entities.get("run", 0))

    # 2) load confounds
    task   = bf.entities['task']
    run_no = int(bf.entities['run'])
    compcor, fd = collect_confounds(
        bids_root=bids_root,
        fmriprep_deriv=fmriprep_deriv,
        subj=subj,
        run_no=run_no,
        task=task
    )

    # 3) load & clean BOLD
    img = image.load_img(bold_path)
    tr  = img.header.get_zooms()[-1]
    if tr > 10:
        log.warning(f"Suspicious TR={tr}, using 2.0s")
        tr = 2.0

    data = img.get_fdata().reshape(-1, img.shape[-1]).T
    cleaned = clean(
        signals=data,
        confounds=compcor,
        t_r=tr,
        detrend=True,
        standardize=True,
        low_pass=0.1,
        high_pass=0.01,
    ).T
    cleaned_vol = cleaned.reshape(img.shape)

    # 4) save cleaned NIfTI
    os.makedirs(out_dir, exist_ok=True)
    clean_fn = os.path.join(out_dir, f"{subj}_run-{run_no}_cleaned_bold.nii.gz")
    try:
        image.new_img_like(img, cleaned_vol).to_filename(clean_fn)
    except TypeError:
        nib.Nifti1Image(cleaned_vol, img.affine).to_filename(clean_fn)

    # 5) write mean FD
    with open(os.path.join(out_dir, "mean_fd.txt"), "w") as fp:
        fp.write(f"{fd:.6f}")

    # 6) eextract ROI time-series for each atlas
    for key, atlas_img in ATLAS_GLOBS.items():
        masker = NiftiLabelsMasker(labels_img=atlas_img,
                                   standardize=True,
                                   t_r=tr)
        ts = masker.fit_transform(clean_fn)

        np.save(os.path.join(out_dir, f"{subj}_run-{run_no}_{key}_ts.npy"), ts)

def run_preprocessing(bids_root, fmriprep_deriv, out_root):
    layout = BIDSLayout(bids_root, validate=False)

    # find all raw BOLD runs
    subjects = sorted({f.subject for f in layout.get(suffix='bold',
                                                      extension='nii.gz')})
    for subj in subjects:
        bold_files = layout.get(
            subject=subj,
            suffix='bold',
            extension='nii.gz',
            return_type='object'
        )
        for bf in bold_files:
            task = bf.entities.get('task')
            # only keep resting runs
            if task == 'restawake':
                cond = 'awake'
            elif task == 'restdeep':
                cond = 'deep'
            elif task == 'restrecovery':
                cond = 'recovery'
            else:
                # skip audio or other tasks
                continue

            out_dir = os.path.join(out_root, subj, cond)
            preprocess_subject(
                bids_root,
                fmriprep_deriv,
                subj,
                bf,
                out_dir
            )
