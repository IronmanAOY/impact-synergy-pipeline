import os
import glob
import logging
import warnings
import re
from pathlib import Path

import numpy as np
import pandas as pd
from bids import BIDSLayout
from nilearn import image
from nilearn.maskers import NiftiLabelsMasker
from nilearn.signal import clean
import nibabel as nib

log = logging.getLogger(__name__)

ATLAS_GLOBS = None


def get_atlas_globs():
    """
    Resolve atlas resources from local files only.
    This keeps preprocessing fully offline and deterministic.
    """
    global ATLAS_GLOBS
    if ATLAS_GLOBS is not None:
        return ATLAS_GLOBS

    atlas_root = Path("atlases")
    atlas_sch = atlas_root / "schaefer_2018" / "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
    atlas_aal = atlas_root / "aal_SPM12" / "aal" / "atlas" / "AAL.nii"
    atlas_shen = atlas_root / "shen_1mm_268_parcellation.nii.gz"

    missing = [str(p) for p in (atlas_sch, atlas_aal, atlas_shen) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required local atlas files. Expected:\n"
            + "\n".join(missing)
            + "\nPopulate atlases/ before running fMRI preprocessing."
        )

    ATLAS_GLOBS = {
        "schaefer400": str(atlas_sch.resolve()),
        "aal90": str(atlas_aal.resolve()),
        "shen268": str(atlas_shen.resolve()),
    }
    return ATLAS_GLOBS


def find_atlas(key):
    atlas_globs = get_atlas_globs()
    try:
        return atlas_globs[key]
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


def _find_preproc_bold(fmriprep_deriv, subj, run_no, task):
    """
    Resolve the fMRIPrep preprocessed BOLD file for a specific subject/task/run.

    Preference order:
      1) MNI152NLin2009cAsym space desc-preproc file for exact run number
      2) any desc-preproc file for exact run number
    """
    func_dir = os.path.join(fmriprep_deriv, f"sub-{subj}", "func")
    pattern = os.path.join(
        func_dir,
        f"sub-{subj}_task-{task}_run-*_desc-preproc_bold.nii.gz",
    )
    candidates = glob.glob(pattern)
    if not candidates:
        return None

    selected = []
    for f in candidates:
        m = re.search(r"_run-0*(\d+)_", os.path.basename(f))
        if m and int(m.group(1)) == int(run_no):
            selected.append(f)

    if not selected:
        return None

    selected = sorted(
        selected,
        key=lambda p: (
            0 if "space-MNI152NLin2009cAsym" in os.path.basename(p) else 1,
            p,
        ),
    )
    return selected[0]


def preprocess_subject(bids_root, fmriprep_deriv, subj, bf, out_dir):
    """
    Preprocess one bold run (bf is a BIDSLayoutFile object).
    Outputs cleaned NIfTI, mean_fd.txt, and ROI‐TS .npy files.
    """
    # 1) Resolve preprocessed BOLD from fMRIPrep derivatives for this run.
    run_no = int(bf.entities.get("run", 0))

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
    preproc_bold = _find_preproc_bold(
        fmriprep_deriv=fmriprep_deriv,
        subj=subj,
        run_no=run_no,
        task=task,
    )
    if preproc_bold is None:
        raise FileNotFoundError(
            f"No fMRIPrep desc-preproc BOLD found for sub-{subj} task-{task} run-{run_no} "
            f"under {os.path.join(fmriprep_deriv, f'sub-{subj}', 'func')}"
        )

    img = image.load_img(preproc_bold)
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
    for key, atlas_img in get_atlas_globs().items():
        masker = NiftiLabelsMasker(labels_img=atlas_img,
                                   standardize=True,
                                   t_r=tr)
        ts = masker.fit_transform(clean_fn)

        np.save(os.path.join(out_dir, f"{subj}_run-{run_no}_{key}_ts.npy"), ts)

def run_preprocessing(bids_root, fmriprep_deriv, out_root, subjects=None):
    layout = BIDSLayout(bids_root, validate=False)
    # find all subjects in the dataset
    all_subj = sorted(
        {
            str(f.entities.get("subject"))
            for f in layout.get(suffix='bold', extension='nii.gz')
            if f.entities.get("subject") is not None
        }
    )
    if subjects:
        # only keep the ones specified on the CLI
        subjects = [s for s in all_subj if s in subjects]
    else:
        subjects = all_subj
    for subj in subjects:
        bold_files = layout.get(
            subject=subj,
            suffix='bold',
            extension='nii.gz',
            return_type='object'
        )
        for bf in bold_files:
            task = bf.entities.get('task')
            # keep both audio and rest runs, grouping by sedation phase
            # sedation level:
            if task.endswith('awake'):
                sed = 'awake'
            elif task.endswith('deep'):
                sed = 'deep'
            elif task.endswith('recovery'):
                sed = 'recovery'
            else:
                continue
        
            # condition:
            if task.startswith('rest'):
                cond = 'rest'
            elif task.startswith('audio'):
                cond = 'audio'
            else:
                continue

            out_dir = os.path.join(out_root, subj, sed, cond)
            preprocess_subject(
                bids_root,
                fmriprep_deriv,
                subj,
                bf,
                out_dir
            )
