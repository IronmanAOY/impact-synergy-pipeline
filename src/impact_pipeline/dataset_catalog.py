from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


PIPELINE_ENABLED_DATASET_IDS = frozenset({"ds003171", "ds005620"})


@dataclass(frozen=True)
class ReportDataset:
    dataset_id: str
    title: str
    modality: str
    report_role: str
    target_metrics: tuple[str, ...]
    source_url: str
    mirror_git_url: str
    snapshot: str
    doi: str
    license: str
    local_root_candidates: tuple[str, ...]
    fetch_strategy: str
    pipeline_ready: bool
    task_hints: tuple[str, ...] = ()
    representative_payload_paths: tuple[str, ...] = ()
    notes: str = ""
    license_note: str = ""
    report_priority: int = 0

    def to_record(self) -> dict[str, object]:
        return asdict(self)


REPORT_DATASETS: dict[str, ReportDataset] = {
    "ds006623": ReportDataset(
        dataset_id="ds006623",
        title="Michigan Human Anesthesia fMRI Dataset-1",
        modality="fmri",
        report_role="state_switch_anchor_fmri",
        target_metrics=("PDI", "NAS", "IIM"),
        source_url="https://openneuro.org/datasets/ds006623",
        mirror_git_url="https://github.com/OpenNeuroDatasets/ds006623.git",
        snapshot="1.0.0",
        doi="10.18112/openneuro.ds006623.v1.0.0",
        license="CC0",
        local_root_candidates=("data/scratch/ds006623", "data/ds006623"),
        fetch_strategy="git_mirror",
        pipeline_ready=False,
        task_hints=("imagery", "rest"),
        representative_payload_paths=(
            "derivatives/fmriprep_output/sub-02/func/"
            "sub-02_task-imagery_run-1_space-MNI152NLin2009cAsym_res-04_desc-preproc_bold.nii.gz",
        ),
        notes=(
            "Primary fMRI state-switch anchor from the report. Strong for awake-vs-sedated "
            "PDI/NAS/IIM analyses, but not a full CI dataset on its own."
        ),
        report_priority=1,
    ),
    "ds005620": ReportDataset(
        dataset_id="ds005620",
        title="A repeated awakening study exploring the capacity of complexity measures to capture dreaming during propofol sedation",
        modality="eeg",
        report_role="state_switch_anchor_eeg",
        target_metrics=("PDI", "NAS", "IIM"),
        source_url="https://openneuro.org/datasets/ds005620",
        mirror_git_url="https://github.com/OpenNeuroDatasets/ds005620.git",
        snapshot="1.0.0",
        doi="10.18112/openneuro.ds005620.v1.0.0",
        license="CC0",
        local_root_candidates=(
            "data/scratch/ds005620_annex",
            "data/scratch/ds005620",
            "data/ds005620",
        ),
        fetch_strategy="full_local",
        pipeline_ready=True,
        task_hints=("awake", "sed", "sed2"),
        representative_payload_paths=(
            "sub-1010/eeg/sub-1010_task-awake_acq-EC_eeg.vhdr",
            "sub-1010/eeg/sub-1010_task-awake_acq-EC_eeg.eeg",
            "sub-1010/eeg/sub-1010_task-awake_acq-EC_eeg.vmrk",
        ),
        notes=(
            "Primary EEG state-switch anchor from the report and one of the current pipeline-enabled datasets."
        ),
        license_note=(
            "The report flagged a license inconsistency between README metadata and dataset_description; "
            "treat citation conservatively and cite the dataset DOI plus original authors."
        ),
        report_priority=2,
    ),
    "ds004295": ReportDataset(
        dataset_id="ds004295",
        title="Reward gain and punishment avoidance reversal learning",
        modality="eeg",
        report_role="ram_feedback_module_eeg",
        target_metrics=("RAM",),
        source_url="https://openneuro.org/datasets/ds004295",
        mirror_git_url="https://github.com/OpenNeuroDatasets/ds004295.git",
        snapshot="1.0.0",
        doi="10.18112/openneuro.ds004295.v1.0.0",
        license="CC0",
        local_root_candidates=("data/scratch/ds004295", "data/ds004295"),
        fetch_strategy="git_mirror",
        pipeline_ready=False,
        task_hints=("task",),
        representative_payload_paths=(
            "sub-s1/eeg/sub-s1_task-task_eeg.set",
            "sub-s1/eeg/sub-s1_task-task_eeg.fdt",
        ),
        notes=(
            "Reward/feedback EEG module prioritized in the report for RAM-oriented estimation."
        ),
        report_priority=3,
    ),
    "ds002547": ReportDataset(
        dataset_id="ds002547",
        title="SharedStates",
        modality="fmri",
        report_role="srpi_self_other_module_fmri",
        target_metrics=("SRPI",),
        source_url="https://openneuro.org/datasets/ds002547",
        mirror_git_url="https://github.com/OpenNeuroDatasets/ds002547.git",
        snapshot="1.1.0",
        doi="10.18112/openneuro.ds002547.v1.1.0",
        license="CC0",
        local_root_candidates=("data/scratch/ds002547", "data/ds002547"),
        fetch_strategy="git_mirror",
        pipeline_ready=False,
        task_hints=("self", "other"),
        representative_payload_paths=(
            "derivatives/fmriprep/sub-01/ses-1/func/"
            "sub-01_ses-1_task-other_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        ),
        notes=(
            "Explicit self-versus-other fMRI dataset used by the report as the cleanest SRPI module."
        ),
        report_priority=4,
    ),
    "ds002685": ReportDataset(
        dataset_id="ds002685",
        title="IBC",
        modality="fmri",
        report_role="awake_reference_multitask_fmri",
        target_metrics=("RAM", "SRPI"),
        source_url="https://openneuro.org/datasets/ds002685",
        mirror_git_url="https://github.com/OpenNeuroDatasets/ds002685.git",
        snapshot="1.3.1",
        doi="10.18112/openneuro.ds002685.v1.3.1",
        license="CC0",
        local_root_candidates=("data/scratch/ds002685", "data/ds002685"),
        fetch_strategy="git_mirror",
        pipeline_ready=False,
        task_hints=("Self", "HcpGambling", "Discount", "PreferenceFaces", "PreferenceFood"),
        representative_payload_paths=(
            "sub-01/ses-00/func/sub-01_ses-00_task-ArchiSocial_dir-ap_bold.nii.gz",
        ),
        notes=(
            "Dense awake-only reference dataset with self and reward tasks. Useful for human normalization "
            "and robustness checks, not for direct awake-vs-deep CI estimation."
        ),
        report_priority=5,
    ),
    "ds005479": ReportDataset(
        dataset_id="ds005479",
        title="Monetary Incentive Delay task - structural and functional images of 37 men; study of associations between circadian characteristics (eveningness, distinctness) and affective processing",
        modality="fmri",
        report_role="ram_reward_module_fmri",
        target_metrics=("RAM",),
        source_url="https://openneuro.org/datasets/ds005479",
        mirror_git_url="https://github.com/OpenNeuroDatasets/ds005479.git",
        snapshot="1.1.1",
        doi="10.18112/openneuro.ds005479.v1.1.1",
        license="CC0",
        local_root_candidates=("data/scratch/ds005479", "data/ds005479"),
        fetch_strategy="git_mirror",
        pipeline_ready=False,
        task_hints=("MID",),
        representative_payload_paths=("sub-01/func/sub-01_task-MID_bold.nii.gz",),
        notes=(
            "Optional fMRI reward module from the report for RAM/reward analyses, with narrower population coverage."
        ),
        report_priority=6,
    ),
    "ds002336": ReportDataset(
        dataset_id="ds002336",
        title="A multi-modal human neuroimaging dataset for data integration: simultaneous EEG and fMRI acquisition during a motor imagery neurofeedback task: XP1",
        modality="eeg_fmri",
        report_role="multimodal_method_development",
        target_metrics=("RAM", "NAS"),
        source_url="https://openneuro.org/datasets/ds002336",
        mirror_git_url="https://github.com/OpenNeuroDatasets/ds002336.git",
        snapshot="2.0.2",
        doi="10.18112/openneuro.ds002336.v2.0.2",
        license="CC0",
        local_root_candidates=("data/scratch/ds002336", "data/ds002336"),
        fetch_strategy="git_mirror",
        pipeline_ready=False,
        task_hints=("motorloc", "eegNF", "fmriNF", "eegfmriNF"),
        representative_payload_paths=(
            "sub-xp101/eeg/sub-xp101_task-eegNF_eeg.vhdr",
            "sub-xp101/eeg/sub-xp101_task-eegNF_eeg.eeg",
            "sub-xp101/eeg/sub-xp101_task-eegNF_eeg.vmrk",
        ),
        notes=(
            "Simultaneous EEG-fMRI neurofeedback dataset retained for multimodal method development rather than CI completion."
        ),
        report_priority=7,
    ),
    "ds003171": ReportDataset(
        dataset_id="ds003171",
        title="Modeling an auditory stimulated brain under altered states of consciousness using the generalized ising model",
        modality="fmri",
        report_role="legacy_state_switch_anchor_fmri",
        target_metrics=("PDI", "NAS", "IIM"),
        source_url="https://openneuro.org/datasets/ds003171",
        mirror_git_url="https://github.com/OpenNeuroDatasets/ds003171.git",
        snapshot="2.0.1",
        doi="10.18112/openneuro.ds003171.v2.0.1",
        license="CC0",
        local_root_candidates=("data/scratch/ds003171", "data/ds003171"),
        fetch_strategy="full_local",
        pipeline_ready=True,
        task_hints=("audioawake", "audiodeep", "restawake", "restdeep"),
        representative_payload_paths=(
            "sub-2525JK/func/sub-2525JK_task-audioawake_run-01_bold.nii.gz",
        ),
        notes=(
            "fMRI replication anchor in the repository. Valuable for PDI/NAS/IIM, but RAM and SRPI remain undefined."
        ),
        report_priority=8,
    ),
}


def get_report_dataset(dataset_id: str | None) -> ReportDataset | None:
    ds = str(dataset_id or "").strip()
    if not ds:
        return None
    return REPORT_DATASETS.get(ds)


def iter_report_datasets() -> tuple[ReportDataset, ...]:
    return tuple(
        sorted(
            REPORT_DATASETS.values(),
            key=lambda item: (int(item.report_priority), str(item.dataset_id)),
        )
    )


def dataset_root_candidates(dataset_id: str, repo_root: Path | str) -> tuple[Path, ...]:
    entry = get_report_dataset(dataset_id)
    if entry is None:
        return tuple()
    root = Path(repo_root).resolve()
    return tuple(root / rel for rel in entry.local_root_candidates)


def resolve_local_dataset_root(dataset_id: str, repo_root: Path | str) -> Path | None:
    for candidate in dataset_root_candidates(dataset_id, repo_root):
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_representative_payload_paths(
    dataset_id: str,
    repo_root: Path | str,
) -> tuple[Path, ...]:
    entry = get_report_dataset(dataset_id)
    local_root = resolve_local_dataset_root(dataset_id, repo_root)
    if entry is None or local_root is None:
        return tuple()
    return tuple(local_root / rel for rel in entry.representative_payload_paths)


def is_pipeline_enabled_dataset(dataset_id: str | None) -> bool:
    ds = str(dataset_id or "").strip()
    return ds in PIPELINE_ENABLED_DATASET_IDS


def _safe_load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _task_examples(root: Path, limit: int = 12) -> list[str]:
    tasks: set[str] = set()
    patterns = (
        "sub-*/func/*_task-*_bold.json",
        "sub-*/*/func/*_task-*_bold.json",
        "sub-*/*_task-*_bold.json",
        "sub-*/eeg/*_task-*_eeg.*",
        "sub-*/*/eeg/*_task-*_eeg.*",
        "sub-*/func/*_task-*_events.tsv",
        "sub-*/*/func/*_task-*_events.tsv",
    )
    for pattern in patterns:
        for path in root.glob(pattern):
            match = re.search(r"_task-([^_]+)_", path.name)
            if match:
                tasks.add(match.group(1))
                if len(tasks) >= limit:
                    return sorted(tasks)
    return sorted(tasks)


def _count_subject_dirs(root: Path) -> int:
    return sum(1 for p in root.glob("sub-*") if p.is_dir())


def _directory_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def _serialize_local_root(local_root: Path | None, repo_root: Path) -> str | None:
    if local_root is None:
        return None
    resolved = local_root.resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(resolved)


def build_inventory(
    repo_root: Path,
    snapshot_status: dict[str, dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    snapshot_status = snapshot_status or {}
    rows: list[dict[str, object]] = []
    for entry in iter_report_datasets():
        local_root = resolve_local_dataset_root(entry.dataset_id, repo_root)
        row = entry.to_record()
        row["local_root"] = _serialize_local_root(local_root, repo_root)
        row["local_status"] = "present" if local_root is not None else "missing"
        row["local_bytes"] = 0
        row["local_gib"] = 0.0
        row["subject_dirs"] = 0
        row["task_examples"] = list(entry.task_hints)
        row["has_derivatives"] = False
        row["has_git_repo"] = False
        row["has_annex_metadata"] = False
        row["annex_objects_root"] = None
        row["annex_objects_externalized"] = False
        row["representative_payload_paths"] = list(entry.representative_payload_paths)
        row["representative_payload_present"] = []
        row["representative_payload_missing"] = list(entry.representative_payload_paths)
        row["representative_payload_present_count"] = 0
        row["representative_payload_complete"] = False
        row["snapshot_status_known"] = False
        row["snapshot_mode"] = None
        row["snapshot_total_files"] = None
        row["snapshot_total_bytes"] = None
        row["snapshot_total_gib"] = None
        row["snapshot_missing_files"] = None
        row["snapshot_missing_bytes"] = None
        row["snapshot_missing_gib"] = None
        row["snapshot_complete"] = None
        row["dataset_description_name"] = None
        row["dataset_description_license"] = None
        row["dataset_description_doi"] = None
        cached_status = snapshot_status.get(entry.dataset_id)
        if cached_status is not None:
            row["snapshot_status_known"] = True
            row["snapshot_mode"] = cached_status.get("mode")
            row["snapshot_total_files"] = cached_status.get("snapshot_total_files")
            row["snapshot_total_bytes"] = cached_status.get("snapshot_total_bytes")
            row["snapshot_total_gib"] = cached_status.get("snapshot_total_gib")
            row["snapshot_missing_files"] = cached_status.get("snapshot_missing_files")
            row["snapshot_missing_bytes"] = cached_status.get("snapshot_missing_bytes")
            row["snapshot_missing_gib"] = cached_status.get("snapshot_missing_gib")
            row["snapshot_complete"] = cached_status.get("snapshot_complete")
        if local_root is not None:
            desc = _safe_load_json(local_root / "dataset_description.json")
            n_bytes = _directory_bytes(local_root)
            row["local_bytes"] = int(n_bytes)
            row["local_gib"] = round(float(n_bytes) / (1024.0**3), 3)
            row["subject_dirs"] = int(_count_subject_dirs(local_root))
            row["task_examples"] = _task_examples(local_root) or list(entry.task_hints)
            row["has_derivatives"] = bool((local_root / "derivatives").exists())
            row["has_git_repo"] = bool((local_root / ".git").exists())
            row["has_annex_metadata"] = bool(
                (local_root / ".datalad").exists()
                or (local_root / ".git" / "annex").exists()
            )
            annex_objects = local_root / ".git" / "annex" / "objects"
            if annex_objects.exists() or annex_objects.is_symlink():
                row["annex_objects_root"] = str(annex_objects.resolve(strict=False))
                row["annex_objects_externalized"] = annex_objects.is_symlink()
            present = [
                relpath
                for relpath in entry.representative_payload_paths
                if (local_root / relpath).exists()
            ]
            missing = [
                relpath
                for relpath in entry.representative_payload_paths
                if not (local_root / relpath).exists()
            ]
            row["representative_payload_present"] = present
            row["representative_payload_missing"] = missing
            row["representative_payload_present_count"] = len(present)
            row["representative_payload_complete"] = not missing
            row["dataset_description_name"] = desc.get("Name")
            row["dataset_description_license"] = desc.get("License")
            row["dataset_description_doi"] = desc.get("DatasetDOI")
        rows.append(row)
    return rows
