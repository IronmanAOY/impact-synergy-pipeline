from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


REAL_DATA_ORIGIN = "real"
DUMMY_DATA_ORIGIN = "dummy"
VALID_DATA_ORIGINS = (REAL_DATA_ORIGIN, DUMMY_DATA_ORIGIN)
PROVENANCE_COLUMNS = (
    "dataset_id",
    "data_origin",
    "dataset_role",
    "provenance_label",
)
TEST_OBJECTS_ROOTNAME = "test_objects"


def normalize_data_origin(value) -> str:
    raw = str(value or REAL_DATA_ORIGIN).strip().lower()
    aliases = {
        "real": REAL_DATA_ORIGIN,
        "study": REAL_DATA_ORIGIN,
        "study_data": REAL_DATA_ORIGIN,
        "production": REAL_DATA_ORIGIN,
        "dummy": DUMMY_DATA_ORIGIN,
        "test": DUMMY_DATA_ORIGIN,
        "test_object": DUMMY_DATA_ORIGIN,
        "test_objects": DUMMY_DATA_ORIGIN,
        "synthetic": DUMMY_DATA_ORIGIN,
        "fake": DUMMY_DATA_ORIGIN,
    }
    origin = aliases.get(raw, raw)
    if origin not in VALID_DATA_ORIGINS:
        raise ValueError(
            f"Unknown data origin '{value}'. Expected one of {VALID_DATA_ORIGINS}."
        )
    return origin


def dataset_role_for_origin(data_origin: str) -> str:
    origin = normalize_data_origin(data_origin)
    if origin == DUMMY_DATA_ORIGIN:
        return "synthetic_validation"
    return "study_data"


def provenance_label_for_origin(data_origin: str) -> str:
    origin = normalize_data_origin(data_origin)
    if origin == DUMMY_DATA_ORIGIN:
        return "synthetic_validation"
    return "real_study_data"


def is_test_object_origin(data_origin: str) -> bool:
    return normalize_data_origin(data_origin) == DUMMY_DATA_ORIGIN


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


def ensure_test_objects_scaffold(repo_root: Path | str) -> dict[str, Path]:
    root = Path(repo_root).resolve() / TEST_OBJECTS_ROOTNAME
    paths = {
        "root": root,
        "datasets": root / "datasets",
        "runs": root / "runs",
        "metric_bank": root / "metric_bank",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


@dataclass(frozen=True)
class DatasetProvenance:
    repo_root: Path
    requested_out_dir: Path
    effective_out_dir: Path
    dataset_id: str
    data_origin: str
    dataset_role: str
    provenance_label: str
    test_objects_root: Path
    metric_bank_dataset_dir: Path | None

    def as_result_metadata(self) -> dict[str, str]:
        return {
            "dataset_id": str(self.dataset_id),
            "data_origin": str(self.data_origin),
            "dataset_role": str(self.dataset_role),
            "provenance_label": str(self.provenance_label),
        }

    def as_manifest_dict(self) -> dict[str, object]:
        out = {
            "dataset_id": str(self.dataset_id),
            "data_origin": str(self.data_origin),
            "dataset_role": str(self.dataset_role),
            "provenance_label": str(self.provenance_label),
            "requested_out_dir": str(self.requested_out_dir),
            "effective_out_dir": str(self.effective_out_dir),
            "test_objects_root": str(self.test_objects_root),
            "created_unix": float(time.time()),
        }
        if self.metric_bank_dataset_dir is not None:
            out["metric_bank_dataset_dir"] = str(self.metric_bank_dataset_dir)
        return out


def resolve_dataset_provenance(
    *,
    repo_root: Path | str,
    out_dir: Path | str,
    dataset_id: str,
    data_origin=REAL_DATA_ORIGIN,
) -> DatasetProvenance:
    root = Path(repo_root).resolve()
    requested = Path(out_dir).expanduser().resolve()
    dataset_id_txt = str(dataset_id).strip()
    origin = normalize_data_origin(data_origin)
    role = dataset_role_for_origin(origin)
    label = provenance_label_for_origin(origin)
    scaffold = ensure_test_objects_scaffold(root)

    if origin == DUMMY_DATA_ORIGIN:
        base = requested
        if not _is_relative_to(base, scaffold["root"]):
            base = scaffold["runs"]
        effective = base
        if effective.name != dataset_id_txt:
            effective = effective / dataset_id_txt
        metric_bank_dataset_dir = scaffold["metric_bank"] / dataset_id_txt
    else:
        effective = requested
        if dataset_id_txt != "ds003171" and effective.name != dataset_id_txt:
            effective = effective / dataset_id_txt
        metric_bank_dataset_dir = None

    return DatasetProvenance(
        repo_root=root,
        requested_out_dir=requested,
        effective_out_dir=effective.resolve(),
        dataset_id=dataset_id_txt,
        data_origin=origin,
        dataset_role=role,
        provenance_label=label,
        test_objects_root=scaffold["root"],
        metric_bank_dataset_dir=metric_bank_dataset_dir,
    )


def write_json(path: Path | str, payload) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
