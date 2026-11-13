"""Frozen settings and source identities; data selection never uses test scores."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .common import atomic_json, digest_json, read_json, sha256_file

SAM_COMMIT = "dca509fe793f601edb92606367a655c15ac00fdf"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"


@dataclass(frozen=True)
class Settings:
    dataset_root: str = "E:/Project/Database/MS-COCO"
    split_seed: int = 20261026
    calibration_images: int = 1000
    expected_images: int = 5000
    pilot_images: int = 100
    thresholds: tuple = tuple(round(i * 0.05, 2) for i in range(1, 20))
    budgets: tuple = (10, 20, 40)
    primary_budget: int = 20
    models: tuple = ("yolov8n", "yolo11n", "yolo11s")
    seeds: tuple = (0, 1, 2, 3, 4)
    fixed_theta: float = 0.25
    sam_model: str = "vit_b"
    batch_size: int = 16
    device: str = "auto"
    bootstrap_reps: int = 2000
    bootstrap_seed: int = 20261026
    compute_mask_ap: bool = True
    compute_ap_model_selection: bool = True

    def validate(self):
        if not 0 < self.calibration_images < self.expected_images:
            raise ValueError("Calibration must be a nonempty proper image subset.")
        if len(set(self.thresholds)) != len(self.thresholds) or not all(0 < t <= 1 for t in self.thresholds):
            raise ValueError("Thresholds must be unique and in (0,1].")
        if self.fixed_theta not in self.thresholds or self.primary_budget not in self.budgets:
            raise ValueError("Fixed threshold and primary budget must be in their grids.")
        if not self.budgets or any(type(b) is not int or b <= 0 for b in self.budgets):
            raise ValueError("Prompt budgets must be positive integers.")
        if self.sam_model != "vit_b" or self.batch_size < 1:
            raise ValueError("Protocol v1 fixes SAM ViT-B and a positive decoding batch size.")
        if self.bootstrap_reps < 1 or self.pilot_images > self.calibration_images:
            raise ValueError("Invalid bootstrap count or pilot size.")


def create_split(image_ids, seed, n_calibration):
    ids = sorted(int(x) for x in image_ids)
    if len(ids) != len(set(ids)) or not 0 < n_calibration < len(ids):
        raise ValueError("Split needs unique IDs and nonempty calibration/evaluation subsets.")
    ranked = sorted(ids, key=lambda i: (hashlib.sha256(f"{seed}:{i}".encode()).hexdigest(), i))
    return {"split_seed": seed, "calibration_ids": sorted(ranked[:n_calibration]),
            "evaluation_ids": sorted(ranked[n_calibration:])}


def discover_sources(project_root, settings):
    matrix = Path(project_root) / "results/detection/matrix_runs.csv"
    with matrix.open(encoding="utf-8-sig", newline="") as stream:
        selected = [r for r in csv.DictReader(stream) if r["dataset"] == "mscoco"
                    and r["model"] in settings.models and int(r["seed"]) in settings.seeds]
    expected = {(m, s) for m in settings.models for s in settings.seeds}
    if len(selected) != len(expected) or {(r["model"], int(r["seed"])) for r in selected} != expected:
        raise ValueError("Source registry does not contain exactly the requested COCO matrix.")
    sources = []
    for row in sorted(selected, key=lambda r: (settings.models.index(r["model"]), int(r["seed"]))):
        directory = Path(row["output"]).resolve()
        completed = read_json(directory / "run_complete.json")
        cached = read_json(directory / "cache_complete.json")
        if completed.get("status") != "complete" or cached.get("status") != "complete":
            raise ValueError(f"Incomplete source: {directory}")
        if completed["run_fingerprint"] != cached["run_fingerprint"]:
            raise ValueError(f"Source manifest disagreement: {directory}")
        if completed.get("n_images") != settings.expected_images:
            raise ValueError(f"Unexpected source image count: {directory}")
        prediction_file = directory / "filtered_predictions.json"
        if not prediction_file.is_file():
            raise FileNotFoundError(prediction_file)
        sources.append({"run_id": f"{row['model']}_seed{row['seed']}", "model": row["model"],
                        "seed": int(row["seed"]), "prediction_file": str(prediction_file),
                        "source_fingerprint": completed["run_fingerprint"],
                        "declared_sha256": cached.get("filtered_predictions_sha256")})
    return sources


def freeze_protocol(root, output, settings, sources, annotation, *, mode, device_info):
    root, output = Path(root), Path(output)
    project = root.parents[1]
    source_files = sorted((root / "downstream").glob("*.py")) + sorted((project / "region_eval").glob("*.py"))
    code_hashes = {str(p.relative_to(project)): sha256_file(p) for p in source_files}
    record = {"protocol_version": "downstream-sam-1.0.0", "mode": mode,
              "settings": asdict(settings), "sources": sources,
              "annotation_sha256": sha256_file(annotation),
              "sam_commit": SAM_COMMIT,
              "sam_checkpoint_sha256": sha256_file(root / "weights/sam_vit_b_01ec64.pth"),
              "code_sha256": code_hashes, "runtime": device_info,
              "endpoint": "class-union foreground mIoU on held-out images",
              "ci_scope": "paired image bootstrap, conditional on fixed calibration selections",
              "source_prediction_policy": "existing upstream COCO crowd-filtered post-NMS caches"}
    # Normalize tuple settings before comparing an in-memory record with saved JSON lists.
    record = json.loads(json.dumps(record, allow_nan=False))
    record["fingerprint"] = digest_json(record)
    path = output / "protocol.json"
    if path.exists():
        previous = read_json(path)
        if previous != record:
            raise ValueError("Protocol/data/code/runtime changed. Keep the old output and choose a NEW run name.")
    else:
        atomic_json(path, record)
    return record
