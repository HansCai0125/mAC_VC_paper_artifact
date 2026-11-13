"""Validated registry of the saved prediction caches used by VC analyses.

The paper has two deliberately different source collections:

* ``paired_yolo_sources`` contains the three YOLO architectures, five seeds,
  and both datasets (30 runs).  It is the correct source for paired
  cross-dataset stress tests.
* ``main_natural_sources`` adds the five VisDrone RT-DETR-L runs (35 runs).
  It is the correct source for the natural collective-coverage analysis.

No function in this module trains a model or performs inference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path

from .offline import PROJECT_ROOT


REGISTRY_VERSION = "vc-v1-cache-registry-0.1.0"
MATRIX_ROOT = PROJECT_ROOT / "results" / "detection"
RTDETR_ROOT = PROJECT_ROOT / "results" / "rtdetr"

YOLO_MODELS = ("yolov8n", "yolo11n", "yolo11s")
MODEL_ORDER = (*YOLO_MODELS, "rtdetr_l")
DATASETS = ("mscoco", "visdrone")
SEEDS = tuple(range(5))


@dataclass(frozen=True)
class CachedRun:
    dataset: str
    model: str
    seed: int
    model_id: str
    run_directory: Path
    prediction_file: Path
    analysis_directory: Path
    run_fingerprint: str
    cache_sha256: str
    n_images: int
    source_suite: str

    def record(self) -> dict:
        row = asdict(self)
        for key in ("run_directory", "prediction_file", "analysis_directory"):
            row[key] = str(row[key])
        return row


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _normal_dataset(value: str) -> str:
    value = str(value).strip().lower()
    aliases = {
        "mscoco": "mscoco",
        "mscoco2017": "mscoco",
        "ms-coco": "mscoco",
        "visdrone": "visdrone",
        "visdrone2019-det": "visdrone",
    }
    if value not in aliases:
        raise ValueError(f"unknown dataset label: {value}")
    return aliases[value]


def _validate_run(*, dataset, model, seed, run_directory, source_suite) -> CachedRun:
    run_directory = Path(run_directory).resolve()
    complete_path = run_directory / "run_complete.json"
    cache_path = run_directory / "cache_complete.json"
    prediction_file = run_directory / "filtered_predictions.json"
    analysis_directory = run_directory / "analysis"
    gt_file = analysis_directory / "vc_gt_primary.csv"
    for path in (complete_path, cache_path, prediction_file, gt_file):
        if not path.is_file():
            raise FileNotFoundError(f"incomplete cached run; missing {path}")
    complete = _read_json(complete_path)
    cache = _read_json(cache_path)
    if complete.get("status") != "complete" or cache.get("status") != "complete":
        raise ValueError(f"cached run is not complete: {run_directory}")
    fingerprint = complete.get("run_fingerprint")
    if not fingerprint or cache.get("run_fingerprint") != fingerprint:
        raise ValueError(f"run/cache fingerprint mismatch: {run_directory}")
    actual_dataset = _normal_dataset(complete.get("dataset"))
    if actual_dataset != dataset:
        raise ValueError(
            f"dataset mismatch for {run_directory}: {actual_dataset} != {dataset}"
        )
    actual_seed = complete.get("training_seed", seed)
    if int(actual_seed) != int(seed):
        raise ValueError(f"seed mismatch for {run_directory}")
    n_images = int(complete.get("n_images", cache.get("n_images", 0)))
    if n_images <= 0 or int(cache.get("n_images", n_images)) != n_images:
        raise ValueError(f"invalid image count: {run_directory}")
    return CachedRun(
        dataset=dataset,
        model=model,
        seed=int(seed),
        model_id=str(complete.get("model_id") or f"{model}_seed{seed}"),
        run_directory=run_directory,
        prediction_file=prediction_file,
        analysis_directory=analysis_directory,
        run_fingerprint=fingerprint,
        cache_sha256=str(cache.get("filtered_predictions_sha256", "")),
        n_images=n_images,
        source_suite=source_suite,
    )


def paired_yolo_sources(matrix_root=MATRIX_ROOT) -> list[CachedRun]:
    """Return the complete 3-model x 2-dataset x 5-seed matrix."""
    matrix_root = Path(matrix_root).resolve()
    complete = _read_json(matrix_root / "matrix_complete.json")
    if complete.get("status") != "complete":
        raise ValueError("formal YOLO matrix is not complete")
    if int(complete.get("completed_combinations", 0)) != 30:
        raise ValueError("formal YOLO matrix must contain 30 combinations")
    rows = _read_csv(matrix_root / "matrix_runs.csv")
    lookup = {
        (row["dataset"], row["model"], int(row["seed"])): row
        for row in rows
    }
    expected = {
        (dataset, model, seed)
        for dataset in DATASETS
        for model in YOLO_MODELS
        for seed in SEEDS
    }
    if set(lookup) != expected:
        missing = sorted(expected - set(lookup))
        extra = sorted(set(lookup) - expected)
        raise ValueError(f"formal YOLO matrix changed; missing={missing}, extra={extra}")
    sources = [
        _validate_run(
            dataset=dataset,
            model=model,
            seed=seed,
            run_directory=lookup[(dataset, model, seed)]["output"],
            source_suite="detection",
        )
        for dataset in DATASETS
        for model in YOLO_MODELS
        for seed in SEEDS
    ]
    return sorted(
        sources,
        key=lambda row: (
            DATASETS.index(row.dataset), MODEL_ORDER.index(row.model), row.seed
        ),
    )


def visdrone_rtdetr_sources(root=RTDETR_ROOT) -> list[CachedRun]:
    """Return only the five VisDrone RT-DETR-L seeds used by the paper."""
    root = Path(root).resolve()
    suite = _read_json(root / "suite_complete.json")
    if suite.get("status") != "complete":
        raise ValueError("RT-DETR suite is not complete")
    selected = [
        row for row in suite.get("results", [])
        if _normal_dataset(row.get("dataset")) == "visdrone"
    ]
    if {int(row["training_seed"]) for row in selected} != set(SEEDS):
        raise ValueError("VisDrone RT-DETR suite must contain seeds 0--4")
    sources = [
        _validate_run(
            dataset="visdrone",
            model="rtdetr_l",
            seed=int(row["training_seed"]),
            run_directory=row["output"],
            source_suite="rtdetr",
        )
        for row in selected
    ]
    return sorted(sources, key=lambda row: row.seed)


def main_natural_sources() -> list[CachedRun]:
    """Return 35 main-paper runs: 30 paired YOLO + 5 VisDrone RT-DETR."""
    sources = [*paired_yolo_sources(), *visdrone_rtdetr_sources()]
    identities = {(row.dataset, row.model, row.seed) for row in sources}
    if len(sources) != 35 or len(identities) != 35:
        raise ValueError("natural collective-gain registry must contain 35 runs")
    return sorted(
        sources,
        key=lambda row: (
            DATASETS.index(row.dataset), MODEL_ORDER.index(row.model), row.seed
        ),
    )


def registry_audit(sources: list[CachedRun]) -> dict:
    counts = {}
    for source in sources:
        key = f"{source.dataset}/{source.model}"
        counts[key] = counts.get(key, 0) + 1
    return {
        "registry_version": REGISTRY_VERSION,
        "n_runs": len(sources),
        "n_prediction_bytes": sum(row.prediction_file.stat().st_size for row in sources),
        "groups": dict(sorted(counts.items())),
        "sources": [row.record() for row in sources],
    }
