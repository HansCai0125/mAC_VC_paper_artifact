"""Full-validation, checkpoint-based verification of VC v1 anti-gaming.

This is a separate eight-checkpoint experiment. The three YOLO COCO and four
VisDrone model families overlap the paper's model set, but one archived COCO
YOLO11n checkpoint differs from the paper-selected checkpoint. No result from
this module should silently replace a five-seed or paper-table estimate.
"""

import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from time import sleep
from uuid import uuid4

from .case_analysis import _file_hash
from .gaming import GAMING_VERSION, load_shard_sample, run_gaming_grid, write_gaming_report
from .gaming_formal import (
    DIAGNOSTIC_AP_LEVELS,
    DUPLICATE_FACTORS,
    EXPAND_SCALES,
    SHRINK_SCALES,
)
from .offline import PROJECT_ROOT
from .vc_full import run_dataset_full
from .vc_rtdetr_run import run_rtdetr_dataset_full
from .vc_v1 import PRIMARY_THETA, VC_VERSION


VERSION = "vc-v1-gaming-from-best-0.1.0"
BEST_ROOT = PROJECT_ROOT / "anti_gaming" / "Best"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "vc_v1" / "gaming_from_best_v1"
DATASET_ROOTS = {
    "mscoco": Path(r"E:\Project\Database\MS-COCO"),
    "visdrone": Path(r"E:\Project\Database\VisDrone"),
}
DATASET_FOLDERS = {"mscoco": "MSCOCO", "visdrone": "VisDrone"}
MODEL_FOLDERS = {
    "yolov8n": "yolo8n", "yolo11n": "yolo11n",
    "yolo11s": "yolo11s", "rtdetr_l": "RTDETR",
}
MODEL_LABELS = {
    "yolov8n": "YOLOv8n", "yolo11n": "YOLO11n",
    "yolo11s": "YOLO11s", "rtdetr_l": "RT-DETR-L",
}
DATASET_COUNTS = {"mscoco": 5000, "visdrone": 548}
INFERENCE = dict(imgsz=640, confidence_floor=0.001, nms_iou=0.7, max_det=1000)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fingerprint(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _replace_with_retry(temporary, target):
    """Handle transient Windows locks on result files."""
    for attempt in range(20):
        try:
            os.replace(temporary, target)
            return
        except PermissionError:
            if attempt == 19:
                raise
            sleep(min(0.05 * 2 ** attempt, 2.0))

def _reference_manifest(dataset, model):
    """Return the existing cache used for comparison, not as inference input."""
    if model == "rtdetr_l":
        return (
            PROJECT_ROOT / "outputs" / "vc_v1" / "rtdetr_full_v1" / "runs"
            / dataset / "rtdetr_l" / "seed_0" / "run_manifest.json"
        )
    if model == "yolo11n" and dataset == "visdrone":
        return (
            PROJECT_ROOT / "outputs" / "vc_v1" / "full_seed0_v1"
            / dataset / "run_manifest.json"
        )
    seed = 1 if dataset == "mscoco" and model == "yolo11n" else 0
    return (
        PROJECT_ROOT / "outputs" / "vc_v1" / "formal_matrix_v1" / "runs"
        / dataset / model / f"seed_{seed}" / "run_manifest.json"
    )


def _checkpoint_rows():
    rows = []
    for dataset in DATASET_ROOTS:
        for model in MODEL_FOLDERS:
            weight = BEST_ROOT / DATASET_FOLDERS[dataset] / MODEL_FOLDERS[model] / "best.pt"
            if not weight.is_file():
                raise FileNotFoundError(f"required checkpoint is missing: {weight}")
            reference = _reference_manifest(dataset, model)
            if not reference.is_file():
                raise FileNotFoundError(f"paper comparison manifest is missing: {reference}")
            digest = _file_hash(weight)
            reference_digest = _read_json(reference)["weights_sha256"]
            rows.append(dict(
                dataset=dataset, model=model, model_id=MODEL_LABELS[model] + "_Best",
                weights=str(weight.resolve()), weights_sha256=digest,
                dataset_root=str(DATASET_ROOTS[dataset].resolve()),
                n_images=DATASET_COUNTS[dataset],
                comparison_manifest=str(reference.resolve()),
                comparison_checkpoint_sha256=reference_digest,
                matches_comparison_checkpoint=(digest == reference_digest),
                in_paper_model_set=(dataset == "visdrone" or model != "rtdetr_l"),
            ))
    return rows


def _protocol(rows):
    value = dict(
        version=VERSION, vc_version=VC_VERSION, gaming_version=GAMING_VERSION,
        scope=(
            "eight individual archived checkpoints; separate verification, "
            "not the existing five-seed paper matrix"
        ),
        checkpoints=rows,
        inference=INFERENCE,
        rtdetr_postprocessing="native RT-DETR output; no YOLO NMS",
        primary_theta=PRIMARY_THETA,
        matching_iou=0.5,
        shrink_scales=list(SHRINK_SCALES),
        expand_scales=list(EXPAND_SCALES),
        duplicate_factors=list(DUPLICATE_FACTORS),
        diagnostic_ap_levels=[list(item) for item in DIAGNOSTIC_AP_LEVELS],
        diagnostic_ap_role="sparse shared-cache diagnostic, not official AP",
        source_sha256={
            name: _file_hash(PROJECT_ROOT / name)
            for name in (
                "region_eval/gaming.py", "region_eval/gaming_from_best.py",
                "region_eval/gaming_formal.py", "region_eval/vc_v1.py",
            )
        },
        training=False,
        fresh_model_inference=True,
    )
    value["protocol_fingerprint"] = _fingerprint(value)
    return value


def preflight(*, output=DEFAULT_OUTPUT):
    """Check checkpoint identity and dataset layout without loading a model."""
    output = Path(output).resolve()
    for dataset, root in DATASET_ROOTS.items():
        if not root.is_dir():
            raise FileNotFoundError(f"{dataset} dataset root is missing: {root}")
        if dataset == "mscoco":
            required = (root / "images" / "val", root / "data.yaml",
                        root / "annotations" / "instances_val2017.json")
        else:
            split = root / "VisDrone2019-DET-val"
            required = (split / "images", split / "annotations",
                        root / "visdrone.yaml")
        for path in required:
            if not path.exists():
                raise FileNotFoundError(f"{dataset} validation input is missing: {path}")
    rows = _checkpoint_rows()
    protocol = _protocol(rows)
    print(f"PREFLIGHT: {len(rows)} checkpoints; output={output}", flush=True)
    for row in rows:
        status = "MATCH" if row["matches_comparison_checkpoint"] else "DIFFERENT"
        print(
            f"  {row['dataset']} {row['model']}: {row['n_images']} images; "
            f"sha256={row['weights_sha256'][:12]}...; "
            f"paper/cache comparison={status}", flush=True,
        )
    print(
        "COCO RT-DETR-L is supplementary; the COCO YOLO11n Best checkpoint "
        "differs from the paper-selected checkpoint.", flush=True,
    )
    print("PREFLIGHT SUCCESS: no training or inference performed.", flush=True)
    return protocol


def _inference_run(row, output, device):
    # The formal matrix already uses this retry for transient Windows locks.
    from . import vc_full as vc_full_module
    from .vc_matrix import _atomic_json_with_retry
    vc_full_module._atomic_json = _atomic_json_with_retry
    arguments = dict(
        kind=row["dataset"], dataset_root=Path(row["dataset_root"]),
        weights=Path(row["weights"]), output=output,
        sample_count=row["n_images"], model_id=row["model_id"],
        device=device, imgsz=INFERENCE["imgsz"],
        confidence_floor=INFERENCE["confidence_floor"],
        max_det=INFERENCE["max_det"],
    )
    if row["model"] == "rtdetr_l":
        return run_rtdetr_dataset_full(**arguments, seed=0)
    return run_dataset_full(
        **arguments, subset_seed=0, nms_iou=INFERENCE["nms_iou"])


def _gaming_run(row, inference_output, gaming_output, protocol):
    document, source = load_shard_sample(
        inference_output, row["n_images"], minimum_score=PRIMARY_THETA)
    completed = gaming_output / "gaming_complete.json"
    if completed.is_file():
        previous = _read_json(completed)
        if (
            previous.get("status") == "complete"
            and previous.get("source", {}).get("run_fingerprint") == source["run_fingerprint"]
            and previous.get("source", {}).get("cache_sha256") == source["cache_sha256"]
            and previous.get("protocol", {}).get("protocol_fingerprint")
            == protocol["protocol_fingerprint"]
        ):
            print(f"SKIP completed gaming: {gaming_output}", flush=True)
            return
        raise ValueError(f"gaming result differs from this run; choose a new RUN_NAME: {gaming_output}")
    if gaming_output.exists():
        raise ValueError(f"incomplete gaming directory exists: {gaming_output}")
    rows, audit = run_gaming_grid(
        document, shrink_scales=SHRINK_SCALES,
        expand_scales=EXPAND_SCALES, duplicate_factors=DUPLICATE_FACTORS,
        threshold_grid=(), primary_theta=PRIMARY_THETA,
        diagnostic_ap_levels=DIAGNOSTIC_AP_LEVELS,
    )
    gaming_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = gaming_output.with_name(f".{gaming_output.name}.{uuid4().hex}.tmp")
    write_gaming_report(
        temporary, document=document, source=source, rows=rows,
        audit=audit, protocol=protocol,
    )
    temporary.rename(gaming_output)


def _collect_rows(output, checkpoints):
    collected = []
    for row in checkpoints:
        path = output / "gaming" / row["dataset"] / row["model"] / "gaming_metrics.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for entry in csv.DictReader(handle):
                collected.append(dict(
                    dataset=entry.pop("dataset"), model=row["model"],
                    archived_checkpoint_sha256=row["weights_sha256"],
                    matches_comparison_checkpoint=row["matches_comparison_checkpoint"],
                    in_paper_model_set=row["in_paper_model_set"],
                    **entry,
                ))
    return collected


def _collect_threshold_rows(output, checkpoints):
    rows = []
    for checkpoint in checkpoints:
        run = output / "inference" / checkpoint["dataset"] / checkpoint["model"]
        completion = _read_json(run / "run_complete.json")
        source = Path(completion["analysis"]) / "vc_thresholds.csv"
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            for entry in csv.DictReader(handle):
                rows.append(dict(
                    dataset=checkpoint["dataset"], model=checkpoint["model"],
                    archived_checkpoint_sha256=checkpoint["weights_sha256"],
                    matches_comparison_checkpoint=checkpoint["matches_comparison_checkpoint"],
                    in_paper_model_set=checkpoint["in_paper_model_set"],
                    **entry,
                ))
    return rows

def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot write an empty results table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _plot_only_resume_allowed(previous, current):
    """Accept the v0.1.0 plotting-label fix without changing scored data."""
    old_runner_hash = "171fda011325a3193ddf3f069e212aa31a473ae91ede0526986e4b08fea7637c"
    runner = "region_eval/gaming_from_best.py"
    old_hashes = previous.get("source_sha256", {})
    new_hashes = current.get("source_sha256", {})
    if old_hashes.get(runner) != old_runner_hash:
        return False
    if {k: v for k, v in old_hashes.items() if k != runner} != {
        k: v for k, v in new_hashes.items() if k != runner
    }:
        return False
    excluded = {"protocol_fingerprint", "source_sha256"}
    return (
        {k: v for k, v in previous.items() if k not in excluded}
        == {k: v for k, v in current.items() if k not in excluded}
    )

def _plot_one(output, dataset, rows, family, variant, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source_dataset = "MSCOCO2017" if dataset == "mscoco" else "VisDrone2019-DET"
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for ax, model in zip(axes.flat, MODEL_FOLDERS):
        selected = sorted(
            (row for row in rows if row["dataset"] == source_dataset
             and row["model"] == model and row["manipulation"] == family
             and row["variant"] == variant),
            key=lambda row: float(row["level"]),
        )
        if len(selected) != (len(DUPLICATE_FACTORS) if family == "duplicate"
                             else len(EXPAND_SCALES) if family == "expand"
                             else len(SHRINK_SCALES)):
            raise ValueError(f"incomplete figure data: {dataset}/{model}/{family}/{variant}")
        x = [float(row["level"]) for row in selected]
        for field, label, color in (
            ("mAC", "mAC", "#2171b5"),
            ("VC", "VC", "#d95f0e"),
            ("F1", "F1", "#238b45"),
        ):
            ax.plot(x, [float(row[field]) for row in selected],
                    marker="o", markersize=3, linewidth=1.5,
                    color=color, label=label)
        ax.set_title(MODEL_LABELS[model] + (
            " (different checkpoint)" if dataset == "mscoco" and model == "yolo11n"
            else ""), fontsize=10)
        ax.set_xlabel("Multiplicity" if family == "duplicate" else "Box side scale")
        ax.set_ylabel("Absolute score")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, ncol=3)
    fig.suptitle(f"{dataset.upper()}: {family} / {variant}; one archived checkpoint per model")
    fig.tight_layout()
    fig.savefig(output / f"{filename}.png", dpi=220)
    fig.savefig(output / f"{filename}.pdf")
    plt.close(fig)


def run_best_weight_gaming(*, output=DEFAULT_OUTPUT, device="auto"):
    """Rerun all eight Best checkpoints; resume by image and by completed run."""
    output = Path(output).resolve()
    protocol = preflight(output=output)
    protocol_path = output / "protocol.json"
    if protocol_path.is_file():
        previous = _read_json(protocol_path)
        if previous.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
            if not _plot_only_resume_allowed(previous, protocol):
                raise ValueError(f"protocol changed; choose a new RUN_NAME: {output}")
            print("RESUME: applying plotting-label fix; reusing all completed inference and gaming runs", flush=True)
            protocol = previous
    else:
        output.mkdir(parents=True, exist_ok=True)
        if list(output.iterdir()):
            raise ValueError(f"nonempty output without protocol.json: {output}")
        _write_json(protocol_path, protocol)

    runtime_dir = PROJECT_ROOT / "outputs" / "vc_v1" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(runtime_dir)
    os.environ["YOLO_AUTOINSTALL"] = "false"
    checkpoints = protocol["checkpoints"]
    for index, row in enumerate(checkpoints, 1):
        identity = f"{row['dataset']}:{row['model']}"
        inference_output = output / "inference" / row["dataset"] / row["model"]
        gaming_output = output / "gaming" / row["dataset"] / row["model"]
        print(f"BEST {index:02d}/{len(checkpoints):02d} START: {identity}", flush=True)
        try:
            _inference_run(row, inference_output, device)
            _gaming_run(row, inference_output, gaming_output, protocol)
        except BaseException as exc:
            _write_json(output / "status.json", dict(
                status="failed_or_interrupted", identity=identity,
                error=repr(exc), updated_at=datetime.now(timezone.utc).isoformat(),
            ))
            raise
        _write_json(output / "status.json", dict(
            status="running", completed_runs=index, total_runs=len(checkpoints),
            last_completed=identity, updated_at=datetime.now(timezone.utc).isoformat(),
        ))
        print(f"BEST {index:02d}/{len(checkpoints):02d} COMPLETE: {identity}", flush=True)

    rows = _collect_rows(output, checkpoints)
    if len(rows) != 30 * len(checkpoints):
        raise ValueError(f"expected {30 * len(checkpoints)} attack rows, found {len(rows)}")
    _write_csv(output / "gaming_runs.csv", rows)
    threshold_rows = _collect_threshold_rows(output, checkpoints)
    if len(threshold_rows) != 19 * len(checkpoints):
        raise ValueError("incomplete 19-threshold results")
    _write_csv(output / "threshold_rows.csv", threshold_rows)
    for dataset in DATASET_ROOTS:
        for family, variant, filename in (
            ("shrink", "center_scale", f"{dataset}_shrink"),
            ("expand", "center_scale", f"{dataset}_expand"),
            ("duplicate", "exact", f"{dataset}_exact_duplicates"),
            ("duplicate", "controlled_jitter_2pct_center_1pct_size",
             f"{dataset}_jittered_duplicates"),
        ):
            _plot_one(output, dataset, rows, family, variant, filename)
    complete = dict(
        status="complete", version=VERSION, vc_version=VC_VERSION,
        gaming_version=GAMING_VERSION,
        protocol_fingerprint=protocol["protocol_fingerprint"],
        completed_runs=len(checkpoints), attack_rows=len(rows),
        threshold_rows=len(threshold_rows),
        full_validation_images_per_checkpoint=DATASET_COUNTS,
        checkpoint_comparison=[
            dict(dataset=row["dataset"], model=row["model"],
                 matches_comparison_checkpoint=row["matches_comparison_checkpoint"],
                 in_paper_model_set=row["in_paper_model_set"])
            for row in checkpoints
        ],
        outputs_sha256={
            "gaming_runs.csv": _file_hash(output / "gaming_runs.csv"),
            "threshold_rows.csv": _file_hash(output / "threshold_rows.csv"),
            **{
                f"{dataset}_{suffix}.{extension}": _file_hash(
                    output / f"{dataset}_{suffix}.{extension}")
                for dataset in DATASET_ROOTS
                for suffix in ("shrink", "expand", "exact_duplicates",
                               "jittered_duplicates")
                for extension in ("png", "pdf")
            },
        },
        training=False, fresh_model_inference=True,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    _write_json(output / "complete.json", complete)
    _write_json(output / "status.json", complete)
    print(json.dumps(complete, ensure_ascii=False, allow_nan=False), flush=True)
    print(f"SUCCESS: eight Best checkpoint gaming runs complete: {output}", flush=True)
    return output
