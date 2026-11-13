"""Formal offline anti-gaming analysis for the frozen VC v1 protocol.

The runner consumes the completed formal prediction caches. It selects the
three predeclared COCO checkpoints used by the paper and all five VisDrone
training seeds per architecture. It never trains a model or runs inference.
"""

from collections import defaultdict
import csv
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import shutil
import statistics
from time import sleep
from uuid import uuid4

from .case_analysis import _file_hash
from .gaming import (
    GAMING_VERSION,
    JITTER_CENTER_FRACTION,
    JITTER_SIZE_FRACTION,
    load_shard_sample,
    run_gaming_grid,
    write_gaming_report,
)
from .offline import PROJECT_ROOT
from .vc_v1 import PRIMARY_THETA, THETA_GRID, VC_VERSION


FORMAL_GAMING_VERSION = "vc-v1-formal-gaming-0.1.1"
MATRIX_ROOT = PROJECT_ROOT / "results" / "detection"
FIXED_ANALYSIS_ROOT = (
    PROJECT_ROOT / "results" / "fixed_checkpoint_analysis_v1"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "gaming_main"
EXPECTED_MATRIX_FINGERPRINT = (
    "916baa3f128416d7f82a8bd680511a11e8586944669af951b5811aa67deb5141"
)
MODELS = ("yolov8n", "yolo11n", "yolo11s")
VISDRONE_SEEDS = (0, 1, 2, 3, 4)
COCO_SELECTED_SEEDS = {"yolov8n": 0, "yolo11n": 1, "yolo11s": 0}

# The .71 and 1.41 anchors approximately halve and double box area.
SHRINK_SCALES = (1.00, 0.95, 0.90, 0.85, 0.80, 0.75, 0.71, 0.65, 0.60, 0.50)
EXPAND_SCALES = (1.00, 1.05, 1.10, 1.20, 1.30, 1.41, 1.60, 1.80, 2.00, 3.00)
DUPLICATE_FACTORS = (1, 2, 4, 8, 16)

# Diagnostic AP is deliberately sparse because it is expensive and remains a
# shared-cache diagnostic rather than official dataset AP. VC/mAC/F1/LRP are
# computed at every level.
DIAGNOSTIC_AP_LEVELS = (
    ("baseline", "unmodified", 1.0),
    ("shrink", "center_scale", 0.71),
    ("expand", "center_scale", 1.41),
    ("expand", "center_scale", 3.0),
    ("duplicate", "exact", 8),
    (
        "duplicate",
        "controlled_jitter_2pct_center_1pct_size",
        8,
    ),
)

METRIC_FIELDS = (
    "VC", "mAC", "F1", "LRP", "AP50", "AP75", "AP50_95", "AR50_95",
    "n_pred", "orphan_pred_count", "orphan_per_gt", "orphan_rate",
    "normalized_VC", "normalized_mAC", "normalized_F1",
    "normalized_one_minus_LRP", "normalized_AP50", "normalized_AP50_95",
    "normalized_n_pred", "normalized_orphan_count",
)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _fingerprint(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _atomic_text(path, value, attempts=20):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        for attempt in range(attempts):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt + 1 == attempts:
                    raise
                sleep(min(0.05 * (2 ** attempt), 2.0))
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path, value):
    _atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _atomic_csv(path, rows):
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(path, "\ufeff" + buffer.getvalue())


def _as_number(value):
    if value in (None, "", "NA"):
        return None
    return float(value)


def _source_records(matrix_root):
    complete_path = matrix_root / "matrix_complete.json"
    complete = _read_json(complete_path)
    if (complete.get("status") != "complete"
            or complete.get("matrix_fingerprint") != EXPECTED_MATRIX_FINGERPRINT
            or complete.get("completed_combinations") != 30):
        raise ValueError("the completed formal matrix is missing or changed")
    rows = _read_csv(matrix_root / "matrix_runs.csv")
    if len(rows) != 30:
        raise ValueError("formal matrix must contain 30 run rows")
    lookup = {
        (row["dataset"], row["model"], int(row["seed"])): row
        for row in rows
    }
    if len(lookup) != 30:
        raise ValueError("formal matrix contains duplicate run identities")
    selected = []
    for model in MODELS:
        seed = COCO_SELECTED_SEEDS[model]
        selected.append(lookup[("mscoco", model, seed)])
        for visdrone_seed in VISDRONE_SEEDS:
            selected.append(lookup[("visdrone", model, visdrone_seed)])
    for row in selected:
        run_directory = Path(row["output"]).resolve()
        run_complete = _read_json(run_directory / "run_complete.json")
        cache_complete = _read_json(run_directory / "cache_complete.json")
        if (run_complete.get("status") != "complete"
                or cache_complete.get("status") != "complete"
                or cache_complete.get("run_fingerprint")
                != run_complete.get("run_fingerprint")):
            raise ValueError(f"incomplete source cache: {run_directory}")
        if int(row["n_images"]) != cache_complete.get("n_images"):
            raise ValueError(f"source image count changed: {run_directory}")
        row["source_run_fingerprint"] = run_complete["run_fingerprint"]
        row["source_cache_sha256"] = cache_complete[
            "filtered_predictions_sha256"]
    return selected


def _protocol(matrix_root, fixed_root, records):
    fixed_complete = _read_json(fixed_root / "analysis_complete.json")
    if fixed_complete.get("status") != "complete":
        raise ValueError("fixed-checkpoint analysis is incomplete")
    fixed_selections = {
        row["model"]: int(row["selected_seed"])
        for row in _read_csv(fixed_root / "coco_fixed_bootstrap.csv")
        if row["metric"] == "VC"
    }
    if fixed_selections != COCO_SELECTED_SEEDS:
        raise ValueError("COCO fixed-checkpoint selections changed")
    value = dict(
        formal_gaming_version=FORMAL_GAMING_VERSION,
        gaming_version=GAMING_VERSION,
        vc_version=VC_VERSION,
        source_matrix_fingerprint=EXPECTED_MATRIX_FINGERPRINT,
        source_code={
            "gaming.py": _file_hash(Path(__file__).with_name("gaming.py")),
            "gaming_formal.py": _file_hash(Path(__file__)),
        },
        source_runs=[
            dict(
                dataset=row["dataset"], model=row["model"],
                seed=int(row["seed"]),
                run_fingerprint=row["source_run_fingerprint"],
                cache_sha256=row["source_cache_sha256"],
            )
            for row in records
        ],
        source_files={
            "matrix_complete.json": _file_hash(matrix_root / "matrix_complete.json"),
            "matrix_runs.csv": _file_hash(matrix_root / "matrix_runs.csv"),
            "matrix_thresholds.csv": _file_hash(matrix_root / "matrix_thresholds.csv"),
            "fixed_analysis_complete.json": _file_hash(
                fixed_root / "analysis_complete.json"),
        },
        selection=dict(
            mscoco=(
                "one predeclared checkpoint per architecture; "
                "yolov8n seed 0, yolo11n seed 1, yolo11s seed 0"
            ),
            visdrone="five training seeds per architecture (0-4)",
            selected_run_count=len(records),
        ),
        primary_theta=PRIMARY_THETA,
        analysis_minimum_score=PRIMARY_THETA,
        matching_iou=0.50,
        target_predictions=(
            "frozen one-to-one baseline true positives at theta=0.25"
        ),
        shrink_scales=list(SHRINK_SCALES),
        expand_scales=list(EXPAND_SCALES),
        duplicate_factors=list(DUPLICATE_FACTORS),
        duplicate_variants=[
            "exact equal-confidence copies",
            (
                "deterministic controlled jitter: 2% centre and 1% size; "
                "confidence unchanged"
            ),
        ],
        jitter=dict(
            centre_fraction=JITTER_CENTER_FRACTION,
            size_fraction=JITTER_SIZE_FRACTION,
        ),
        diagnostic_ap_levels=[list(row) for row in DIAGNOSTIC_AP_LEVELS],
        diagnostic_ap_role=(
            "sparse shared-cache diagnostic; not official dataset AP"
        ),
        theta_sensitivity=(
            "reuse the exact completed 0.05-0.95 formal-matrix rows"
        ),
        training=False,
        inference=False,
    )
    value["protocol_fingerprint"] = _fingerprint(value)
    return value


def _completed_run_matches(output, row, protocol):
    path = output / "gaming_complete.json"
    if not path.is_file():
        return False
    complete = _read_json(path)
    return (
        complete.get("status") == "complete"
        and complete.get("gaming_version") == GAMING_VERSION
        and complete.get("source", {}).get("run_fingerprint")
        == row["source_run_fingerprint"]
        and complete.get("source", {}).get("cache_sha256")
        == row["source_cache_sha256"]
        and complete.get("protocol", {}).get("protocol_fingerprint")
        == protocol["protocol_fingerprint"]
    )


def _read_run_rows(output, record):
    rows = []
    for source in _read_csv(output / "gaming_metrics.csv"):
        result = dict(
            dataset=record["dataset"],
            model=record["model"],
            seed=int(record["seed"]),
            model_id=record["model_id"],
            uncertainty_unit=(
                "fixed checkpoint" if record["dataset"] == "mscoco"
                else "training seed"
            ),
        )
        for key, value in source.items():
            if key in {"dataset", "model_id", "source_run_fingerprint"}:
                continue
            if key in {"manipulation", "variant", "level_label"}:
                result[key] = value
            else:
                result[key] = _as_number(value)
        rows.append(result)
    return rows


def _aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"], row["model"], row["manipulation"],
            row["variant"], row["level"], row["level_label"],
        )
        groups[key].append(row)
    output = []
    for key in sorted(groups):
        group = groups[key]
        dataset, model, manipulation, variant, level, label = key
        expected = 1 if dataset == "mscoco" else 5
        if len(group) != expected:
            raise ValueError(
                f"unexpected run count for {key}: {len(group)} != {expected}")
        row = dict(
            dataset=dataset,
            model=model,
            manipulation=manipulation,
            variant=variant,
            level=level,
            level_label=label,
            uncertainty=(
                "fixed checkpoint point estimate" if dataset == "mscoco"
                else "five training seeds; sample standard deviation"
            ),
            n_runs=len(group),
        )
        for field in METRIC_FIELDS:
            values = [item[field] for item in group if item.get(field) is not None]
            if values and len(values) != len(group):
                raise ValueError(f"partially missing metric: {key} {field}")
            row[f"{field}_mean"] = (
                statistics.fmean(values) if values else None)
            row[f"{field}_std"] = (
                statistics.stdev(values) if len(values) > 1
                else (0.0 if values else None)
            )
        output.append(row)
    return output


def _theta_rows(matrix_root):
    selected = []
    for row in _read_csv(matrix_root / "matrix_thresholds.csv"):
        dataset, model, seed = row["dataset"], row["model"], int(row["seed"])
        keep = (
            dataset == "visdrone" and seed in VISDRONE_SEEDS
        ) or (
            dataset == "mscoco" and COCO_SELECTED_SEEDS.get(model) == seed
        )
        if keep:
            selected.append(row)
    groups = defaultdict(list)
    for row in selected:
        key = (row["dataset"], row["model"], float(row["confidence_threshold"]))
        groups[key].append(row)
    metrics = ("VC", "mAC", "standard_F1", "standard_LRP", "orphan_rate")
    output = []
    for key in sorted(groups):
        values = groups[key]
        expected = 1 if key[0] == "mscoco" else 5
        if len(values) != expected:
            raise ValueError(f"incomplete theta group: {key}")
        row = dict(
            dataset=key[0], model=key[1], confidence_threshold=key[2],
            uncertainty=(
                "fixed checkpoint point estimate" if key[0] == "mscoco"
                else "five training seeds; sample standard deviation"
            ),
            n_runs=len(values),
        )
        for field in metrics:
            numeric = [float(item[field]) for item in values]
            row[f"{field}_mean"] = statistics.fmean(numeric)
            row[f"{field}_std"] = (
                statistics.stdev(numeric) if len(numeric) > 1 else 0.0)
        output.append(row)
    if len(output) != len(MODELS) * len(THETA_GRID) * 2:
        raise ValueError("theta sensitivity table is incomplete")
    return output


def _write_plots(output, aggregate, theta_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("shrink", "center_scale", "Shrink"),
        ("expand", "center_scale", "Expand"),
        ("duplicate", "exact", "Exact duplicate"),
        (
            "duplicate", "controlled_jitter_2pct_center_1pct_size",
            "Controlled-jitter duplicate",
        ),
    )
    colours = {"yolov8n": "#377eb8", "yolo11n": "#ff7f00", "yolo11s": "#4daf4a"}
    for dataset in ("mscoco", "visdrone"):
        figure, axes = plt.subplots(2, 2, figsize=(12, 8.5))
        for axis, (family, variant, title) in zip(axes.flat, panels):
            for model in MODELS:
                rows = sorted(
                    (
                        row for row in aggregate
                        if row["dataset"] == dataset
                        and row["model"] == model
                        and row["manipulation"] == family
                        and row["variant"] == variant
                    ),
                    key=lambda row: row["level"],
                )
                x = [row["level"] for row in rows]
                y = [row["normalized_VC_mean"] for row in rows]
                error = [row["normalized_VC_std"] for row in rows]
                axis.errorbar(
                    x, y, yerr=(error if dataset == "visdrone" else None),
                    marker="o", markersize=3, capsize=2,
                    color=colours[model], label=model,
                )
            axis.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
            axis.set_title(title)
            axis.set_xlabel("Side scale" if family in {"shrink", "expand"} else "Multiplicity")
            axis.set_ylabel("VC / baseline VC")
            axis.grid(alpha=0.25)
        axes.flat[0].legend(frameon=False)
        figure.suptitle(
            f"{dataset}: controlled anti-gaming response"
            + (" (mean ± seed SD)" if dataset == "visdrone" else " (fixed checkpoints)"))
        figure.tight_layout()
        figure.savefig(output / f"{dataset}_gaming_vc.png", dpi=220)
        plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    for axis, dataset, model in (
            (axes[0, 0], "mscoco", "yolov8n"),
            (axes[0, 1], "mscoco", "yolo11n"),
            (axes[0, 2], "mscoco", "yolo11s"),
            (axes[1, 0], "visdrone", "yolov8n"),
            (axes[1, 1], "visdrone", "yolo11n"),
            (axes[1, 2], "visdrone", "yolo11s")):
        rows = sorted(
            (row for row in theta_rows if row["dataset"] == dataset and row["model"] == model),
            key=lambda row: row["confidence_threshold"],
        )
        x = [row["confidence_threshold"] for row in rows]
        for field, label in (
                ("VC_mean", "VC"), ("mAC_mean", "mAC"),
                ("standard_F1_mean", "F1")):
            axis.plot(x, [row[field] for row in rows], marker="o", markersize=2.5, label=label)
        axis.plot(
            x, [1.0 - row["standard_LRP_mean"] for row in rows],
            marker="o", markersize=2.5, label="1-LRP",
        )
        axis.axvline(PRIMARY_THETA, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(f"{dataset} / {model}")
        axis.set_xlabel("Confidence threshold theta")
        axis.set_ylabel("Score")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, ncol=2)
    figure.suptitle("Theta sensitivity from the completed formal cache matrix")
    figure.tight_layout()
    figure.savefig(output / "theta_sensitivity.png", dpi=220)
    plt.close(figure)


def _summary_markdown(protocol, raw_rows, aggregate_rows):
    return f"""# VC v1 formal controlled anti-gaming analysis

This output evaluates the frozen VC v1 metric under controlled box shrink,
box expansion, and prediction duplication. It reads completed prediction
caches only. No training and no model inference are performed.

## Statistical scope

- MSCOCO: one predeclared checkpoint per architecture ({COCO_SELECTED_SEEDS}).
- VisDrone: five training seeds per architecture; aggregate rows report the
  mean and sample standard deviation.
- Primary confidence threshold: {PRIMARY_THETA}.
- Attack targets: baseline one-to-one true positives, frozen before each
  manipulation family.
- Controlled duplicate jitter: {JITTER_CENTER_FRACTION:.0%} of box size for
  centre position and {JITTER_SIZE_FRACTION:.0%} for width/height; confidence
  is unchanged.

## Outputs

- `gaming_runs.csv`: all per-run attack results ({len(raw_rows)} rows).
- `gaming_aggregate.csv`: model-level fixed-checkpoint or five-seed summaries
  ({len(aggregate_rows)} rows).
- `theta_sensitivity.csv`: the already completed 0.05-0.95 theta results,
  selected and aggregated under the same paper protocol.
- `mscoco_gaming_vc.png` and `visdrone_gaming_vc.png`: normalized VC curves.
- `theta_sensitivity.png`: VC, mAC, F1, and 1-LRP versus theta.
- `runs/`: independent, resumable source-run reports and provenance manifests.

Diagnostic AP is recalculated only for the six declared representative
conditions. It remains an internal shared-cache diagnostic. The full curves
for VC, mAC, F1, and LRP are the primary anti-gaming evidence.

Protocol fingerprint: `{protocol['protocol_fingerprint']}`.
"""


def run_formal_gaming(
        *, matrix_root=MATRIX_ROOT, fixed_root=FIXED_ANALYSIS_ROOT,
        output=DEFAULT_OUTPUT):
    """Run or resume the paper-scope offline anti-gaming matrix."""
    matrix_root = Path(matrix_root).resolve()
    fixed_root = Path(fixed_root).resolve()
    output = Path(output).resolve()
    records = _source_records(matrix_root)
    protocol = _protocol(matrix_root, fixed_root, records)
    complete_path = output / "gaming_formal_complete.json"
    if complete_path.is_file():
        complete = _read_json(complete_path)
        if complete.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
            raise ValueError("completed Gaming output uses a different protocol")
        print(f"SKIP: formal Gaming analysis already complete: {output}")
        return output

    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.is_file():
        previous = _read_json(protocol_path)
        if previous.get("protocol_fingerprint") != protocol["protocol_fingerprint"]:
            completed_manifests = list(
                output.glob("runs/**/gaming_complete.json"))
            if completed_manifests:
                raise ValueError(
                    "incomplete Gaming output contains completed runs from a "
                    "different protocol; choose a fresh output directory")
            shutil.rmtree(output)
            output.mkdir(parents=True, exist_ok=False)
            _atomic_json(protocol_path, protocol)
    else:
        _atomic_json(protocol_path, protocol)

    completed = 0
    try:
        for index, record in enumerate(records, 1):
            dataset, model, seed = record["dataset"], record["model"], int(record["seed"])
            final = output / "runs" / dataset / model / f"seed_{seed}"
            if _completed_run_matches(final, record, protocol):
                completed += 1
                print(f"GAMING {index:02d}/{len(records)} SKIP: {dataset}:{model}:seed{seed}", flush=True)
                continue
            if final.exists():
                shutil.rmtree(final)
            final.parent.mkdir(parents=True, exist_ok=True)
            temporary = final.with_name(f".{final.name}.{uuid4().hex}.tmp")
            print(f"GAMING {index:02d}/{len(records)} START: {dataset}:{model}:seed{seed}", flush=True)
            try:
                document, source = load_shard_sample(
                    Path(record["output"]), int(record["n_images"]),
                    minimum_score=PRIMARY_THETA,
                )
                rows, audit = run_gaming_grid(
                    document,
                    shrink_scales=SHRINK_SCALES,
                    expand_scales=EXPAND_SCALES,
                    duplicate_factors=DUPLICATE_FACTORS,
                    threshold_grid=(),
                    primary_theta=PRIMARY_THETA,
                    diagnostic_ap_levels=DIAGNOSTIC_AP_LEVELS,
                )
                write_gaming_report(
                    temporary,
                    document=document,
                    source=source,
                    rows=rows,
                    audit=audit,
                    protocol=protocol,
                )
                temporary.rename(final)
            except BaseException:
                if temporary.exists():
                    shutil.rmtree(temporary)
                raise
            completed += 1
            _atomic_json(output / "gaming_status.json", dict(
                status="running",
                protocol_fingerprint=protocol["protocol_fingerprint"],
                completed_runs=completed,
                total_runs=len(records),
                last_completed=f"{dataset}:{model}:seed{seed}",
                updated_at=datetime.now(timezone.utc).isoformat(),
            ))
            print(f"GAMING {index:02d}/{len(records)} COMPLETE: {final}", flush=True)

        raw_rows = []
        for record in records:
            final = (
                output / "runs" / record["dataset"] / record["model"]
                / f"seed_{int(record['seed'])}"
            )
            if not _completed_run_matches(final, record, protocol):
                raise ValueError(f"completed Gaming run changed: {final}")
            raw_rows.extend(_read_run_rows(final, record))
        aggregate_rows = _aggregate(raw_rows)
        theta_rows = _theta_rows(matrix_root)
        _atomic_csv(output / "gaming_runs.csv", raw_rows)
        _atomic_csv(output / "gaming_aggregate.csv", aggregate_rows)
        _atomic_csv(output / "theta_sensitivity.csv", theta_rows)
        _write_plots(output, aggregate_rows, theta_rows)
        _atomic_text(
            output / "README.md",
            _summary_markdown(protocol, raw_rows, aggregate_rows),
        )
        result_names = (
            "protocol.json", "gaming_runs.csv", "gaming_aggregate.csv",
            "theta_sensitivity.csv", "mscoco_gaming_vc.png",
            "visdrone_gaming_vc.png", "theta_sensitivity.png", "README.md",
        )
        complete = dict(
            status="complete",
            formal_gaming_version=FORMAL_GAMING_VERSION,
            gaming_version=GAMING_VERSION,
            vc_version=VC_VERSION,
            protocol_fingerprint=protocol["protocol_fingerprint"],
            completed_runs=len(records),
            mscoco_fixed_checkpoints=3,
            visdrone_seed_runs=15,
            attack_rows=len(raw_rows),
            aggregate_rows=len(aggregate_rows),
            theta_rows=len(theta_rows),
            training=False,
            inference=False,
            result_files={name: _file_hash(output / name) for name in result_names},
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        _atomic_json(complete_path, complete)
        _atomic_json(output / "gaming_status.json", complete)
    except BaseException as exc:
        _atomic_json(output / "gaming_status.json", dict(
            status="failed_or_interrupted",
            protocol_fingerprint=protocol["protocol_fingerprint"],
            completed_runs=completed,
            total_runs=len(records),
            error=repr(exc),
            updated_at=datetime.now(timezone.utc).isoformat(),
        ))
        raise

    print(json.dumps(complete, ensure_ascii=False, allow_nan=False), flush=True)
    print(f"SUCCESS: formal offline Gaming analysis complete: {output}", flush=True)
    print("No training and no model inference were performed.", flush=True)
    return output
