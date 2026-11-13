"""Controlled collective-fragmentation experiment for VC v1.

The experiment consumes the completed formal prediction caches.  It replaces
baseline true-positive predictions with deterministic fragments of the
prediction/GT intersection, then measures representation fragmentation and
retained collective support.  It never trains a model or performs inference.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import statistics
from uuid import uuid4

from .case_analysis import _file_hash
from .geometry import area, intersection
from .gaming import _normalizations, load_shard_sample, score_document
from .gaming_formal import (
    EXPECTED_MATRIX_FINGERPRINT,
    MATRIX_ROOT,
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _fingerprint,
    _read_csv,
    _read_json,
    _source_records,
)
from .matching_analysis import match_class
from .offline import PROJECT_ROOT
from .responsibility_analysis import assignment_weights
from .vc_v1 import MATCHING_IOU, PRIMARY_THETA, VC_VERSION


FRAGMENTATION_VERSION = "vc-v1-collective-fragmentation-0.1.0"
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "results" / "fragmentation"
)

Q_GRID = (1, 2, 3, 4)
RETENTION_GRID = (1.0, 0.8, 0.6, 0.4, 0.2)
RETENTION_Q = 4

TEXT_FIELDS = {"experiment", "variant", "level_label"}
COMMON_FIELDS = {"dataset", "model_id", "source_run_fingerprint"}

METRIC_FIELDS = (
    "VC",
    "mAC",
    "F1",
    "LRP",
    "AP50",
    "AP75",
    "AP50_95",
    "AR50_95",
    "n_images",
    "n_gt",
    "n_pred",
    "assigned_pred_count",
    "orphan_pred_count",
    "orphan_rate",
    "orphan_per_gt",
    "normalized_VC",
    "normalized_mAC",
    "normalized_F1",
    "normalized_one_minus_LRP",
    "normalized_AP50",
    "normalized_AP50_95",
    "normalized_n_pred",
    "normalized_orphan_count",
    "target_pair_count",
    "replaced_prediction_count",
    "generated_fragment_count",
    "added_prediction_count",
    "intended_owner_match_count",
    "ownership_change_count",
    "ownership_change_rate",
    "fragment_orphan_count",
    "retained_area_ratio_actual",
    "retained_area_ratio_abs_error",
    "max_target_retention_abs_error",
)


def _target_pairs(document, theta=PRIMARY_THETA, iou_threshold=MATCHING_IOU):
    """Freeze baseline one-to-one TP owners and their controlled support."""
    image_lookup = {image["image_id"]: image for image in document["images"]}
    if len(image_lookup) != len(document["images"]):
        raise ValueError("duplicate image ids in source document")
    targets = {}
    for class_id in document["metadata"]["class_ids"]:
        events, _ = match_class(document, class_id, iou_threshold, theta)
        for event in events:
            if not event["is_tp"]:
                continue
            image = image_lookup[event["image_id"]]
            prediction = next(
                row
                for row in image["predictions"]
                if row["id"] == event["pred_id"]
            )
            owner = next(
                row
                for row in image["gts"]
                if row["id"] == event["matched_gt_id"]
            )
            support = intersection(
                tuple(prediction["xyxy"]), tuple(owner["xyxy"])
            )
            if support is None or area(support) <= 0.0:
                raise AssertionError("a baseline TP must have positive support")
            key = (image["image_id"], prediction["id"])
            if key in targets:
                raise ValueError(f"prediction matched more than once: {key}")
            targets[key] = dict(
                image_id=image["image_id"],
                pred_id=prediction["id"],
                owner_gt_id=owner["id"],
                class_id=class_id,
                score=prediction["score"],
                support_xyxy=list(support),
                support_area=area(support),
                baseline_iou=event["matched_iou"],
            )
    if not targets:
        raise ValueError("source run contains no baseline true-positive targets")
    return targets


def _fragment_identifier(parent_id, q, rho, row, column):
    rho_tag = f"{rho:.2f}".replace(".", "p")
    return f"{parent_id}__cf_q{q}_r{rho_tag}_{row}_{column}"


def _fragments(target, q, rho):
    if type(q) is not int or q <= 0:
        raise ValueError("q must be a positive integer")
    if not math.isfinite(rho) or not 0.0 < rho <= 1.0:
        raise ValueError("rho must be finite and in (0, 1]")
    x1, y1, x2, y2 = target["support_xyxy"]
    side_scale = math.sqrt(rho)
    result = []
    for row in range(q):
        # Reuse the exact support endpoints on the outer grid boundary. This
        # avoids a one-ulp overshoot for floating-point COCO coordinates.
        cell_y1 = y1 if row == 0 else y1 + (y2 - y1) * row / q
        cell_y2 = y2 if row + 1 == q else y1 + (y2 - y1) * (row + 1) / q
        for column in range(q):
            cell_x1 = x1 if column == 0 else x1 + (x2 - x1) * column / q
            cell_x2 = x2 if column + 1 == q else x1 + (x2 - x1) * (column + 1) / q
            centre_x = (cell_x1 + cell_x2) / 2.0
            centre_y = (cell_y1 + cell_y2) / 2.0
            half_width = (cell_x2 - cell_x1) * side_scale / 2.0
            half_height = (cell_y2 - cell_y1) * side_scale / 2.0
            if rho == 1.0:
                box = [cell_x1, cell_y1, cell_x2, cell_y2]
            else:
                box = [
                    max(cell_x1, centre_x - half_width),
                    max(cell_y1, centre_y - half_height),
                    min(cell_x2, centre_x + half_width),
                    min(cell_y2, centre_y + half_height),
                ]
            # Clamp after all arithmetic. This only removes floating-point
            # round-off at the support boundary; the checks below still reject
            # a genuinely invalid fragment.
            box = [
                max(x1, min(x2, float(box[0]))),
                max(y1, min(y2, float(box[1]))),
                max(x1, min(x2, float(box[2]))),
                max(y1, min(y2, float(box[3]))),
            ]
            if not box[0] < box[2] or not box[1] < box[3]:
                raise AssertionError("fragment construction became degenerate")
            if not (
                x1 <= box[0] < box[2] <= x2
                and y1 <= box[1] < box[3] <= y2
            ):
                raise AssertionError("fragment escaped its controlled support")
            result.append(
                dict(
                    id=_fragment_identifier(
                        target["pred_id"], q, rho, row, column
                    ),
                    xyxy=box,
                    class_id=target["class_id"],
                    score=target["score"],
                )
            )
    return result


def _ownership_audit(document, intended_owners):
    intended_owner_matches = 0
    ownership_changes = 0
    orphans = 0
    seen = 0
    for image in document["images"]:
        gt_by_class = defaultdict(list)
        pred_by_class = defaultdict(list)
        for gt in image["gts"]:
            gt_by_class[gt["class_id"]].append(gt)
        for prediction in image["predictions"]:
            pred_by_class[prediction["class_id"]].append(prediction)
        for class_id in sorted(set(gt_by_class) | set(pred_by_class)):
            weights = assignment_weights(
                gt_by_class[class_id], pred_by_class[class_id], "hard_iou"
            )
            for prediction in pred_by_class[class_id]:
                key = (image["image_id"], prediction["id"])
                if key not in intended_owners:
                    continue
                seen += 1
                owners = tuple(weights[prediction["id"]])
                if not owners:
                    orphans += 1
                elif owners == (intended_owners[key],):
                    intended_owner_matches += 1
                else:
                    ownership_changes += 1
    if seen != len(intended_owners):
        raise AssertionError(
            f"ownership audit missed fragments: {seen} != {len(intended_owners)}"
        )
    if orphans:
        raise AssertionError(
            "controlled fragments lie inside their owner GT and cannot be orphaned"
        )
    return dict(
        intended_owner_match_count=intended_owner_matches,
        ownership_change_count=ownership_changes,
        ownership_change_rate=(ownership_changes / seen if seen else 0.0),
        fragment_orphan_count=orphans,
    )


def transform_document(document, targets, *, q, rho):
    """Replace frozen target predictions by deterministic support fragments."""
    result = deepcopy(document)
    intended_owners = {}
    replaced = 0
    generated = 0
    total_support_area = 0.0
    total_fragment_area = 0.0
    maximum_target_error = 0.0

    for image in result["images"]:
        predictions = []
        for prediction in image["predictions"]:
            key = (image["image_id"], prediction["id"])
            target = targets.get(key)
            if target is None:
                predictions.append(prediction)
                continue
            fragments = _fragments(target, q, rho)
            fragment_area = sum(area(tuple(row["xyxy"])) for row in fragments)
            expected_area = rho * target["support_area"]
            relative_error = abs(fragment_area - expected_area) / max(
                1.0, expected_area
            )
            maximum_target_error = max(maximum_target_error, relative_error)
            total_support_area += target["support_area"]
            total_fragment_area += fragment_area
            replaced += 1
            generated += len(fragments)
            for fragment in fragments:
                fragment_key = (image["image_id"], fragment["id"])
                if fragment_key in intended_owners:
                    raise ValueError(f"duplicate fragment id: {fragment_key}")
                intended_owners[fragment_key] = target["owner_gt_id"]
            predictions.extend(fragments)
        image["predictions"] = predictions

    if replaced != len(targets):
        raise AssertionError(
            f"not every frozen target was replaced: {replaced} != {len(targets)}"
        )
    expected_generated = len(targets) * q * q
    if generated != expected_generated:
        raise AssertionError(
            f"fragment count changed: {generated} != {expected_generated}"
        )
    maximum_predictions = max(
        (len(image["predictions"]) for image in result["images"]), default=1
    )
    result["metadata"]["postprocessing"]["max_det"] = max(
        result["metadata"]["postprocessing"]["max_det"], maximum_predictions
    )
    owner_audit = _ownership_audit(result, intended_owners)
    actual_ratio = total_fragment_area / total_support_area
    audit = dict(
        target_pair_count=len(targets),
        q=q,
        fragment_count=q * q,
        retention=rho,
        replaced_prediction_count=replaced,
        generated_fragment_count=generated,
        added_prediction_count=generated - replaced,
        retained_area_ratio_actual=actual_ratio,
        retained_area_ratio_abs_error=abs(actual_ratio - rho),
        max_target_retention_abs_error=maximum_target_error,
        **owner_audit,
    )
    if audit["retained_area_ratio_abs_error"] > 1e-12:
        raise AssertionError("aggregate retained-area ratio failed tolerance")
    if maximum_target_error > 1e-12:
        raise AssertionError("per-target retained-area ratio failed tolerance")
    return result, audit


def run_fragmentation_grid(document):
    """Evaluate the two frozen collective-fragmentation experiment families."""
    targets = _target_pairs(document)
    cache = {}

    def evaluate(q, rho):
        key = (q, float(rho))
        if key not in cache:
            transformed, audit = transform_document(
                document, targets, q=q, rho=rho
            )
            metrics = score_document(
                transformed, PRIMARY_THETA, include_diagnostic_ap=True
            )
            cache[key] = (metrics, audit)
        return cache[key]

    baseline_metrics, baseline_audit = evaluate(1, 1.0)
    rows = []

    def append(experiment, variant, level, label, q, rho):
        metrics, audit = evaluate(q, rho)
        row = dict(
            experiment=experiment,
            variant=variant,
            level=float(level),
            level_label=label,
            primary_theta=PRIMARY_THETA,
            matching_iou=MATCHING_IOU,
            **audit,
            **metrics,
        )
        rows.append(_normalizations(row, baseline_metrics))

    for q in Q_GRID:
        append(
            "fragmentation",
            "fixed_union_support",
            q * q,
            f"{q}x{q} grid ({q*q} fragments)",
            q,
            1.0,
        )
    for rho in RETENTION_GRID:
        append(
            "retention",
            "fixed_4x4_grid",
            rho,
            f"retained support {rho:.1f}",
            RETENTION_Q,
            rho,
        )

    if len(rows) != len(Q_GRID) + len(RETENTION_GRID):
        raise AssertionError("fragmentation grid row count changed")
    return rows, dict(
        baseline_target_pair_count=len(targets),
        baseline_metrics=baseline_metrics,
        baseline_audit=baseline_audit,
        unique_scored_conditions=len(cache),
        target_rule=(
            "baseline one-to-one true positives at theta=0.25 and IoU=0.50; "
            "controlled support is prediction/owner-GT intersection"
        ),
    )


def _protocol(matrix_root, records):
    value = dict(
        fragmentation_version=FRAGMENTATION_VERSION,
        vc_version=VC_VERSION,
        source_matrix_fingerprint=EXPECTED_MATRIX_FINGERPRINT,
        source_code={
            "collective_fragmentation.py": _file_hash(Path(__file__)),
            "gaming.py": _file_hash(Path(__file__).with_name("gaming.py")),
            "matching_analysis.py": _file_hash(
                Path(__file__).with_name("matching_analysis.py")
            ),
            "responsibility_analysis.py": _file_hash(
                Path(__file__).with_name("responsibility_analysis.py")
            ),
            "vc_v1.py": _file_hash(Path(__file__).with_name("vc_v1.py")),
        },
        source_runs=[
            dict(
                dataset=row["dataset"],
                model=row["model"],
                seed=int(row["seed"]),
                run_fingerprint=row["source_run_fingerprint"],
                cache_sha256=row["source_cache_sha256"],
            )
            for row in records
        ],
        purpose=(
            "test whether regional metrics retain graded collective-support "
            "information after one-to-one matching saturates"
        ),
        primary_theta=PRIMARY_THETA,
        matching_iou=MATCHING_IOU,
        support_definition="prediction intersect frozen matched owner GT",
        fragmentation_q=list(Q_GRID),
        fragmentation_counts=[q * q for q in Q_GRID],
        retention_q=RETENTION_Q,
        retention_levels=list(RETENTION_GRID),
        tile_rule=(
            "regular q-by-q support partition; each tile is centre-shrunk "
            "by sqrt(rho) so total retained area equals rho times support"
        ),
        confidence_rule="every fragment inherits its parent confidence",
        non_target_rule="all GT and non-target predictions remain unchanged",
        diagnostic_ap_role=(
            "fixed-theta AP on controlled cached outputs; not official dataset AP"
        ),
        training=False,
        inference=False,
    )
    value["protocol_fingerprint"] = _fingerprint(value)
    return value


def _completed_run_matches(output, record, protocol):
    path = output / "fragmentation_complete.json"
    if not path.is_file():
        return False
    complete = _read_json(path)
    return (
        complete.get("status") == "complete"
        and complete.get("fragmentation_version") == FRAGMENTATION_VERSION
        and complete.get("source", {}).get("run_fingerprint")
        == record["source_run_fingerprint"]
        and complete.get("source", {}).get("cache_sha256")
        == record["source_cache_sha256"]
        and complete.get("protocol_fingerprint")
        == protocol["protocol_fingerprint"]
    )


def _write_run_report(output, *, document, source, rows, audit, protocol):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"fragmentation output already exists: {output}")
    output.mkdir(parents=True)
    metadata = document["metadata"]
    common = dict(
        dataset=metadata["dataset"],
        model_id=metadata["model_id"],
        source_run_fingerprint=source["run_fingerprint"],
    )
    complete_rows = [dict(common, **row) for row in rows]
    _atomic_csv(output / "fragmentation_metrics.csv", complete_rows)
    _atomic_json(output / "fragmentation_audit.json", audit)
    readme = f"""# Controlled collective-fragmentation result

- Dataset: `{metadata['dataset']}`
- Model: `{metadata['model_id']}`
- Images: `{len(document['images'])}`
- Frozen target pairs: `{audit['baseline_target_pair_count']}`
- Conditions: `{len(rows)}` rows, `{audit['unique_scored_conditions']}` unique scores
- Training: none
- Model inference: none

The q=1 prediction/GT-intersection support is the controlled reference.
`fragmentation_metrics.csv` reports full-union fragmentation and fixed-4x4
coverage retention. Diagnostic AP is evaluated at theta={PRIMARY_THETA}.
"""
    _atomic_text(output / "README.md", readme)
    result_names = (
        "fragmentation_metrics.csv",
        "fragmentation_audit.json",
        "README.md",
    )
    complete = dict(
        status="complete",
        fragmentation_version=FRAGMENTATION_VERSION,
        vc_version=VC_VERSION,
        protocol_fingerprint=protocol["protocol_fingerprint"],
        created_at=datetime.now(timezone.utc).isoformat(),
        dataset=metadata["dataset"],
        model_id=metadata["model_id"],
        source=source,
        audit=audit,
        result_files={name: _file_hash(output / name) for name in result_names},
    )
    _atomic_json(output / "fragmentation_complete.json", complete)


def _parse_number(value):
    if value in (None, "", "NA"):
        return None
    return float(value)


def _read_run_rows(output, record):
    rows = []
    for source in _read_csv(output / "fragmentation_metrics.csv"):
        row = dict(
            dataset=record["dataset"],
            model=record["model"],
            seed=int(record["seed"]),
            model_id=record["model_id"],
            uncertainty_unit=(
                "fixed checkpoint"
                if record["dataset"] == "mscoco"
                else "training seed"
            ),
        )
        for key, value in source.items():
            if key in COMMON_FIELDS:
                continue
            if key in TEXT_FIELDS:
                row[key] = value
            else:
                row[key] = _parse_number(value)
        rows.append(row)
    return rows


def _aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["model"],
            row["experiment"],
            row["variant"],
            row["level"],
            row["level_label"],
            row["q"],
            row["fragment_count"],
            row["retention"],
        )
        groups[key].append(row)
    output = []
    for key in sorted(groups):
        group = groups[key]
        dataset = key[0]
        expected = 1 if dataset == "mscoco" else 5
        if len(group) != expected:
            raise ValueError(
                f"unexpected run count for {key}: {len(group)} != {expected}"
            )
        row = dict(
            dataset=key[0],
            model=key[1],
            experiment=key[2],
            variant=key[3],
            level=key[4],
            level_label=key[5],
            q=key[6],
            fragment_count=key[7],
            retention=key[8],
            uncertainty=(
                "fixed checkpoint point estimate"
                if dataset == "mscoco"
                else "five training seeds; sample standard deviation"
            ),
            n_runs=len(group),
        )
        for field in METRIC_FIELDS:
            values = [item[field] for item in group if item.get(field) is not None]
            if values and len(values) != len(group):
                raise ValueError(f"partially missing metric: {key} {field}")
            row[f"{field}_mean"] = (
                statistics.fmean(values) if values else None
            )
            row[f"{field}_std"] = (
                statistics.stdev(values)
                if len(values) > 1
                else (0.0 if values else None)
            )
        output.append(row)
    return output


def preflight_collective_fragmentation(*, matrix_root=MATRIX_ROOT):
    matrix_root = Path(matrix_root).resolve()
    records = _source_records(matrix_root)
    protocol = _protocol(matrix_root, records)
    summary = dict(
        status="passed",
        protocol_fingerprint=protocol["protocol_fingerprint"],
        source_runs=len(records),
        mscoco_runs=sum(row["dataset"] == "mscoco" for row in records),
        visdrone_runs=sum(row["dataset"] == "visdrone" for row in records),
        reported_rows_per_run=len(Q_GRID) + len(RETENTION_GRID),
        unique_scores_per_run=(
            len(Q_GRID) + len(RETENTION_GRID) - 1
        ),
        expected_run_rows=len(records) * (len(Q_GRID) + len(RETENTION_GRID)),
        expected_aggregate_rows=6 * (len(Q_GRID) + len(RETENTION_GRID)),
        training=False,
        inference=False,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(
        "PREFLIGHT SUCCESS: all formal caches required by the frozen "
        "collective-fragmentation protocol are available.",
        flush=True,
    )
    return summary


def _summary_markdown(protocol, raw_rows, aggregate_rows):
    return f"""# Controlled collective-fragmentation experiment

This output evaluates whether mAC and VC retain graded collective-support
information after standard one-to-one matching loses individual valid boxes.
It uses completed prediction caches only. No training or inference is run.

- Source runs: 18 (three fixed MS COCO checkpoints and fifteen VisDrone seed runs)
- Per-run rows: {len(raw_rows)}
- Aggregate rows: {len(aggregate_rows)}
- Fragment counts: {[q*q for q in Q_GRID]}
- Fixed-4x4 retention levels: {list(RETENTION_GRID)}
- Primary theta: {PRIMARY_THETA}
- Matching IoU: {MATCHING_IOU}
- Protocol fingerprint: `{protocol['protocol_fingerprint']}`

The experiment is a controlled metric-property test. It does not redefine
fragmented predictions as correct object instances. Diagnostic AP is computed
at the fixed theta on the transformed cached outputs and is distinct from
official dataset AP.
"""


def run_collective_fragmentation(
    *, matrix_root=MATRIX_ROOT, output=DEFAULT_OUTPUT
):
    """Run or resume the complete 18-run fragmentation experiment."""
    matrix_root = Path(matrix_root).resolve()
    output = Path(output).resolve()
    records = _source_records(matrix_root)
    protocol = _protocol(matrix_root, records)
    complete_path = output / "collective_fragmentation_complete.json"

    if complete_path.is_file():
        complete = _read_json(complete_path)
        if complete.get("protocol_fingerprint") != protocol[
            "protocol_fingerprint"
        ]:
            raise ValueError(
                "completed collective-fragmentation output uses another protocol"
            )
        print(f"SKIP: collective fragmentation already complete: {output}")
        return output

    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.is_file():
        previous = _read_json(protocol_path)
        if previous.get("protocol_fingerprint") != protocol[
            "protocol_fingerprint"
        ]:
            if list(output.glob("runs/**/fragmentation_complete.json")):
                raise ValueError(
                    "partial fragmentation output uses another protocol; "
                    "choose a fresh output directory"
                )
            shutil.rmtree(output)
            output.mkdir(parents=True)
    _atomic_json(protocol_path, protocol)

    completed = 0
    try:
        for index, record in enumerate(records, 1):
            dataset = record["dataset"]
            model = record["model"]
            seed = int(record["seed"])
            final = output / "runs" / dataset / model / f"seed_{seed}"
            if _completed_run_matches(final, record, protocol):
                completed += 1
                print(
                    f"FRAGMENT {index:02d}/{len(records)} SKIP: "
                    f"{dataset}:{model}:seed{seed}",
                    flush=True,
                )
                continue
            if final.exists():
                shutil.rmtree(final)
            final.parent.mkdir(parents=True, exist_ok=True)
            temporary = final.with_name(f".{final.name}.{uuid4().hex}.tmp")
            print(
                f"FRAGMENT {index:02d}/{len(records)} START: "
                f"{dataset}:{model}:seed{seed}; unique scores=8",
                flush=True,
            )
            try:
                document, source = load_shard_sample(
                    Path(record["output"]),
                    int(record["n_images"]),
                    minimum_score=PRIMARY_THETA,
                )
                rows, audit = run_fragmentation_grid(document)
                _write_run_report(
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
            _atomic_json(
                output / "collective_fragmentation_status.json",
                dict(
                    status="running",
                    protocol_fingerprint=protocol["protocol_fingerprint"],
                    completed_runs=completed,
                    total_runs=len(records),
                    last_completed=f"{dataset}:{model}:seed{seed}",
                    updated_at=datetime.now(timezone.utc).isoformat(),
                ),
            )
            print(
                f"FRAGMENT {index:02d}/{len(records)} COMPLETE: {final}",
                flush=True,
            )

        raw_rows = []
        for record in records:
            final = (
                output
                / "runs"
                / record["dataset"]
                / record["model"]
                / f"seed_{int(record['seed'])}"
            )
            if not _completed_run_matches(final, record, protocol):
                raise ValueError(f"completed fragmentation run changed: {final}")
            raw_rows.extend(_read_run_rows(final, record))
        expected_raw = len(records) * (len(Q_GRID) + len(RETENTION_GRID))
        if len(raw_rows) != expected_raw:
            raise ValueError(
                f"expected {expected_raw} per-run rows, found {len(raw_rows)}"
            )
        aggregate_rows = _aggregate(raw_rows)
        expected_aggregate = 6 * (len(Q_GRID) + len(RETENTION_GRID))
        if len(aggregate_rows) != expected_aggregate:
            raise ValueError(
                f"expected {expected_aggregate} aggregate rows, "
                f"found {len(aggregate_rows)}"
            )
        _atomic_csv(output / "fragmentation_runs.csv", raw_rows)
        _atomic_csv(output / "fragmentation_aggregate.csv", aggregate_rows)
        _atomic_text(
            output / "README.md",
            _summary_markdown(protocol, raw_rows, aggregate_rows),
        )
        result_names = (
            "protocol.json",
            "fragmentation_runs.csv",
            "fragmentation_aggregate.csv",
            "README.md",
        )
        complete = dict(
            status="complete",
            fragmentation_version=FRAGMENTATION_VERSION,
            vc_version=VC_VERSION,
            protocol_fingerprint=protocol["protocol_fingerprint"],
            completed_runs=len(records),
            run_rows=len(raw_rows),
            aggregate_rows=len(aggregate_rows),
            training=False,
            inference=False,
            result_files={
                name: _file_hash(output / name) for name in result_names
            },
            completed_at=datetime.now(timezone.utc).isoformat(),
        )
        _atomic_json(complete_path, complete)
        _atomic_json(output / "collective_fragmentation_status.json", complete)
    except BaseException as exc:
        _atomic_json(
            output / "collective_fragmentation_status.json",
            dict(
                status="failed_or_interrupted",
                protocol_fingerprint=protocol["protocol_fingerprint"],
                completed_runs=completed,
                total_runs=len(records),
                error=repr(exc),
                updated_at=datetime.now(timezone.utc).isoformat(),
            ),
        )
        raise

    print(json.dumps(complete, ensure_ascii=False), flush=True)
    print(f"SUCCESS: collective fragmentation complete: {output}", flush=True)
    print("No training and no model inference were performed.", flush=True)
    return output

