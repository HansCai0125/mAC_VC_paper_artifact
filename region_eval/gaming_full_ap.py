"""Complete the formal anti-gaming grid with diagnostic AP at every level.

This module reuses the frozen post-NMS prediction caches and the already
completed formal anti-gaming rows.  It computes only the AP values that were
left blank by the sparse formal protocol, merges them with the audited rows,
and writes a separate, resumable result tree.  It never trains a detector or
runs model inference.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from uuid import uuid4

from .case_analysis import _file_hash
from .gaming import load_shard_sample, run_gaming_grid, write_gaming_report
from .gaming_formal import (
    DEFAULT_OUTPUT as FORMAL_OUTPUT,
    DUPLICATE_FACTORS,
    EXPECTED_MATRIX_FINGERPRINT,
    FIXED_ANALYSIS_ROOT,
    MATRIX_ROOT,
    METRIC_FIELDS,
    SHRINK_SCALES,
    EXPAND_SCALES,
    _aggregate,
    _atomic_csv,
    _atomic_json,
    _atomic_text,
    _completed_run_matches,
    _fingerprint,
    _read_csv,
    _read_json,
    _read_run_rows,
    _source_records,
)
from .offline import PROJECT_ROOT
from .vc_v1 import PRIMARY_THETA, VC_VERSION


FULL_AP_VERSION = "vc-v1-full-diagnostic-ap-0.1.0"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "vc_v1" / "gaming_full_ap_v1"

AP_FIELDS = (
    "AP50",
    "AP75",
    "AP50_95",
    "AR50_95",
    "normalized_AP50",
    "normalized_AP50_95",
)

TEXT_FIELDS = {"manipulation", "variant", "level_label"}
COMMON_FIELDS = {"dataset", "model_id", "source_run_fingerprint"}
COMPARE_FIELDS = tuple(
    field
    for field in METRIC_FIELDS
    if field not in AP_FIELDS
) + (
    "primary_theta",
    "baseline_tp_count",
    "modified_prediction_count",
    "added_prediction_count",
    "theta",
)


def _run_key(row):
    return (
        row["manipulation"],
        row["variant"],
        float(row["level"]),
    )


def _parse_formal_run(path):
    rows = []
    for source in _read_csv(path):
        row = {}
        for key, value in source.items():
            if key in COMMON_FIELDS:
                continue
            if key in TEXT_FIELDS:
                row[key] = value
            else:
                row[key] = None if value in (None, "", "NA") else float(value)
        rows.append(row)
    return rows


def _all_attack_levels():
    values = [("baseline", "unmodified", 1.0)]
    values.extend(("shrink", "center_scale", value) for value in SHRINK_SCALES)
    values.extend(("expand", "center_scale", value) for value in EXPAND_SCALES)
    for variant in (
        "exact",
        "controlled_jitter_2pct_center_1pct_size",
    ):
        values.extend(
            ("duplicate", variant, value) for value in DUPLICATE_FACTORS
        )
    return tuple(values)


def _formal_run_directory(formal_output, record):
    return (
        formal_output
        / "runs"
        / record["dataset"]
        / record["model"]
        / f"seed_{int(record['seed'])}"
    )


def _supplement_protocol(formal_output, formal_protocol, records):
    value = dict(
        full_ap_version=FULL_AP_VERSION,
        vc_version=VC_VERSION,
        purpose=(
            "complete diagnostic mAP50:95 response curves for the formal "
            "controlled anti-gaming experiment"
        ),
        source_matrix_fingerprint=EXPECTED_MATRIX_FINGERPRINT,
        source_formal_output=str(formal_output),
        source_formal_protocol_fingerprint=formal_protocol[
            "protocol_fingerprint"
        ],
        source_code={
            "gaming.py": _file_hash(Path(__file__).with_name("gaming.py")),
            "gaming_formal.py": _file_hash(
                Path(__file__).with_name("gaming_formal.py")
            ),
            "gaming_full_ap.py": _file_hash(Path(__file__)),
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
        primary_theta=PRIMARY_THETA,
        attack_grids=dict(
            shrink_scales=list(SHRINK_SCALES),
            expand_scales=list(EXPAND_SCALES),
            duplicate_factors=list(DUPLICATE_FACTORS),
        ),
        diagnostic_ap_levels=[list(row) for row in _all_attack_levels()],
        diagnostic_ap_role=(
            "diagnostic AP under the same fixed theta and manipulated "
            "post-NMS predictions; not official dataset AP"
        ),
        completion_strategy=(
            "reuse audited formal rows and recompute only missing AP levels, "
            "plus the baseline AP needed for normalization"
        ),
        training=False,
        inference=False,
    )
    value["protocol_fingerprint"] = _fingerprint(value)
    return value


def _validate_source(formal_output, records):
    complete = _read_json(formal_output / "gaming_formal_complete.json")
    protocol = _read_json(formal_output / "protocol.json")
    if complete.get("status") != "complete":
        raise ValueError("formal gaming analysis is not complete")
    if (
        complete.get("protocol_fingerprint")
        != protocol.get("protocol_fingerprint")
    ):
        raise ValueError("formal gaming completion and protocol disagree")
    for record in records:
        run_directory = _formal_run_directory(formal_output, record)
        if not _completed_run_matches(run_directory, record, protocol):
            raise ValueError(f"formal gaming run changed: {run_directory}")
    return protocol


def _missing_ap_levels(rows):
    missing = {
        _run_key(row)
        for row in rows
        if any(row.get(field) is None for field in AP_FIELDS[:4])
    }
    # Baseline AP supplies the denominator for every normalized AP value.
    missing.add(("baseline", "unmodified", 1.0))
    return missing


def _same_number(left, right, tolerance=1e-12):
    if left is None or right is None:
        return left is right
    scale = max(1.0, abs(float(left)), abs(float(right)))
    return abs(float(left) - float(right)) <= tolerance * scale


def _merge_rows(formal_rows, recomputed_rows):
    formal = {_run_key(row): row for row in formal_rows}
    recomputed = {_run_key(row): row for row in recomputed_rows}
    if len(formal) != 30 or set(formal) != set(recomputed):
        raise ValueError("formal and recomputed attack grids do not match")

    merged = []
    for key in formal:
        old = formal[key]
        new = recomputed[key]
        for field in COMPARE_FIELDS:
            if not _same_number(old.get(field), new.get(field)):
                raise ValueError(
                    f"non-AP metric changed at {key}: {field} "
                    f"{old.get(field)} != {new.get(field)}"
                )
        row = dict(old)
        for field in AP_FIELDS:
            if row.get(field) is None:
                row[field] = new.get(field)
            elif new.get(field) is not None and not _same_number(
                row[field], new[field]
            ):
                raise ValueError(
                    f"existing AP changed at {key}: {field} "
                    f"{row[field]} != {new[field]}"
                )
        if any(row.get(field) is None for field in AP_FIELDS):
            raise ValueError(f"AP completion failed at {key}")
        merged.append(row)
    return merged


def preflight_full_ap(
    *,
    matrix_root=MATRIX_ROOT,
    formal_output=FORMAL_OUTPUT,
):
    matrix_root = Path(matrix_root).resolve()
    formal_output = Path(formal_output).resolve()
    records = _source_records(matrix_root)
    formal_protocol = _validate_source(formal_output, records)
    missing_rows = 0
    complete_rows = 0
    for record in records:
        rows = _parse_formal_run(
            _formal_run_directory(formal_output, record) / "gaming_metrics.csv"
        )
        if len(rows) != 30:
            raise ValueError("each formal gaming run must contain 30 rows")
        for row in rows:
            if row["AP50_95"] is None:
                missing_rows += 1
            else:
                complete_rows += 1
    summary = dict(
        status="passed",
        source_formal_protocol_fingerprint=formal_protocol[
            "protocol_fingerprint"
        ],
        runs=len(records),
        rows=len(records) * 30,
        existing_ap_rows=complete_rows,
        missing_ap_rows=missing_rows,
        training=False,
        inference=False,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print(
        "PREFLIGHT SUCCESS: cached predictions can supply a connected "
        "diagnostic mAP curve without training or inference.",
        flush=True,
    )
    return summary


def _summary_markdown(protocol, raw_rows, aggregate_rows):
    return f"""# Full diagnostic mAP anti-gaming supplement

This output fills every missing AP50, AP75, mAP50:95, and AR50:95 value in
the completed formal anti-gaming grid. It reuses the frozen post-NMS
prediction caches and does not train a model or run model inference.

The mAP values are diagnostic measurements at the fixed confidence threshold
`theta={PRIMARY_THETA}` after each controlled box manipulation. They support
continuous anti-gaming response curves, but they are distinct from official
dataset AP computed from the detector's complete confidence ranking.

- Source formal protocol: `{protocol['source_formal_protocol_fingerprint']}`
- Completed source runs: 18
- Per-run attack rows: {len(raw_rows)}
- Aggregate rows: {len(aggregate_rows)}
- Output protocol: `{protocol['protocol_fingerprint']}`
"""


def run_full_ap_supplement(
    *,
    matrix_root=MATRIX_ROOT,
    formal_output=FORMAL_OUTPUT,
    output=DEFAULT_OUTPUT,
):
    """Run or resume AP completion for all formal anti-gaming severities."""
    matrix_root = Path(matrix_root).resolve()
    formal_output = Path(formal_output).resolve()
    output = Path(output).resolve()
    records = _source_records(matrix_root)
    formal_protocol = _validate_source(formal_output, records)
    protocol = _supplement_protocol(formal_output, formal_protocol, records)
    complete_path = output / "gaming_full_ap_complete.json"

    if complete_path.is_file():
        complete = _read_json(complete_path)
        if complete.get("protocol_fingerprint") != protocol[
            "protocol_fingerprint"
        ]:
            raise ValueError("completed full-AP output uses another protocol")
        print(f"SKIP: full diagnostic AP already complete: {output}", flush=True)
        return output

    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.is_file():
        previous = _read_json(protocol_path)
        if previous.get("protocol_fingerprint") != protocol[
            "protocol_fingerprint"
        ]:
            if list(output.glob("runs/**/gaming_complete.json")):
                raise ValueError(
                    "partial full-AP output uses another protocol; choose a "
                    "fresh output directory"
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
                    f"FULL AP {index:02d}/{len(records)} SKIP: "
                    f"{dataset}:{model}:seed{seed}",
                    flush=True,
                )
                continue
            if final.exists():
                shutil.rmtree(final)
            final.parent.mkdir(parents=True, exist_ok=True)
            temporary = final.with_name(f".{final.name}.{uuid4().hex}.tmp")
            formal_run = _formal_run_directory(formal_output, record)
            formal_rows = _parse_formal_run(
                formal_run / "gaming_metrics.csv"
            )
            missing = _missing_ap_levels(formal_rows)
            print(
                f"FULL AP {index:02d}/{len(records)} START: "
                f"{dataset}:{model}:seed{seed}; "
                f"AP evaluations={len(missing)}",
                flush=True,
            )
            try:
                document, source = load_shard_sample(
                    Path(record["output"]),
                    int(record["n_images"]),
                    minimum_score=PRIMARY_THETA,
                )
                recomputed_rows, audit = run_gaming_grid(
                    document,
                    shrink_scales=SHRINK_SCALES,
                    expand_scales=EXPAND_SCALES,
                    duplicate_factors=DUPLICATE_FACTORS,
                    threshold_grid=(),
                    primary_theta=PRIMARY_THETA,
                    diagnostic_ap_levels=missing,
                )
                merged_rows = _merge_rows(formal_rows, recomputed_rows)
                write_gaming_report(
                    temporary,
                    document=document,
                    source=source,
                    rows=merged_rows,
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
                output / "gaming_full_ap_status.json",
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
                f"FULL AP {index:02d}/{len(records)} COMPLETE: {final}",
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
                raise ValueError(f"completed full-AP run changed: {final}")
            raw_rows.extend(_read_run_rows(final, record))
        if len(raw_rows) != 540:
            raise ValueError(f"expected 540 per-run rows, found {len(raw_rows)}")
        if any(row[field] is None for row in raw_rows for field in AP_FIELDS):
            raise ValueError("full-AP table still contains blank AP values")
        aggregate_rows = _aggregate(raw_rows)
        if len(aggregate_rows) != 180:
            raise ValueError(
                f"expected 180 aggregate rows, found {len(aggregate_rows)}"
            )
        _atomic_csv(output / "gaming_runs.csv", raw_rows)
        _atomic_csv(output / "gaming_aggregate.csv", aggregate_rows)
        _atomic_text(
            output / "README.md",
            _summary_markdown(protocol, raw_rows, aggregate_rows),
        )
        result_names = (
            "protocol.json",
            "gaming_runs.csv",
            "gaming_aggregate.csv",
            "README.md",
        )
        complete = dict(
            status="complete",
            full_ap_version=FULL_AP_VERSION,
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
        _atomic_json(output / "gaming_full_ap_status.json", complete)
    except BaseException as exc:
        _atomic_json(
            output / "gaming_full_ap_status.json",
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
    print(f"SUCCESS: full diagnostic AP complete: {output}", flush=True)
    print("No training and no model inference were performed.", flush=True)
    return output

