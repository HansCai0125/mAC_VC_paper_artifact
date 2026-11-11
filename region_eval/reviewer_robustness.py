"""Read-only reviewer robustness analyses of completed VC prediction caches.

This module never loads model weights or performs inference.  The operating-point
analysis reads the already completed 19-threshold CSVs.  The ownership analysis
compares the published max-IoU rule with a max-intersection rule at theta=0.25.
"""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics

from .geometry import area, intersection, union_area


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "vc_v1" / "reviewer_robustness_v1"
FORMAL = ROOT / "outputs" / "vc_v1" / "formal_matrix_v1"
RTDETR = ROOT / "outputs" / "vc_v1" / "rtdetr_full_v1"
RTDETR_REPORT = ROOT / "outputs" / "vc_v1" / "paper_results_rtdetr_v1"
VERSION = "reviewer-robustness-1.0.0"
THETA = 0.25
THRESHOLDS = tuple(round(i * 0.05, 2) for i in range(1, 20))
SETTINGS = (
    ("mscoco", "yolov8n"),
    ("mscoco", "yolo11n"),
    ("mscoco", "yolo11s"),
    ("visdrone", "yolov8n"),
    ("visdrone", "yolo11n"),
    ("visdrone", "yolo11s"),
    ("visdrone", "rtdetr_l"),
)
N_IMAGES = {"mscoco": 5000, "visdrone": 548}
MODEL_LABEL = {"yolov8n": "YOLOv8n", "yolo11n": "YOLO11n",
               "yolo11s": "YOLO11s", "rtdetr_l": "RT-DETR-L"}


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _summary(values: list[float]) -> tuple[float, float]:
    if len(values) != 5:
        raise ValueError(f"expected five seeds, got {len(values)}")
    return statistics.fmean(values), statistics.stdev(values)


def run_dir(dataset: str, model: str, seed: int) -> Path:
    # YOLO11n seed 0 was reused in place by the formal matrix.
    if model == "yolo11n" and seed == 0:
        return ROOT / "outputs" / "vc_v1" / "full_seed0_v1" / dataset
    base = RTDETR if model == "rtdetr_l" else FORMAL
    return base / "runs" / dataset / model / f"seed_{seed}"


def preflight() -> dict:
    if _read_json(FORMAL / "matrix_complete.json").get("status") != "complete":
        raise ValueError("YOLO formal matrix is incomplete")
    if _read_json(RTDETR / "suite_complete.json").get("status") != "complete":
        raise ValueError("RT-DETR suite is incomplete")
    entries = []
    for dataset, model in SETTINGS:
        for seed in range(5):
            folder = run_dir(dataset, model, seed)
            cache = folder / "filtered_predictions.json"
            marker = _read_json(folder / "cache_complete.json")
            complete = _read_json(folder / "run_complete.json")
            analysis = _read_json(folder / "analysis" / "analysis_complete.json")
            if not cache.is_file() or marker.get("status") != "complete":
                raise ValueError(f"missing completed cache: {folder}")
            if complete.get("status") != "complete" or analysis.get("status") != "complete":
                raise ValueError(f"run or analysis incomplete: {folder}")
            if marker["filtered_predictions_sha256"] != analysis["source_sha256"]:
                raise ValueError(f"cache/analysis hash mismatch: {folder}")
            primary = complete["primary"]
            if primary["n_images"] != N_IMAGES[dataset] or primary["confidence_threshold"] != THETA:
                raise ValueError(f"unexpected validation set or threshold: {folder}")
            entries.append({"dataset": dataset, "model": model, "seed": seed,
                            "cache": str(cache), "cache_bytes": cache.stat().st_size,
                            "cache_sha256": marker["filtered_predictions_sha256"]})
    return {"status": "passed", "version": VERSION, "n_runs": len(entries),
            "n_settings": len(SETTINGS), "runs": entries,
            "note": "All analyses reuse existing complete-validation caches; no training or inference."}


def _threshold_rows() -> dict[tuple[str, str, int, float], dict]:
    sources = (
        (FORMAL / "matrix_thresholds.csv", False),
        (RTDETR_REPORT / "rtdetr_run_thresholds.csv", True),
    )
    result = {}
    for path, rtdetr_only in sources:
        for row in _read_csv(path):
            dataset, model, seed = row["dataset"], row["model"], int(row["seed"])
            if rtdetr_only and (dataset, model) != ("visdrone", "rtdetr_l"):
                continue
            if not rtdetr_only and (dataset, model) not in SETTINGS:
                continue
            theta = round(float(row["confidence_threshold"]), 2)
            key = (dataset, model, seed, theta)
            if key in result:
                raise ValueError(f"duplicate threshold row: {key}")
            if int(row["n_images"]) != N_IMAGES[dataset]:
                raise ValueError(f"wrong validation count: {key}")
            result[key] = row
    expected = {(d, m, s, t) for d, m in SETTINGS for s in range(5) for t in THRESHOLDS}
    if set(result) != expected:
        raise ValueError(f"threshold grid incomplete: missing={len(expected - set(result))}, "
                         f"unexpected={len(set(result) - expected)}")
    return result


def _ranks(values: dict[str, float]) -> dict[str, int]:
    return {model: 1 + sum(other > value for other in values.values())
            for model, value in values.items()}


def operating_point_analysis() -> tuple[list[dict], list[dict]]:
    rows = _threshold_rows()
    per_seed, summary = [], []
    for dataset, model in SETTINGS:
        f1_by_theta = {
            theta: statistics.fmean(float(rows[(dataset, model, seed, theta)]["standard_F1"])
                                    for seed in range(5))
            for theta in THRESHOLDS
        }
        # On an exact tie use the smaller threshold; the rule is fixed in advance.
        selected = min(THRESHOLDS, key=lambda t: (-f1_by_theta[t], t))
        for seed in range(5):
            fixed = rows[(dataset, model, seed, THETA)]
            chosen = rows[(dataset, model, seed, selected)]
            per_seed.append({"dataset": dataset, "model": model, "seed": seed,
                             "theta_F1": selected, "theta_reference": THETA,
                             **{f"{name}_{point}": float(source[column])
                                for point, source in (("reference", fixed), ("F1_opt", chosen))
                                for name, column in (("F1", "standard_F1"),
                                                     ("mAC", "mAC"), ("VC", "VC"))}})
        group = [r for r in per_seed if (r["dataset"], r["model"]) == (dataset, model)]
        item = {"dataset": dataset, "model": model, "n_seeds": 5,
                "theta_F1": selected, "mean_F1_at_selected_theta": f1_by_theta[selected]}
        for metric in ("F1", "mAC", "VC"):
            for point in ("reference", "F1_opt"):
                mean, sd = _summary([r[f"{metric}_{point}"] for r in group])
                item[f"{metric}_{point}_mean"] = mean
                item[f"{metric}_{point}_sd"] = sd
        summary.append(item)
    for dataset in N_IMAGES:
        group = [r for r in summary if r["dataset"] == dataset]
        for metric in ("F1", "mAC", "VC"):
            fixed_ranks = _ranks({r["model"]: r[f"{metric}_reference_mean"] for r in group})
            selected_ranks = _ranks({r["model"]: r[f"{metric}_F1_opt_mean"] for r in group})
            for row in group:
                model = row["model"]
                row[f"{metric}_rank_reference"] = fixed_ranks[model]
                row[f"{metric}_rank_F1_opt"] = selected_ranks[model]
                row[f"{metric}_rank_changed"] = fixed_ranks[model] != selected_ranks[model]
    return per_seed, summary


def _local_dice(gt_rect: tuple, owned_rects: list[tuple]) -> float:
    if not owned_rects:
        return 0.0
    support = union_area(owned_rects)
    covered = union_area([hit for rect in owned_rects
                          if (hit := intersection(gt_rect, rect)) is not None])
    return 2.0 * covered / (support + area(gt_rect))


def evaluate_owner_rules(document: dict, *, theta: float = THETA) -> dict:
    """Recompute both VC variants from identical retained predictions and GTs."""
    local_sums = {"max_iou": 0.0, "max_intersection": 0.0}
    n_gt = n_pred = n_orphan = n_ambiguous = n_changed = n_assigned = 0
    for image in document["images"]:
        by_gt, by_pred = defaultdict(list), defaultdict(list)
        for gt in image["gts"]:
            by_gt[gt["class_id"]].append(gt)
            n_gt += 1
        for pred in image["predictions"]:
            if pred["score"] >= theta:
                by_pred[pred["class_id"]].append(pred)
                n_pred += 1
        for class_id in set(by_gt) | set(by_pred):
            gts, preds = by_gt[class_id], by_pred[class_id]
            owned = {rule: defaultdict(list) for rule in local_sums}
            for pred in preds:
                pred_rect = tuple(pred["xyxy"])
                pred_area = area(pred_rect)
                hits = []
                for gt in gts:
                    overlap = intersection(pred_rect, tuple(gt["xyxy"]))
                    if overlap is None:
                        continue
                    overlap_area = area(overlap)
                    iou = overlap_area / (pred_area + area(tuple(gt["xyxy"])) - overlap_area)
                    hits.append((gt["id"], overlap_area, iou))
                if not hits:
                    n_orphan += 1
                    continue
                n_assigned += 1
                n_ambiguous += len(hits) > 1
                owner_iou = min(hits, key=lambda h: (-h[2], -h[1], h[0]))[0]
                owner_intersection = min(hits, key=lambda h: (-h[1], -h[2], h[0]))[0]
                n_changed += owner_iou != owner_intersection
                owned["max_iou"][owner_iou].append(pred_rect)
                owned["max_intersection"][owner_intersection].append(pred_rect)
            for gt in gts:
                gt_rect = tuple(gt["xyxy"])
                for rule in local_sums:
                    local_sums[rule] += _local_dice(gt_rect, owned[rule][gt["id"]])
    denominator = n_gt + n_orphan
    if n_assigned + n_orphan != n_pred or n_changed > n_ambiguous:
        raise AssertionError("ownership accounting invariant failed")
    return {"n_images": len(document["images"]), "n_gt": n_gt, "n_pred": n_pred,
            "n_assigned": n_assigned, "n_orphan": n_orphan,
            "n_ambiguous": n_ambiguous, "n_owner_changed": n_changed,
            "owner_change_rate_ambiguous": n_changed / n_ambiguous if n_ambiguous else 0.0,
            "owner_change_rate_assigned": n_changed / n_assigned if n_assigned else 0.0,
            "VC_max_iou": local_sums["max_iou"] / denominator if denominator else None,
            "VC_max_intersection": local_sums["max_intersection"] / denominator
            if denominator else None}


def _ownership_one(entry: dict) -> dict:
    dataset, model, seed = entry["dataset"], entry["model"], entry["seed"]
    folder = run_dir(dataset, model, seed)
    with (folder / "filtered_predictions.json").open(encoding="utf-8") as stream:
        document = json.load(stream)
    observed = evaluate_owner_rules(document)
    if observed["n_images"] != N_IMAGES[dataset]:
        raise AssertionError(f"wrong image count: {dataset}/{model}/seed{seed}")
    expected = _read_json(folder / "run_complete.json")["primary"]
    for actual_key, baseline_key in (("n_gt", "n_gt"), ("n_pred", "n_pred"),
                                     ("n_assigned", "assigned_pred_count"),
                                     ("n_orphan", "orphan_pred_count")):
        if observed[actual_key] != expected[baseline_key]:
            raise AssertionError(f"{actual_key} disagrees with published VC run: {folder}")
    if abs(observed["VC_max_iou"] - expected["VC"]) > 1e-9:
        raise AssertionError(f"max-IoU VC disagrees with published VC run: {folder}; "
                             f"observed={observed['VC_max_iou']}, expected={expected['VC']}")
    observed.update({"dataset": dataset, "model": model, "seed": seed,
                     "theta": THETA, "source_sha256": entry["cache_sha256"],
                     "source_bytes": entry["cache_bytes"],
                     "delta_VC": observed["VC_max_intersection"] - observed["VC_max_iou"],
                     "absolute_delta_VC": abs(observed["VC_max_intersection"]
                                              - observed["VC_max_iou"])})
    return observed


def _ownership_summary(per_run: list[dict]) -> list[dict]:
    result = []
    for dataset, model in SETTINGS:
        group = [r for r in per_run if (r["dataset"], r["model"]) == (dataset, model)]
        if len(group) != 5 or {r["seed"] for r in group} != set(range(5)):
            raise ValueError(f"incomplete ownership group: {dataset}/{model}")
        item = {"dataset": dataset, "model": model, "n_seeds": 5}
        for field in ("VC_max_iou", "VC_max_intersection", "delta_VC",
                      "absolute_delta_VC", "owner_change_rate_ambiguous",
                      "owner_change_rate_assigned"):
            item[f"{field}_mean"], item[f"{field}_sd"] = _summary([r[field] for r in group])
        item["absolute_delta_VC_max"] = max(r["absolute_delta_VC"] for r in group)
        item["n_owner_changed_total"] = sum(r["n_owner_changed"] for r in group)
        item["n_ambiguous_total"] = sum(r["n_ambiguous"] for r in group)
        item["orphan_count_invariant"] = True
        result.append(item)
    for dataset in N_IMAGES:
        group = [r for r in result if r["dataset"] == dataset]
        fixed = _ranks({r["model"]: r["VC_max_iou_mean"] for r in group})
        other = _ranks({r["model"]: r["VC_max_intersection_mean"] for r in group})
        for row in group:
            row["VC_rank_max_iou"] = fixed[row["model"]]
            row["VC_rank_max_intersection"] = other[row["model"]]
            row["VC_rank_changed"] = fixed[row["model"]] != other[row["model"]]
    return result


def _format_pm(row: dict, name: str) -> str:
    return f"{row[name + '_mean']:.4f} $\\pm$ {row[name + '_sd']:.4f}"


def _latex_row(*cells: object) -> str:
    return " & ".join(str(cell) for cell in cells) + r" \\"


def _rank_changes(row: dict) -> str:
    changed = [metric for metric in ("F1", "mAC", "VC")
               if row[f"{metric}_rank_changed"]]
    return ", ".join(changed) if changed else "none"


def _write_tables(output: Path, ownership: list[dict] | None,
                  operating: list[dict]) -> None:
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             _latex_row("Dataset", "Model", r"$\theta_{F1}$", "F1",
                        r"$mAC$", "VC", "Ranks changed?"), r"\midrule"]
    for row in operating:
        lines.append(_latex_row(
            row["dataset"], MODEL_LABEL[row["model"]],
            f'{row["theta_F1"]:.2f}',
            _format_pm(row, "F1_F1_opt"),
            _format_pm(row, "mAC_F1_opt"),
            _format_pm(row, "VC_F1_opt"),
            _rank_changes(row),
        ))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (output / "operating_point_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    if ownership is None:
        return
    lines = [r"\begin{tabular}{llccccc}", r"\toprule",
             _latex_row("Dataset", "Model", "VC (IoU)",
                        "VC (intersection)", r"$|\Delta VC|$",
                        "Owner changed", "Rank changed?"), r"\midrule"]
    for row in ownership:
        lines.append(_latex_row(
            row["dataset"], MODEL_LABEL[row["model"]],
            _format_pm(row, "VC_max_iou"),
            _format_pm(row, "VC_max_intersection"),
            f'{row["absolute_delta_VC_mean"]:.4f}',
            f'{100 * row["owner_change_rate_ambiguous_mean"]:.2f}\%',
            "yes" if row["VC_rank_changed"] else "no",
        ))
    lines += [r"\bottomrule", r"\end{tabular}"]
    (output / "ownership_table.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(*, preflight_only: bool = False, operating_only: bool = False,
                 limit_ownership_runs: int | None = None) -> Path:
    plan = preflight()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT / "preflight.json", plan)
    if preflight_only:
        return OUTPUT
    per_seed, operating = operating_point_analysis()
    _write_csv(OUTPUT / "operating_point_per_seed.csv", per_seed, tuple(per_seed[0]))
    _write_csv(OUTPUT / "operating_point_summary.csv", operating, tuple(operating[0]))
    _write_tables(OUTPUT, None, operating)
    if operating_only:
        return OUTPUT
    files = []
    completed = OUTPUT / "ownership_runs"
    completed.mkdir(exist_ok=True)
    for index, entry in enumerate(plan["runs"], 1):
        name = f"{entry['dataset']}_{entry['model']}_seed_{entry['seed']}.json"
        path = completed / name
        if path.exists():
            row = _read_json(path)
            if row.get("version") != VERSION or row.get("source_sha256") != entry["cache_sha256"]:
                raise ValueError(f"stale ownership result, choose a fresh output directory: {path}")
            print(f"OWNERSHIP {index:02d}/35 REUSED {name}", flush=True)
        else:
            print(f"OWNERSHIP {index:02d}/35 START {name}", flush=True)
            row = _ownership_one(entry)
            row["version"] = VERSION
            _write_json(path, row)
            print(f"OWNERSHIP {index:02d}/35 COMPLETE VC {row['VC_max_iou']:.6f} -> "
                  f"{row['VC_max_intersection']:.6f}", flush=True)
        files.append(row)
        if limit_ownership_runs is not None and index >= limit_ownership_runs:
            return OUTPUT
    ownership = _ownership_summary(files)
    _write_csv(OUTPUT / "ownership_per_seed.csv", files, tuple(files[0]))
    _write_csv(OUTPUT / "ownership_summary.csv", ownership, tuple(ownership[0]))
    _write_tables(OUTPUT, ownership, operating)
    _write_json(OUTPUT / "analysis_complete.json", {
        "status": "complete", "version": VERSION, "n_ownership_runs": len(files),
        "n_settings": len(SETTINGS), "theta": THETA,
        "owner_change_rate_denominator": "predictions with >=2 positive-overlap same-class candidate GTs",
        "operating_point_selection": "highest mean F1 across five seeds on the 19-point validation sweep; lower theta breaks exact ties",
        "selection_interpretation": "post-hoc descriptive robustness on the same validation split, not an unbiased retuned benchmark",
        "no_training_or_inference": True,
        "ownership_rank_changed": [f"{r['dataset']}:{r['model']}" for r in ownership
                                   if r["VC_rank_changed"]],
        "operating_rank_changed": {metric: [f"{r['dataset']}:{r['model']}" for r in operating
                                             if r[f"{metric}_rank_changed"]]
                                   for metric in ("F1", "mAC", "VC")},
    })
    return OUTPUT
