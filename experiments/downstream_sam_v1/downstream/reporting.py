"""Seed-level endpoints and paired image bootstrap for the SAM utility study.

No threshold selection is performed here. Inputs must contain held-out images
under already-frozen calibration choices. A seed's class counts are aggregated
over images before class IoUs are averaged; different seeds are never pooled.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


POLICIES = ("fixed_025", "F1", "LRP", "BestIoGT", "mAC", "VC")
BASELINES = ("F1", "LRP", "BestIoGT", "mAC")
METRICS = ("mIoU", "foreground_precision", "foreground_recall", "mean_prompts")
BOOTSTRAP_SCOPE = (
    "Paired image-level percentile 95% intervals for differences of mean seed "
    "mIoU. The same resampled image multiplicities are used for all seeds and "
    "policies. Calibration choices, fitted detectors, and SAM are held fixed; "
    "these intervals do not quantify recalibration or retraining uncertainty."
)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty report: {path.name}")
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def scores_from_counts(counts: np.ndarray) -> dict[str, float | None]:
    """Compute endpoints from pooled (classes, TP/FP/FN) integer counts.

    Classes with positive pooled union participate in macro IoU, including
    false-positive-only classes. A class absent in both prediction and GT is
    excluded. Pixel precision/recall are micro averages across classes.
    """
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != 3:
        raise ValueError("Pooled counts must have shape (n_classes, 3)")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError("Pixel counts must be finite and nonnegative")
    values = counts.astype(np.float64, copy=False)
    union = values.sum(axis=1)
    present = union > 0
    miou = float(np.mean(values[present, 0] / union[present])) if present.any() else None
    tp, fp, fn = values.sum(axis=0)
    return {
        "mIoU": miou,
        "foreground_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "foreground_recall": float(tp / (tp + fn)) if tp + fn else None,
    }


def bootstrap_miou(counts: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return (draws, configurations) macro IoU with shared image weights.

    Processing only images active for a class avoids a large dense contraction
    over the mostly-zero image-by-class tensor. This is exactly weighted pixel
    aggregation, not a bootstrap of per-image IoUs.
    """
    counts = np.asarray(counts)
    weights = np.asarray(weights, dtype=np.float64)
    if counts.ndim != 4 or counts.shape[-1] != 3:
        raise ValueError("Counts must have shape (images, configurations, classes, 3)")
    if weights.ndim != 2 or weights.shape[1] != counts.shape[0]:
        raise ValueError("Bootstrap weights must align with the image dimension")
    if np.any(weights < 0) or not np.all(np.isfinite(weights)):
        raise ValueError("Invalid bootstrap weights")
    n_draws, n_configs = weights.shape[0], counts.shape[1]
    iou_sums = np.zeros((n_draws, n_configs), dtype=np.float64)
    n_present = np.zeros((n_draws, n_configs), dtype=np.int32)
    for class_index in range(counts.shape[2]):
        current = counts[:, :, class_index, :]
        active = np.flatnonzero(np.any(current != 0, axis=(1, 2)))
        if not len(active):
            continue
        matrix = current[active].reshape(len(active), n_configs * 3).astype(np.float64)
        pooled = (weights[:, active] @ matrix).reshape(n_draws, n_configs, 3)
        union = pooled.sum(axis=2)
        present = union > 0
        ious = np.divide(pooled[:, :, 0], union, out=np.zeros_like(union), where=present)
        iou_sums += ious
        n_present += present
    return np.divide(
        iou_sums, n_present, out=np.full_like(iou_sums, np.nan), where=n_present > 0
    )


def _load_run(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"image_ids", "counts", "prompt_counts", "config_json"}
        if not required.issubset(archive.files):
            raise ValueError(f"Missing NPZ fields in {path}: {required - set(archive.files)}")
        image_ids = archive["image_ids"]
        counts = archive["counts"]
        prompts = archive["prompt_counts"]
        encoded = archive["config_json"]
        if encoded.ndim != 0 or encoded.dtype.kind not in "US":
            raise ValueError(f"config_json must be a scalar string: {path}")
        configs = json.loads(str(encoded.item()))
    if image_ids.ndim != 1 or not len(image_ids) or image_ids.dtype.kind not in "iu":
        raise ValueError(f"image_ids must be nonempty integer IDs: {path}")
    if len(np.unique(image_ids)) != len(image_ids):
        raise ValueError(f"Duplicate image IDs: {path}")
    if counts.ndim != 4 or counts.shape[0] != len(image_ids) or counts.shape[-1] != 3:
        raise ValueError(f"Invalid counts dimensions: {path}")
    if counts.shape[2] < 1 or counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ValueError(f"Counts must be nonnegative integers with at least one class: {path}")
    if prompts.shape != counts.shape[:2] or prompts.dtype.kind not in "iu" or np.any(prompts < 0):
        raise ValueError(f"Invalid prompt_counts: {path}")
    targets = counts[:, :, :, 0] + counts[:, :, :, 2]
    if not np.array_equal(targets, np.broadcast_to(targets[:, :1, :], targets.shape)):
        raise ValueError(f"Ground-truth pixel totals vary across selection policies: {path}")
    if not isinstance(configs, list) or len(configs) != counts.shape[1] or not configs:
        raise ValueError(f"Configurations do not align with counts: {path}")
    run_identity = None
    keys = set()
    for slot, config in enumerate(configs):
        fields = {"run_id", "model", "seed", "budget", "policy", "theta"}
        if not isinstance(config, dict) or not fields.issubset(config):
            raise ValueError(f"Incomplete configuration in {path}")
        if not isinstance(config["run_id"], str) or not isinstance(config["model"], str):
            raise ValueError(f"Invalid run identity in {path}")
        if type(config["seed"]) is not int or type(config["budget"]) is not int or config["budget"] < 1:
            raise ValueError(f"Invalid seed or budget in {path}")
        theta = config["theta"]
        if isinstance(theta, bool) or not isinstance(theta, (int, float)) or not np.isfinite(theta) or not 0 <= theta <= 1:
            raise ValueError(f"Invalid threshold in {path}")
        if config["policy"] not in POLICIES:
            raise ValueError(f"Unknown selection policy in {path}")
        identity = (config["run_id"], config["model"], config["seed"])
        if run_identity is None:
            run_identity = identity
        if identity != run_identity:
            raise ValueError(f"NPZ contains more than one run: {path}")
        key = (config["budget"], config["policy"])
        if key in keys:
            raise ValueError(f"Duplicate policy/budget in {path}: {key}")
        keys.add(key)
        if np.any(prompts[:, slot] > config["budget"]):
            raise ValueError(f"Prompt counts exceed their budget in {path}")
    budgets = sorted({budget for budget, _ in keys})
    expected = {(budget, policy) for budget in budgets for policy in POLICIES}
    if keys != expected:
        raise ValueError(f"Each budget requires all six selection policies: {path}")
    return image_ids, counts, prompts, configs


def _mean_sd(values: list[float]) -> tuple[float, float | None]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else None


def _plots(output_dir: Path, rows: list[dict]) -> dict:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        return {"status": "unavailable", "reason": str(exc), "files": []}
    models = list(dict.fromkeys(row["model"] for row in rows))
    colors = {
        "fixed_025": "#777777", "F1": "#009E73", "LRP": "#CC79A7",
        "BestIoGT": "#56B4E9", "mAC": "#0072B2", "VC": "#D55E00",
    }
    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42}):
        figure, axes = plt.subplots(1, len(models), figsize=(4.1 * len(models), 3.9), squeeze=False)
        for column, model in enumerate(models):
            axis = axes[0, column]
            for policy in POLICIES:
                selected = sorted(
                    (row for row in rows if row["model"] == model and row["policy"] == policy),
                    key=lambda row: row["budget"],
                )
                axis.errorbar(
                    [row["budget"] for row in selected],
                    [row["mIoU_mean"] for row in selected],
                    yerr=[row["mIoU_std"] or 0.0 for row in selected],
                    color=colors[policy], marker="o", capsize=2, linewidth=1.6,
                    label="Fixed 0.25" if policy == "fixed_025" else policy,
                )
            axis.set_title(model)
            axis.set_xlabel("Maximum prompts per image")
            axis.set_xticks(sorted({row["budget"] for row in rows if row["model"] == model}))
            axis.set_ylabel("Held-out foreground mIoU")
            axis.grid(axis="y", alpha=0.25)
            axis.spines[["top", "right"]].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=6, frameon=False)
        figure.text(0.5, 0.015, "Error bars: sample SD across detector seeds", ha="center", fontsize=9)
        figure.tight_layout(rect=(0, 0.055, 1, 0.88))
        paths = [output_dir / "budget_vs_miou.png", output_dir / "budget_vs_miou.pdf"]
        for path in paths:
            figure.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(figure)
    return {"status": "complete", "files": [str(path) for path in paths]}


def build_reports(
    run_files: list[Path],
    output_dir: Path,
    bootstrap_reps: int = 2000,
    bootstrap_seed: int = 20261026,
    mask_ap_rows: list[dict] | None = None,
) -> dict:
    """Create reports after all runs have produced aligned held-out NPZ files.

    Bootstrap intervals describe image uncertainty conditional on fixed choices.
    Sample SD is computed separately across the observed detector seed runs.
    """
    if not run_files:
        raise ValueError("No completed per-run files supplied")
    if type(bootstrap_reps) is not int or bootstrap_reps < 2:
        raise ValueError("bootstrap_reps must be an integer >= 2")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    run_files = [Path(path).resolve() for path in run_files]
    if len(set(run_files)) != len(run_files):
        raise ValueError("The same per-run file was supplied twice")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    complete_file = output_dir / "report_complete.json"
    # A previous completion receipt must not falsely describe a failed rebuild.
    if complete_file.exists():
        previous = output_dir / "report_previous_complete.json"
        complete_file.replace(previous)

    ap_lookup: dict[tuple, dict] = {}
    for row in mask_ap_rows or []:
        key = (row["run_id"], row["budget"], row["policy"])
        if key in ap_lookup:
            raise ValueError(f"Duplicate mask AP entry: {key}")
        for metric in ("mask_AP50_95", "mask_AP50"):
            value = row.get(metric)
            if value is not None and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"Invalid {metric} for {key}")
        ap_lookup[key] = row

    reference_ids = None
    reference_settings = None
    reference_class_count = None
    reference_targets = None
    weights = None
    per_run = []
    used_ap = set()
    seen_run_ids, seen_seeds = set(), set()
    boot_by_group: dict[tuple, list[np.ndarray]] = {}
    for run_index, path in enumerate(run_files, 1):
        image_ids, counts, prompt_counts, configs = _load_run(path)
        identity = configs[0]
        run_id, model, seed = identity["run_id"], identity["model"], identity["seed"]
        if run_id in seen_run_ids or (model, seed) in seen_seeds:
            raise ValueError(f"Duplicate run identity or model/seed: {run_id}")
        seen_run_ids.add(run_id)
        seen_seeds.add((model, seed))
        settings = {(row["budget"], row["policy"]) for row in configs}
        if reference_ids is None:
            reference_ids = image_ids.copy()
            reference_settings = settings
            reference_class_count = counts.shape[2]
            reference_targets = counts[:, 0, :, 0] + counts[:, 0, :, 2]
            generator = np.random.default_rng(bootstrap_seed)
            probabilities = np.full(len(image_ids), 1.0 / len(image_ids))
            weights = generator.multinomial(
                len(image_ids), probabilities, size=bootstrap_reps
            ).astype(np.float64)
        elif not np.array_equal(image_ids, reference_ids):
            raise ValueError(f"Held-out image IDs/order differ across runs: {path}")
        elif settings != reference_settings or counts.shape[2] != reference_class_count:
            raise ValueError(f"Budget/policy/class coverage differs across runs: {path}")
        elif not np.array_equal(counts[:, 0, :, 0] + counts[:, 0, :, 2], reference_targets):
            raise ValueError(f"Ground-truth pixel totals differ across runs: {path}")
        print(f"REPORT {run_index:02d}/{len(run_files):02d}: {run_id}; paired image bootstrap", flush=True)
        pooled = counts.sum(axis=0, dtype=np.int64)
        boot_scores = bootstrap_miou(counts, weights)
        for slot, config in enumerate(configs):
            endpoints = scores_from_counts(pooled[slot])
            if endpoints["mIoU"] is None:
                raise ValueError(f"No evaluable union for {run_id}, {config['policy']}")
            key = (run_id, config["budget"], config["policy"])
            ap = ap_lookup.get(key)
            if ap is not None:
                used_ap.add(key)
            row = {
                "run_id": run_id, "model": model, "seed": seed,
                "budget": config["budget"], "policy": config["policy"],
                "theta": config["theta"], "n_images": len(image_ids),
                **endpoints,
                "mean_prompts": float(prompt_counts[:, slot].mean()),
                "mask_AP50_95": ap.get("mask_AP50_95") if ap else None,
                "mask_AP50": ap.get("mask_AP50") if ap else None,
            }
            per_run.append(row)
            group = (model, config["budget"], config["policy"])
            boot_by_group.setdefault(group, []).append(boot_scores[:, slot].copy())
        del counts, pooled, boot_scores
    if set(ap_lookup) != used_ap:
        raise ValueError(f"Mask AP rows do not match experiment runs: {set(ap_lookup) - used_ap}")

    group_rows: dict[tuple, list[dict]] = {}
    for row in per_run:
        group_rows.setdefault((row["model"], row["budget"], row["policy"]), []).append(row)
    preferred_models = ("yolov8n", "yolo11n", "yolo11s")
    ordered_models = sorted({row["model"] for row in per_run}, key=lambda x: (preferred_models.index(x) if x in preferred_models else 99, x))
    budgets = sorted({row["budget"] for row in per_run})
    summaries = []
    for model in ordered_models:
        for budget in budgets:
            for policy in POLICIES:
                group = (model, budget, policy)
                rows = sorted(group_rows[group], key=lambda row: row["seed"])
                summary = {
                    "model": model, "budget": budget, "policy": policy,
                    "n_seeds": len(rows), "n_images_per_seed": rows[0]["n_images"],
                    "theta_by_seed": ";".join(f"{row['seed']}:{row['theta']:.2f}" for row in rows),
                }
                for metric in (*METRICS, "mask_AP50_95", "mask_AP50"):
                    observed = [row[metric] for row in rows if row[metric] is not None]
                    if len(observed) == len(rows):
                        mean, sd = _mean_sd(observed)
                    else:
                        mean, sd = None, None
                    summary[f"{metric}_mean"] = mean
                    summary[f"{metric}_std"] = sd
                summaries.append(summary)

    paired = []
    for model in ordered_models:
        for budget in budgets:
            vc_rows = {row["seed"]: row for row in group_rows[(model, budget, "VC")]}
            vc_boot = np.mean(np.stack(boot_by_group[(model, budget, "VC")]), axis=0)
            for baseline in BASELINES:
                comparator_rows = {row["seed"]: row for row in group_rows[(model, budget, baseline)]}
                if set(vc_rows) != set(comparator_rows):
                    raise ValueError("Seed pairing differs between policies")
                deltas = [vc_rows[seed]["mIoU"] - comparator_rows[seed]["mIoU"] for seed in sorted(vc_rows)]
                delta, delta_sd = _mean_sd(deltas)
                baseline_boot = np.mean(np.stack(boot_by_group[(model, budget, baseline)]), axis=0)
                draws = vc_boot - baseline_boot
                finite = draws[np.isfinite(draws)]
                if len(finite) < max(2, int(0.9 * bootstrap_reps)):
                    raise ValueError("Too many bootstrap draws have no evaluable class union")
                low, high = np.quantile(finite, [0.025, 0.975])
                paired.append({
                    "model": model, "budget": budget, "policy": "VC", "comparator": baseline,
                    "n_seeds": len(deltas), "n_images": len(reference_ids),
                    "delta_mIoU": delta, "delta_mIoU_pp": 100.0 * delta,
                    "seed_delta_std": delta_sd,
                    "ci95_low": float(low), "ci95_high": float(high),
                    "ci95_low_pp": 100.0 * float(low), "ci95_high_pp": 100.0 * float(high),
                    "bootstrap_reps": bootstrap_reps, "bootstrap_valid_reps": len(finite),
                    "bootstrap_seed": bootstrap_seed,
                    "interval_scope": "paired_images_conditional_on_frozen_choices",
                })

    paths = {
        "per_run_metrics": output_dir / "per_run_metrics.csv",
        "model_summary": output_dir / "model_summary.csv",
        "paired_differences": output_dir / "paired_differences.csv",
    }
    _write_csv(paths["per_run_metrics"], per_run)
    _write_csv(paths["model_summary"], summaries)
    _write_csv(paths["paired_differences"], paired)
    plot_result = _plots(output_dir, summaries)
    for path in [*paths.values(), *(Path(p) for p in plot_result["files"])]:
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"Report artifact is missing or empty: {path}")
    result = {
        "status": "complete",
        "report_version": "downstream-sam-report-1.0",
        "n_runs": len(run_files), "n_images_per_run": len(reference_ids),
        "models": ordered_models, "budgets": budgets,
        "all_model_groups_have_five_seeds": all(row["n_seeds"] == 5 for row in summaries),
        "primary_endpoint": "macro_class_foreground_mIoU_after_image_pixel_aggregation",
        "class_rule": "Include classes with nonzero pooled union, including false-positive-only classes.",
        "seed_aggregation": "Arithmetic mean of independently computed seed endpoints; sample SD uses ddof=1.",
        "bootstrap_scope": BOOTSTRAP_SCOPE,
        "bootstrap_reps": bootstrap_reps, "bootstrap_seed": bootstrap_seed,
        "interval_method": "pointwise percentile 95%; no simultaneous-coverage or multiplicity-adjusted claim",
        "mask_ap_rows_received": len(ap_lookup),
        "mask_ap_complete": len(ap_lookup) == len(per_run),
        "files": {key: str(path) for key, path in paths.items()},
        "plots": plot_result,
        "efficacy_conclusion": "Not inferred automatically; inspect paired effects, costs, and consistency.",
    }
    _atomic_json(complete_file, result)
    return result
