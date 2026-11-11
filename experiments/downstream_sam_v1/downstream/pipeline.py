"""Prepare -> calibration-only pilot -> select -> held-out inference -> paired reports."""
from __future__ import annotations

from collections import defaultdict
import contextlib
from dataclasses import asdict
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import time
import traceback

import numpy as np

from .common import atomic_json, digest_json, read_json, run_lock, sha256_file, stamp, write_csv
from .protocol import create_split, discover_sources, freeze_protocol
from .selection import calibrate, choose_policies, retain_predictions
from .storage import Store


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def preflight(root, settings, *, hash_sources=True):
    settings.validate()
    root = Path(root).resolve()
    sources = discover_sources(root.parents[1], settings)
    annotation = Path(settings.dataset_root) / "annotations/instances_val2017.json"
    raw = read_json(annotation)
    images = {int(row["id"]): row for row in raw["images"]}
    if len(images) != settings.expected_images or len(images) != len(raw["images"]):
        raise ValueError("Unexpected or duplicate COCO image IDs.")
    stat_rows = []
    image_root = Path(settings.dataset_root) / "images/val"
    for image_id, row in sorted(images.items()):
        path = image_root / row["file_name"]
        if not path.is_file():
            raise FileNotFoundError(path)
        st = path.stat()
        stat_rows.append([image_id, row["file_name"], st.st_size, st.st_mtime_ns])
    for index, source in enumerate(sources, 1):
        if hash_sources:
            source["sha256"] = sha256_file(source["prediction_file"])
            if source.get("declared_sha256") and source["declared_sha256"] != source["sha256"]:
                raise ValueError(f"Prediction cache checksum mismatch: {source['run_id']}")
        log(f"PREFLIGHT {index:02d}/{len(sources):02d}: {source['run_id']}, cache exists")
    report = {"status": "passed", "created_at": stamp(), "settings": asdict(settings),
              "n_sources": len(sources), "n_images": len(images), "sources": sources,
              "annotation_file": str(annotation), "annotation_sha256": sha256_file(annotation),
              "image_inventory_sha256": digest_json(stat_rows),
              "free_bytes": shutil.disk_usage(root).free, "source_content_hashes_checked": hash_sources,
              "no_model_loaded": True}
    atomic_json(root / "validation/preflight_latest.json", report)
    return report, images, annotation


def runtime_info(settings):
    import torch
    import torchvision
    if settings.device == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select the yolo GPU interpreter or explicitly set DEVICE='cpu'.")
        device = "cuda:0"
    else:
        device = settings.device
    if str(device).isdigit():
        device = f"cuda:{device}"
    info = {"python": platform.python_version(), "numpy": np.__version__,
            "torch": torch.__version__, "torchvision": torchvision.__version__,
            "cuda_runtime": torch.version.cuda, "device": str(device),
            "precision": "float32; no autocast; TF32 disabled"}
    import PIL
    import pycocotools.mask  # Fail before calibration if binary dependency is incompatible.
    info["Pillow"] = PIL.__version__
    info["pycocotools"] = importlib.metadata.version("pycocotools")
    if str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
        info["gpu"] = torch.cuda.get_device_name(torch.device(device))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    return info


def box_key(prediction):
    # Same geometric prompt can be reused across classes/runs: SAM receives no class.
    return digest_json([float(v) for v in prediction["xyxy"]])


def _new_predictor(root, settings, runtime):
    from .masks import SamBoxPredictor
    return SamBoxPredictor(root / "weights/sam_vit_b_01ec64.pth", root / "third_party",
                           model_type=settings.sam_model, device=runtime["device"],
                           batch_size=settings.batch_size)


def run_pilot(root, output, settings, store, masks, split, sources, runtime):
    """Only calibration images; no pilot endpoint is used to select a protocol."""
    from PIL import Image
    from .masks import evaluate_class_unions
    marker = output / "pilot/complete.json"
    if marker.exists():
        return read_json(marker)
    pilot_ids = split["calibration_ids"][:settings.pilot_images]
    predictor = _new_predictor(root, settings, runtime)
    rows = []
    started = time.monotonic()
    for index, image_id in enumerate(pilot_ids, 1):
        cached = store.image(sources[0]["run_id"], image_id)
        preds = retain_predictions(cached["predictions"], settings.fixed_theta, settings.primary_budget)
        with Image.open(Path(settings.dataset_root) / "images/val" / masks.images[image_id]["file_name"]) as im:
            rgb = np.asarray(im.convert("RGB"))
        if preds:
            encoding = predictor.set_image(rgb)
            rles, decoding = predictor.predict(np.asarray([p["xyxy"] for p in preds], dtype=np.float32))
        else:
            encoding, decoding, rles = 0.0, 0.0, []
        counts = evaluate_class_unions(preds, rles, masks.ground_truth(image_id),
                                      cached["height"], cached["width"], masks.n_classes)
        if counts.shape != (masks.n_classes, 3) or np.any(counts < 0):
            raise AssertionError("Invalid pilot pixel-count output.")
        rows.append({"image_id": image_id, "n_prompts": len(preds),
                     "encoding_seconds": encoding, "decoding_seconds": decoding})
        if index % 10 == 0 or index == len(pilot_ids):
            log(f"PILOT {index}/{len(pilot_ids)} calibration images; elapsed={time.monotonic()-started:.1f}s")
            atomic_json(output / "progress.json", {"phase": "calibration_pilot", "done": index,
                                                   "total": len(pilot_ids), "updated_at": stamp()})
    del predictor
    write_csv(output / "pilot/timing.csv", rows)
    result = {"status": "complete", "n_images": len(rows), "image_ids": pilot_ids,
              "all_images_in_calibration": set(pilot_ids) <= set(split["calibration_ids"]),
              "n_prompts": sum(r["n_prompts"] for r in rows),
              "elapsed_seconds": time.monotonic() - started,
              "purpose": "technical correctness and runtime only; no efficacy selection"}
    atomic_json(marker, result)
    return result


def calibration_bbox_ap(annotation, image_ids, images, category_ids):
    """Separate COCO-evaluator ranked AP on full original saved calibration predictions."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    detections = []
    for image in images:
        image_id = int(Path(str(image["image_id"])).stem)
        for p in image["predictions"]:
            x1, y1, x2, y2 = p["xyxy"]
            detections.append({"image_id": image_id, "category_id": category_ids[p["class_id"]],
                               "bbox": [x1, y1, x2-x1, y2-y1], "score": p["score"]})
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(annotation))
        if detections:
            dt = gt.loadRes(detections)
        else:
            dt = COCO()
            dt.dataset = {"images": gt.dataset["images"], "categories": gt.dataset["categories"], "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.imgIds = list(image_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {"calibration_bbox_AP50_95": float(evaluator.stats[0]),
            "calibration_bbox_AP50": float(evaluator.stats[1]),
            "protocol": "COCOeval bbox maxDets=100; original saved confidence ranking; upstream crowd filtering retained"}


def select_all(output, settings, store, masks, split, sources, annotation):
    all_configs, ap_rows = {}, []
    for run_index, source in enumerate(sources, 1):
        run_id = source["run_id"]
        marker = output / f"calibration/{run_id}.json"
        if marker.exists():
            result = read_json(marker)
        else:
            images = [store.image(run_id, image_id) for image_id in split["calibration_ids"]]
            log(f"CALIBRATION {run_index}/{len(sources)}: {run_id}")
            def progress(done, total):
                if done % 100 == 0 or done == total:
                    log(f"  {run_id}: {done}/{total} calibration images")
                    atomic_json(output / "progress.json", {"phase": "calibration", "run_id": run_id,
                                                           "done": done, "total": total, "updated_at": stamp()})
            rows = calibrate(images, store.source_metadata(run_id), settings.thresholds,
                             settings.budgets, progress=progress)
            configs = [dict(row, run_id=run_id, model=source["model"], seed=source["seed"])
                       for row in choose_policies(rows, fixed_theta=settings.fixed_theta)]
            if settings.compute_ap_model_selection:
                log(f"RANKED BBOX AP: {run_id}, calibration images only")
            ap = calibration_bbox_ap(annotation, split["calibration_ids"], images, masks.category_ids) \
                if settings.compute_ap_model_selection else None
            result = {"status": "complete", "run_id": run_id, "calibration_image_ids": split["calibration_ids"],
                      "configs": configs, "ap": ap}
            write_csv(output / f"calibration/{run_id}_grid.csv", rows)
            atomic_json(marker, result)
        if result["calibration_image_ids"] != split["calibration_ids"]:
            raise ValueError("Calibration split changed.")
        all_configs[run_id] = result["configs"]
        if result["ap"] is not None:
            ap_rows.append(dict(result["ap"], run_id=run_id, model=source["model"], seed=source["seed"]))
    write_csv(output / "calibration/selected_configurations.csv", [c for rows in all_configs.values() for c in rows])
    write_csv(output / "calibration/ranked_ap.csv", ap_rows)
    # Seal the decisions before opening any held-out image in the segmentation stage.
    atomic_json(output / "calibration/selection_frozen.json", {"configs": all_configs,
                "evaluation_image_ids": split["evaluation_ids"], "selection_uses_evaluation_labels": False,
                "selection_hash": digest_json(all_configs), "status": "complete"})
    return all_configs, ap_rows


def infer_heldout(root, output, settings, store, masks, split, sources, configs, runtime,
                  predictor_factory=None):
    """One image encode shared across all runs; cache full predicted masks, not clipped boxes."""
    from PIL import Image
    from .masks import evaluate_class_unions
    predictor = None
    started = time.monotonic()
    for index, image_id in enumerate(split["evaluation_ids"], 1):
        path = Path(settings.dataset_root) / "images/val" / masks.images[image_id]["file_name"]
        image_sha = sha256_file(path)
        rows = {s["run_id"]: store.image(s["run_id"], image_id) for s in sources}
        selections, prompts = {}, {}
        for run_id, image in rows.items():
            selections[run_id] = []
            for config in configs[run_id]:
                subset = retain_predictions(image["predictions"], config["theta"], config["budget"])
                selections[run_id].append(subset)
                for pred in subset:
                    prompts.setdefault(box_key(pred), pred["xyxy"])
        cached = store.inference(image_id)
        if cached is None:
            if prompts:
                if predictor is None:
                    predictor = predictor_factory() if predictor_factory else _new_predictor(root, settings, runtime)
                with Image.open(path) as im:
                    rgb = np.asarray(im.convert("RGB"))
                encoding = predictor.set_image(rgb)
                keys = sorted(prompts)
                rles, decoding = predictor.predict(np.asarray([prompts[k] for k in keys], dtype=np.float32))
                if len(keys) != len(rles):
                    raise AssertionError("SAM output count does not match prompt count.")
                mask_map = dict(zip(keys, rles))
            else:
                mask_map, encoding, decoding = {}, 0.0, 0.0
            store.save_inference(image_id, image_sha, mask_map, encoding, decoding)
            cached = store.inference(image_id)
        if cached["image_sha"] != image_sha or not set(prompts).issubset(cached["masks"]):
            raise ValueError(f"Inference cache/image changed for image {image_id}.")
        ground_truth = None
        for run_id, image in rows.items():
            if store.evaluation(run_id, image_id) is not None:
                continue
            if ground_truth is None:
                ground_truth = masks.ground_truth(image_id)
            memo, values, n_prompts = {}, [], []
            for subset in selections[run_id]:
                subset_key = tuple((p["id"], p["class_id"]) for p in subset)
                if subset_key not in memo:
                    memo[subset_key] = evaluate_class_unions(
                        subset, [cached["masks"][box_key(p)] for p in subset], ground_truth,
                        image["height"], image["width"], masks.n_classes)
                values.append(memo[subset_key])
                n_prompts.append(len(subset))
            store.save_evaluation(run_id, image_id, np.asarray(values, dtype=np.int64),
                                  np.asarray(n_prompts, dtype=np.int64))
        if index % 10 == 0 or index == len(split["evaluation_ids"]):
            elapsed = time.monotonic() - started
            log(f"HELD-OUT {index}/{len(split['evaluation_ids'])}: all {len(sources)} runs; elapsed={elapsed:.1f}s")
            atomic_json(output / "progress.json", {"phase": "heldout", "done": index,
                       "total": len(split["evaluation_ids"]), "updated_at": stamp(), "elapsed_seconds": elapsed})
    del predictor


def export_per_image(output, store, split, sources, configs):
    files = []
    for source in sources:
        run_id = source["run_id"]
        target = output / f"per_image/{run_id}.npz"
        if not target.exists():
            records = [store.evaluation(run_id, image_id) for image_id in split["evaluation_ids"]]
            if any(row is None for row in records):
                raise ValueError(f"Incomplete held-out evaluations: {run_id}")
            counts = np.stack([row[0] for row in records])
            prompts = np.stack([row[1] for row in records])
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".partial")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, image_ids=np.asarray(split["evaluation_ids"], dtype=np.int64),
                                    counts=counts, prompt_counts=prompts,
                                    config_json=np.asarray(json.dumps(configs[run_id], allow_nan=False)))
            os.replace(temporary, target)
        files.append(target)
    return files


def mask_ap_results(output, settings, store, masks, split, sources, configs, annotation):
    from .masks import evaluate_mask_ap
    if not settings.compute_mask_ap:
        return []
    all_rows = []
    for source in sources:
        run_id = source["run_id"]
        result_path = output / f"mask_ap/{run_id}.json"
        saved = read_json(result_path) if result_path.exists() else {}
        for config in configs[run_id]:
            key = f"{config['budget']}:{config['theta']}"
            if key not in saved:
                detections = []
                for image_id in split["evaluation_ids"]:
                    image = store.image(run_id, image_id)
                    cached = store.inference(image_id)
                    for pred in retain_predictions(image["predictions"], config["theta"], config["budget"]):
                        detections.append({"image_id": image_id, "category_id": masks.category_ids[pred["class_id"]],
                                           "score": pred["score"], "segmentation": cached["masks"][box_key(pred)]})
                log(f"MASK AP: {run_id}, budget={config['budget']}, theta={config['theta']:.2f}")
                saved[key] = evaluate_mask_ap(annotation, split["evaluation_ids"], detections, max_dets=100)
                atomic_json(result_path, saved)
            all_rows.append({"run_id": run_id, "model": source["model"], "seed": source["seed"],
                             "budget": config["budget"], "policy": config["policy"],
                             "mask_AP50_95": saved[key]["mask_AP50_95"], "mask_AP50": saved[key]["mask_AP50"]})
    write_csv(output / "mask_ap/summary.csv", all_rows)
    return all_rows


def cross_model_selection(output, sources, configs, ap_rows):
    """Select model+threshold using calibration only, including standard AP -> F1 baseline."""
    if not ap_rows:
        return
    per_run = {}
    import csv
    with (output / "reports/per_run_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            per_run[(row["run_id"], int(row["budget"]), row["policy"])] = row
    rows = []
    for seed in sorted({s["seed"] for s in sources}):
        available = [s["run_id"] for s in sources if s["seed"] == seed]
        best_ap = sorted((r for r in ap_rows if r["seed"] == seed),
                         key=lambda r: (-r["calibration_bbox_AP50_95"], r["model"]))[0]
        for budget in sorted({c["budget"] for c in configs[available[0]]}):
            for policy in ("AP_then_F1", "F1", "LRP", "BestIoGT", "mAC", "VC"):
                if policy == "AP_then_F1":
                    chosen = next(c for c in configs[best_ap["run_id"]] if c["policy"] == "F1" and c["budget"] == budget)
                else:
                    candidates = [c for rid in available for c in configs[rid] if c["policy"] == policy and c["budget"] == budget]
                    sign = 1 if policy == "LRP" else -1
                    chosen = sorted(candidates, key=lambda c: (sign*c["calibration_value"], -c["theta"], c["model"]))[0]
                heldout = per_run[(chosen["run_id"], budget, chosen["policy"])]
                rows.append({"seed": seed, "budget": budget, "selector": policy, "selected_model": chosen["model"],
                             "selected_run": chosen["run_id"], "theta": chosen["theta"],
                             **{key: heldout[key] for key in ("mIoU", "foreground_precision", "foreground_recall", "mean_prompts")}})
    write_csv(output / "reports/cross_model_selection.csv", rows)


def execute(root, settings, *, mode, run_name, auto_prepare):
    root = Path(root).resolve()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name) is None:
        raise ValueError("RUN_NAME must contain only letters, numbers, hyphens or underscores.")
    output = root / "outputs" / run_name
    with run_lock(output):
        store = None
        try:
            audit, images, annotation = preflight(root, settings)
            if mode == "preflight":
                log("PREFLIGHT COMPLETE. No model inference, installation or weight download performed.")
                return
            from .setup_runtime import check_runtime_and_weights
            dependencies = check_runtime_and_weights(root, download_weights=auto_prepare)
            runtime = runtime_info(settings)
            from .masks import CocoMasks
            masks = CocoMasks(annotation)
            runtime["dependencies"] = dependencies
            runtime["image_inventory_sha256"] = audit["image_inventory_sha256"]
            protocol = freeze_protocol(root, output, settings, audit["sources"], annotation,
                                       mode="calibration-pilot-and-heldout", device_info=runtime)
            split = create_split(images, settings.split_seed, settings.calibration_images)
            split_path = output / "split.json"
            if split_path.exists() and read_json(split_path) != split:
                raise ValueError("Saved image split differs from the frozen protocol.")
            atomic_json(split_path, split)
            if mode == "full" and (output / "complete.json").exists():
                completed = read_json(output / "complete.json")
                required = [output / "reports/model_summary.csv", output / "reports/paired_differences.csv"]
                required += [output / f"per_image/{s['run_id']}.npz" for s in audit["sources"]]
                if completed.get("protocol_fingerprint") == protocol["fingerprint"] and all(p.is_file() for p in required):
                    log(f"ALREADY COMPLETE: {output}. Existing results retained.")
                    return
            if mode == "report":
                selection = read_json(output / "calibration/selection_frozen.json")
                configs = selection["configs"]
                import csv
                ap_rows = []
                for source in audit["sources"]:
                    record = read_json(output / f"calibration/{source['run_id']}.json")
                    if record["ap"] is not None:
                        ap_rows.append(dict(record["ap"], **{k: source[k] for k in ("model", "seed", "run_id")}))
                store = Store(output / "cache.sqlite")
            else:
                store = Store(output / "cache.sqlite")
                from region_eval.coco import prepare_coco_image
                expected_gts = {i: prepare_coco_image(row, masks.annotations[i], masks.category_to_class)[0]
                                for i, row in images.items()}
                for source in audit["sources"]:
                    store.import_source(source, images, max_budget=max(settings.budgets),
                                        confidence_floor=min(settings.thresholds), calibration_ids=split["calibration_ids"],
                                        category_names=masks.category_names, expected_gts=expected_gts, progress=log)
                run_pilot(root, output, settings, store, masks, split, audit["sources"], runtime)
                if mode == "pilot":
                    log(f"PILOT COMPLETE. Formal held-out inference has not started. Output: {output / 'pilot'}")
                    return
                configs, ap_rows = select_all(output, settings, store, masks, split, audit["sources"], annotation)
                infer_heldout(root, output, settings, store, masks, split, audit["sources"], configs, runtime)
            files = export_per_image(output, store, split, audit["sources"], configs)
            ap = mask_ap_results(output, settings, store, masks, split, audit["sources"], configs, annotation)
            from .reporting import build_reports
            report = build_reports(files, output / "reports", bootstrap_reps=settings.bootstrap_reps,
                                   bootstrap_seed=settings.bootstrap_seed, mask_ap_rows=ap or None)
            cross_model_selection(output, audit["sources"], configs, ap_rows)
            timings = [dict(image_id=i, **{k: store.inference(i)[k] for k in
                       ("encoding_seconds", "decoding_seconds", "unique_prompts")}) for i in split["evaluation_ids"]]
            write_csv(output / "reports/shared_inference_timing.csv", timings)
            atomic_json(output / "mask_annotation_audit.json", masks.audit)
            atomic_json(output / "complete.json", {"status": "complete", "protocol_fingerprint": protocol["fingerprint"],
                        "completed_at": stamp(), "n_runs": len(audit["sources"]), "n_evaluation_images": len(split["evaluation_ids"]),
                        "report": report, "timing_scope": "shared inference cache; not separate per-policy runtime benchmarks"})
            atomic_json(output / "progress.json", {"phase": "complete", "updated_at": stamp()})
            log(f"SUCCESS: {output}. No detector training or detector inference performed.")
        except BaseException as exc:
            atomic_json(output / "last_failure.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        "time": stamp(), "type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
                        "resume": "Rerun the same entrypoint. Completed transactional image records are reused."})
            raise
        finally:
            if store is not None:
                store.close()
