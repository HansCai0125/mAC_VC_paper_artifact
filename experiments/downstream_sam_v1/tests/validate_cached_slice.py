"""Read-only real-cache parity check on calibration images; no detector/SAM inference."""
from copy import deepcopy
from pathlib import Path
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parents[1]))
sys.path.insert(0, str(ROOT / "third_party"))

from downstream.common import atomic_json
from downstream.protocol import Settings, create_split, discover_sources
from downstream.selection import calibrate, retain_predictions
from downstream.masks import CocoMasks
from region_eval.coco import prepare_coco_image
from region_eval.vc_v1 import evaluate_vc


def main():
    settings = Settings()
    source = discover_sources(ROOT.parents[1], settings)[0]
    document = json.loads(Path(source["prediction_file"]).read_text(encoding="utf-8"))
    masks = CocoMasks(Path(settings.dataset_root) / "annotations/instances_val2017.json")
    split = create_split(masks.images, settings.split_seed, settings.calibration_images)
    selected_ids = set(split["calibration_ids"][:24])
    selected = []
    for image in document["images"]:
        image_id = int(Path(image["image_id"]).stem)
        expected = prepare_coco_image(masks.images[image_id], masks.annotations[image_id], masks.category_to_class)[0]
        if sorted(image["gts"], key=lambda g: g["id"]) != sorted(expected, key=lambda g: g["id"]):
            raise AssertionError(f"Annotation disagreement: {image_id}")
        if image_id in selected_ids:
            selected.append(image)
            masks.ground_truth(image_id)
    rows = calibrate(selected, document["metadata"], (.05, .25, .95), (10, 20, 40))
    errors = []
    for row in rows:
        subset = dict(document, images=[dict(image, predictions=retain_predictions(image["predictions"], row["theta"], row["budget"]))
                                       for image in selected])
        summary, gt_rows, _ = evaluate_vc(subset, theta=row["theta"])
        expected = {"mAC": summary["mAC"], "VC": summary["VC"], "F1": summary["standard_F1"], "LRP": summary["standard_LRP"],
                    "BestIoGT": sum(g["original_BestIoGT"] for g in gt_rows) / len(gt_rows)}
        for metric, reference in expected.items():
            error = abs(row[metric] - reference)
            if error > 1e-10:
                raise AssertionError(f"{metric} parity failed: {error}")
            errors.append(error)
    result = {"status": "passed", "source": source["run_id"], "annotation_images_checked": len(document["images"]),
              "metric_parity_calibration_images": len(selected), "parameter_settings": len(rows),
              "max_absolute_metric_error": max(errors), "mask_annotation_audit": masks.audit,
              "detector_inference": False, "SAM_inference": False}
    atomic_json(ROOT / "validation/real_cache_check.json", result)
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
