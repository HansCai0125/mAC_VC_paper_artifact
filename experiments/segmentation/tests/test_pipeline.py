"""Pipeline contract checks use synthetic masks, never a trained model."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from downstream.common import atomic_json, digest_json, read_json, sha256_file
from downstream.masks import CocoMasks, encode_binary_mask
from downstream.pipeline import (box_key, export_per_image, infer_heldout,
                                 select_all, mask_ap_results, cross_model_selection)
from downstream.protocol import Settings, create_split, freeze_protocol
from downstream.reporting import build_reports
from downstream.storage import Store


class PipelineTests(unittest.TestCase):
    def test_split_is_repeatable_disjoint_and_exhaustive(self):
        split = create_split(range(100), 17, 20)
        self.assertEqual(split, create_split(reversed(range(100)), 17, 20))
        self.assertFalse(set(split["calibration_ids"]) & set(split["evaluation_ids"]))
        self.assertEqual(set(range(100)), set(split["calibration_ids"] + split["evaluation_ids"]))
        self.assertNotEqual(split, create_split(range(100), 18, 20))

    def test_protocol_resume_handles_tuples_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project/experiments/segmentation"
            (root / "weights").mkdir(parents=True)
            (root / "weights/sam_vit_b_01ec64.pth").write_bytes(b"fake test checkpoint")
            annotation = root / "annotations.json"
            atomic_json(annotation, {"images": []})
            settings = Settings()
            first = freeze_protocol(root, root / "out", settings, [], annotation, mode="fixture", device_info={})
            second = freeze_protocol(root, root / "out", settings, [], annotation, mode="fixture", device_info={})
            self.assertEqual(first, second)
            annotation.write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed"):
                freeze_protocol(root, root / "out", settings, [], annotation, mode="fixture", device_info={})

    def test_end_to_end_resume_and_report_without_sam_or_test_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "output"
            image_dir = root / "data/images/val"
            image_dir.mkdir(parents=True)
            annotation = root / "data/annotations/instances_val2017.json"
            image_ids = list(range(1, 7))
            image_rows = [{"id": i, "file_name": f"{i:012d}.png", "width": 8, "height": 8} for i in image_ids]
            for row in image_rows:
                Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(image_dir / row["file_name"])
            annotations = [{"id": i, "image_id": i, "category_id": 1, "iscrowd": 0,
                            "bbox": [1, 1, 4, 4], "area": 16,
                            "segmentation": [[1, 1, 5, 1, 5, 5, 1, 5]]} for i in image_ids]
            atomic_json(annotation, {"images": image_rows, "annotations": annotations,
                                     "categories": [{"id": 1, "name": "target"}]})
            settings = Settings(dataset_root=str(root / "data"), expected_images=6, calibration_images=2,
                                pilot_images=1, models=("test",), seeds=(0, 1), thresholds=(0.05, 0.25, 0.95),
                                budgets=(1, 2), primary_budget=1, bootstrap_reps=20)
            split = create_split(image_ids, settings.split_seed, 2)
            masks = CocoMasks(annotation)
            metadata = {"class_ids": [0], "class_names": ["target"],
                        "coordinate_frame": "original_image_xyxy", "postprocessing": {"confidence_floor": 0.001}}
            sources = []
            for seed in settings.seeds:
                images = []
                for image_id in image_ids:
                    images.append({"image_id": str(image_id), "width": 8, "height": 8,
                                   "gts": [{"id": f"gt_{image_id}", "class_id": 0, "xyxy": [1., 1., 5., 5.]}],
                                   "predictions": [{"id": "p0", "class_id": 0, "xyxy": [1., 1., 5., 5.], "score": .9},
                                                   {"id": "p1", "class_id": 0, "xyxy": [6., 6., 8., 8.], "score": .1}]})
                path = root / f"test_seed{seed}.json"
                atomic_json(path, {"schema_version": 1, "metadata": metadata, "images": images})
                sources.append({"run_id": f"test_seed{seed}", "model": "test", "seed": seed,
                                "prediction_file": str(path), "sha256": sha256_file(path)})
            store = Store(output / "cache.sqlite")
            try:
                for source in sources:
                    store.import_source(source, masks.images, max_budget=2, confidence_floor=.05,
                                        calibration_ids=split["calibration_ids"], category_names=["target"], progress=lambda _: None)
                configs, ap_rows = select_all(output, settings, store, masks, split, sources, annotation)
                self.assertEqual(read_json(output / "calibration/test_seed0.json")["calibration_image_ids"], split["calibration_ids"])
                frozen_hash = digest_json(configs)

                class FakePredictor:
                    n_images = 0
                    def set_image(self, rgb):
                        self.n_images += 1
                        return .1
                    def predict(self, boxes):
                        rles = []
                        for box in boxes:
                            binary = np.zeros((8, 8), bool)
                            x1, y1, x2, y2 = map(int, box)
                            binary[y1:y2, x1:x2] = True
                            rles.append(encode_binary_mask(binary))
                        return rles, .01
                fake = FakePredictor()
                infer_heldout(root, output, settings, store, masks, split, sources, configs, {}, predictor_factory=lambda: fake)
                self.assertEqual(fake.n_images, len(split["evaluation_ids"]))
                def must_not_load():
                    raise AssertionError("Completed masks must be reused")
                infer_heldout(root, output, settings, store, masks, split, sources, configs, {}, predictor_factory=must_not_load)
                self.assertEqual(digest_json(configs), frozen_hash)
                files = export_per_image(output, store, split, sources, configs)
                ap = mask_ap_results(output, settings, store, masks, split, sources, configs, annotation)
                report = build_reports(files, output / "reports", bootstrap_reps=20, mask_ap_rows=ap)
                self.assertEqual(report["status"], "complete")
                self.assertTrue(report["mask_ap_complete"])
                cross_model_selection(output, sources, configs, ap_rows)
                self.assertTrue((output / "reports/cross_model_selection.csv").is_file())
                # An altered image must never silently reuse its existing mask predictions.
                changed = image_dir / masks.images[split["evaluation_ids"][0]]["file_name"]
                Image.fromarray(np.full((8, 8, 3), 127, np.uint8)).save(changed)
                with self.assertRaisesRegex(ValueError, "changed"):
                    infer_heldout(root, output, settings, store, masks, split, sources, configs, {}, predictor_factory=must_not_load)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
