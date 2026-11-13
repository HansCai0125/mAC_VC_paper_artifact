"""Transactional per-image prediction and inference storage with crash recovery."""
from __future__ import annotations

import io
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np

from .common import sha256_file


def pack(value):
    return zlib.compress(json.dumps(value, separators=(",", ":"), allow_nan=False).encode(), 3)


def unpack(value):
    return json.loads(zlib.decompress(value))


class Store:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS source_runs(run_id TEXT PRIMARY KEY, metadata BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS detections(run_id TEXT, image_id INTEGER, payload BLOB NOT NULL,
                                             PRIMARY KEY(run_id,image_id));
        CREATE TABLE IF NOT EXISTS inferred(image_id INTEGER PRIMARY KEY, image_sha TEXT, masks BLOB,
                                           encoding_seconds REAL, decoding_seconds REAL, unique_prompts INTEGER);
        CREATE TABLE IF NOT EXISTS evaluated(run_id TEXT, image_id INTEGER, payload BLOB NOT NULL,
                                            PRIMARY KEY(run_id,image_id));
        """)

    def close(self):
        self.db.close()

    def source_metadata(self, run_id):
        row = self.db.execute("SELECT metadata FROM source_runs WHERE run_id=?", (run_id,)).fetchone()
        return unpack(row[0]) if row else None

    def import_source(self, source, image_index, *, max_budget, confidence_floor,
                      calibration_ids=(), category_names=None, expected_gts=None, progress=print):
        run_id = source["run_id"]
        if self.source_metadata(run_id) is not None:
            return
        # Hash the bytes actually read before accepting the historical cache identity.
        path = Path(source["prediction_file"])
        actual = sha256_file(path)
        if actual != source["sha256"]:
            raise ValueError(f"Input changed after preflight: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        metadata = document["metadata"]
        from region_eval.vc_v1 import _validate_canonical_document
        _validate_canonical_document(document)
        if category_names is not None and metadata["class_names"] != list(category_names):
            raise ValueError("Cached class order differs from COCO category order.")
        if metadata.get("postprocessing", {}).get("confidence_floor", 1) > confidence_floor:
            raise ValueError("Source confidence floor is above requested threshold grid.")
        if metadata.get("coordinate_frame") != "original_image_xyxy":
            raise ValueError("Cached coordinates are not original-image coordinates.")
        expected = set(image_index)
        calibration_ids = set(calibration_ids)
        seen = set()
        records = []
        for image in document["images"]:
            image_id = int(Path(str(image["image_id"])).stem)
            if image_id not in expected or image_id in seen:
                raise ValueError(f"Unexpected/duplicate image ID in {run_id}: {image_id}")
            seen.add(image_id)
            coco = image_index[image_id]
            if (image["width"], image["height"]) != (coco["width"], coco["height"]):
                raise ValueError(f"Image dimensions disagree: {image_id}")
            if expected_gts is not None:
                actual_gt = sorted(image["gts"], key=lambda g: g["id"])
                expected_gt = sorted(expected_gts[image_id], key=lambda g: g["id"])
                if actual_gt != expected_gt:
                    raise ValueError(f"Cached ground truth differs from the COCO annotation file: {run_id}/{image_id}")
            predictions = sorted(image["predictions"], key=lambda p: (-p["score"], p["id"]))
            # Keep original calibration rankings for separate AP-based model selection.
            if image_id not in calibration_ids:
                predictions = [p for p in predictions if p["score"] >= confidence_floor][:max_budget]
            records.append((run_id, image_id, pack(dict(image, predictions=predictions))))
        if seen != expected:
            raise ValueError(f"Source {run_id} does not cover all declared images.")
        with self.db:
            self.db.execute("DELETE FROM detections WHERE run_id=?", (run_id,))
            self.db.executemany("INSERT INTO detections VALUES (?,?,?)", records)
            self.db.execute("INSERT INTO source_runs VALUES (?,?)", (run_id, pack(metadata)))
        progress(f"CACHE READY: {run_id}, {len(records)} images")

    def image(self, run_id, image_id):
        row = self.db.execute("SELECT payload FROM detections WHERE run_id=? AND image_id=?",
                              (run_id, image_id)).fetchone()
        if row is None:
            raise ValueError(f"Missing prepared predictions {run_id}/{image_id}")
        return unpack(row[0])

    def inference(self, image_id):
        row = self.db.execute("SELECT image_sha,masks,encoding_seconds,decoding_seconds,unique_prompts "
                              "FROM inferred WHERE image_id=?", (image_id,)).fetchone()
        if row is None:
            return None
        return {"image_sha": row[0], "masks": unpack(row[1]), "encoding_seconds": row[2],
                "decoding_seconds": row[3], "unique_prompts": row[4]}

    def save_inference(self, image_id, image_sha, masks, encoding_seconds, decoding_seconds):
        with self.db:
            self.db.execute("INSERT INTO inferred VALUES (?,?,?,?,?,?)", (
                image_id, image_sha, pack(masks), encoding_seconds, decoding_seconds, len(masks)))

    def evaluation(self, run_id, image_id):
        row = self.db.execute("SELECT payload FROM evaluated WHERE run_id=? AND image_id=?",
                              (run_id, image_id)).fetchone()
        if row is None:
            return None
        with np.load(io.BytesIO(row[0]), allow_pickle=False) as data:
            return data["counts"], data["prompt_counts"]

    def save_evaluation(self, run_id, image_id, counts, prompt_counts):
        buffer = io.BytesIO()
        np.savez_compressed(buffer, counts=counts, prompt_counts=prompt_counts)
        with self.db:
            self.db.execute("INSERT INTO evaluated VALUES (?,?,?)", (run_id, image_id, buffer.getvalue()))
