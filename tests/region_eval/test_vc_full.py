import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from region_eval.vc_full import (
    FULL_RUN_VERSION,
    _atomic_json,
    _fingerprint,
    load_existing_shards,
    shard_path,
)


def item(directory, image_id, width=20, height=10):
    return dict(path=Path(directory) / f"{image_id}.jpg", width=width, height=height)


def shard(run_fingerprint, index, sample):
    image_id = sample["path"].stem
    image = dict(
        image_id=image_id,
        width=sample["width"],
        height=sample["height"],
    )
    return dict(
        schema_version=1,
        full_run_version=FULL_RUN_VERSION,
        run_fingerprint=run_fingerprint,
        index=index,
        image_id=image_id,
        raw_image=dict(image, annotations=[], predictions=[]),
        filtered_image=dict(image, gts=[], predictions=[]),
        audit=dict(image_id=image_id),
    )


class VCFullResumeTests(unittest.TestCase):
    def test_atomic_json_replaces_with_strict_json(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            _atomic_json(path, {"value": 1})
            _atomic_json(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"value": 2})
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_completed_shard_is_loaded_and_missing_shard_is_pending(self):
        with TemporaryDirectory() as directory:
            items = [item(directory, "a"), item(directory, "b")]
            fingerprint = _fingerprint({"run": 1})
            path = shard_path(Path(directory) / "shards", 0, "a")
            _atomic_json(path, shard(fingerprint, 0, items[0]))
            completed, pending = load_existing_shards(
                items, Path(directory) / "shards", fingerprint)
            self.assertEqual(list(completed), [0])
            self.assertEqual(pending, [1])

    def test_changed_experiment_cannot_reuse_shard(self):
        with TemporaryDirectory() as directory:
            sample = item(directory, "a")
            path = shard_path(Path(directory) / "shards", 0, "a")
            _atomic_json(path, shard("old", 0, sample))
            with self.assertRaisesRegex(ValueError, "different experiment"):
                load_existing_shards([sample], Path(directory) / "shards", "new")

    def test_unsafe_image_id_cannot_escape_shard_directory(self):
        with self.assertRaises(ValueError):
            shard_path("shards", 0, "../outside")


if __name__ == "__main__":
    unittest.main()
