"""Seed-correct, resumable Ultralytics detection training utilities."""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import shutil
from time import sleep
from uuid import uuid4

import torch
from torch.utils.data import distributed
import ultralytics
import ultralytics.data.build as ultralytics_data_build
import ultralytics.engine.trainer as ultralytics_engine_trainer
from ultralytics import YOLO
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.data.utils import PIN_MEMORY
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import (
    de_parallel,
    torch_distributed_zero_first,
)

from .case_analysis import _file_hash
from .coco import COCO_NAMES
from .vc_smoke import _model_names


SEED_TRAINING_VERSION = "coco-seed-correct-training-0.1.0"
LOADER_BASE_SEED = 6148914691236517205


def loader_seed(seed, rank=-1):
    if type(seed) is not int or seed < 0:
        raise ValueError("training seed must be a nonnegative integer")
    if type(rank) is not int:
        raise ValueError("rank must be an integer")
    return LOADER_BASE_SEED + seed + max(rank, 0)


def build_seeded_dataloader(
        dataset, batch, workers, shuffle=True, rank=-1, seed=0):
    """Build the local Ultralytics loader with the experiment seed included."""
    batch = min(batch, len(dataset))
    device_count = torch.cuda.device_count()
    worker_count = min(
        (os.cpu_count() or 1) // max(device_count, 1), workers)
    sampler = None
    if rank != -1:
        sampler = distributed.DistributedSampler(
            dataset, shuffle=shuffle, seed=seed)
    generator = torch.Generator()
    generator.manual_seed(loader_seed(seed, rank))
    return InfiniteDataLoader(
        dataset=dataset,
        batch_size=batch,
        shuffle=shuffle and sampler is None,
        num_workers=worker_count,
        sampler=sampler,
        pin_memory=PIN_MEMORY,
        collate_fn=getattr(dataset, "collate_fn", None),
        worker_init_fn=seed_worker,
        generator=generator,
    )


def _atomic_json(path, value, attempts=20):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    for attempt in range(1, attempts + 1):
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            sleep(min(0.05 * (2 ** (attempt - 1)), 2.0))
        finally:
            if temporary.exists():
                temporary.unlink()


def _fingerprint(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")
    return sha256(raw).hexdigest()


def checkpoint_parameter_hash(path):
    """Hash model tensors while excluding mutable checkpoint metadata."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        raise ValueError(f"checkpoint contains no model: {path}")
    digest = sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class SeededDetectionTrainer(DetectionTrainer):
    """Detection trainer whose shuffle and worker RNG depend on args.seed."""

    def get_dataloader(
            self, dataset_path, batch_size=16, rank=0, mode="train"):
        if mode not in {"train", "val"}:
            raise ValueError(f"invalid dataloader mode: {mode}")
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning(
                "WARNING: rect=True disables training DataLoader shuffle")
            shuffle = False
        workers = (
            self.args.workers if mode == "train" else self.args.workers * 2)
        loader = build_seeded_dataloader(
            dataset,
            batch_size,
            workers,
            shuffle=shuffle,
            rank=rank,
            seed=int(self.args.seed),
        )
        if mode == "train":
            self._vc_loader_seed = loader_seed(int(self.args.seed), rank)
        return loader

    def preprocess_batch(self, batch):
        """Record the first training batch as direct evidence of seed use."""
        if not getattr(self, "_vc_seed_audit_written", False):
            paths = [Path(value).name for value in batch.get("im_file", [])]
            audit_path = Path(self.save_dir) / "seed_audit.json"
            if not audit_path.is_file():
                _atomic_json(audit_path, dict(
                    seed=int(self.args.seed),
                    loader_seed=int(self._vc_loader_seed),
                    first_batch_image_names=paths,
                    first_batch_sha256=sha256(
                        "\n".join(paths).encode("utf-8")).hexdigest(),
                    recorded_at=datetime.now(timezone.utc).isoformat(),
                ))
            self._vc_seed_audit_written = True
        return super().preprocess_batch(batch)


def _set_global_seed(seed):
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _dataset_cache_signature(path):
    """Validate and fingerprint the label cache Ultralytics consumes."""
    import numpy as np
    from ultralytics.data.utils import get_hash, img2label_paths

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Ultralytics dataset cache is missing: {path}")
    cache = np.load(str(path), allow_pickle=True).item()
    labels = cache.get("labels")
    if not isinstance(labels, list) or not labels:
        raise ValueError(f"invalid Ultralytics dataset cache: {path}")
    image_files = [row["im_file"] for row in labels]
    current_hash = get_hash(img2label_paths(image_files) + image_files)
    if current_hash != cache.get("hash"):
        raise ValueError(
            f"Ultralytics dataset cache is stale: {path}")
    return dict(
        path=str(path),
        bytes=path.stat().st_size,
        sha256=_file_hash(path),
        ultralytics_data_hash=current_hash,
        ultralytics_cache_version=cache.get("version"),
        image_count=len(image_files),
    )


def preflight_seed_training(
        *, models, seeds, data_yaml, output, training_args):
    """Validate inputs and freeze a reproducible training-suite manifest."""
    data_yaml = Path(data_yaml).resolve()
    output = Path(output).resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(f"COCO data YAML is missing: {data_yaml}")
    dataset_root = data_yaml.parent
    train_dir = dataset_root / "images" / "train"
    val_dir = dataset_root / "images" / "val"
    train_labels = dataset_root / "labels" / "train"
    val_labels = dataset_root / "labels" / "val"
    if not all(path.is_dir() for path in (
            train_dir, val_dir, train_labels, val_labels)):
        raise FileNotFoundError(
            "COCO train/val image or label directories are missing")
    if tuple(seeds) != tuple(range(5)) and len(tuple(seeds)) != 2:
        raise ValueError("suite must be the five formal seeds or two smoke seeds")
    if len(set(seeds)) != len(tuple(seeds)):
        raise ValueError("training seeds must be unique")

    model_rows = []
    for model_name, checkpoint in models.items():
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"base checkpoint is missing: {checkpoint}")
        model = YOLO(str(checkpoint), task="detect")
        _model_names(model, COCO_NAMES)
        model_rows.append(dict(
            model=model_name,
            checkpoint=str(checkpoint),
            checkpoint_sha256=_file_hash(checkpoint),
        ))
        del model

    # The .cache files contain the parsed labels and image paths consumed by
    # this Ultralytics version. Hashing them gives a fast, byte-exact snapshot
    # without rereading more than 100,000 small label files on every launch.
    # Ultralytics still performs its normal path/size validation per loader.
    dataset_cache_signatures = {
        "train": _dataset_cache_signature(
            dataset_root / "labels" / "train.cache"),
        "val": _dataset_cache_signature(
            dataset_root / "labels" / "val.cache"),
    }
    build_source = Path(ultralytics_data_build.__file__).resolve()
    trainer_source = Path(ultralytics_engine_trainer.__file__).resolve()
    plan = dict(
        seed_training_version=SEED_TRAINING_VERSION,
        data_yaml=str(data_yaml),
        data_yaml_sha256=_file_hash(data_yaml),
        dataset_cache_signatures=dataset_cache_signatures,
        models=model_rows,
        seeds=list(seeds),
        training_args=training_args,
        seeded_training_source_sha256=_file_hash(Path(__file__)),
        environment=dict(
            torch_version=torch.__version__,
            ultralytics_version=ultralytics.__version__,
            ultralytics_source=str(
                Path(ultralytics.__file__).resolve()),
            ultralytics_build_source_sha256=_file_hash(build_source),
            ultralytics_trainer_source_sha256=_file_hash(trainer_source),
        ),
    )
    plan["suite_fingerprint"] = _fingerprint(plan)
    return output, plan


def _archive_checkpointless_attempt(run_dir):
    if not run_dir.is_dir() or not any(run_dir.iterdir()):
        return
    archive_root = run_dir.parents[2] / "incomplete_without_checkpoint"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = archive_root / f"{run_dir.parent.name}_{run_dir.name}_{stamp}"
    if target.exists():
        raise FileExistsError(target)
    if run_dir.resolve() == target.resolve():
        raise ValueError("invalid archive target")
    shutil.move(str(run_dir), str(target))
    print(f"ARCHIVE: checkpointless partial attempt -> {target}", flush=True)


def _validated_completion(path, suite_fingerprint):
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("suite_fingerprint") != suite_fingerprint:
        raise ValueError(f"completed run belongs to another suite: {path}")
    checkpoint = Path(value["checkpoint"])
    if checkpoint_parameter_hash(checkpoint) != value["parameter_sha256"]:
        raise ValueError(f"completed checkpoint changed: {checkpoint}")
    return value


def run_seed_training_suite(
        *, models, seeds, data_yaml, output, training_args, purpose,
        expected_suite_fingerprint=None):
    """Run/resume a seed-correct suite and prove seed-specific data order."""
    output, plan = preflight_seed_training(
        models=models,
        seeds=seeds,
        data_yaml=data_yaml,
        output=output,
        training_args=training_args,
    )
    if (expected_suite_fingerprint is not None
            and plan["suite_fingerprint"] != expected_suite_fingerprint):
        raise ValueError(
            "seed-training suite differs from its prevalidated fingerprint")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "suite_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("suite_fingerprint") != plan["suite_fingerprint"]:
            raise ValueError(
                f"training protocol changed; choose a new output: {output}")
    else:
        unexpected = list(output.iterdir())
        if unexpected:
            raise ValueError(
                f"training output is nonempty without a manifest: {output}")
        _atomic_json(manifest_path, dict(
            status="started",
            purpose=purpose,
            created_at=datetime.now(timezone.utc).isoformat(),
            **plan,
        ))

    completed = []
    for model_name, base_checkpoint in models.items():
        for seed in seeds:
            project = output / "runs" / model_name
            run_dir = project / f"seed_{seed}"
            completion_path = run_dir / "seed_training_complete.json"
            prior = _validated_completion(
                completion_path, plan["suite_fingerprint"])
            if prior is not None:
                completed.append(prior)
                print(
                    f"SKIP: {model_name} seed {seed} already complete",
                    flush=True)
                continue

            last_checkpoint = run_dir / "weights" / "last.pt"
            if last_checkpoint.is_file():
                print(
                    f"RESUME: {model_name} seed {seed} from {last_checkpoint}",
                    flush=True)
                model = YOLO(str(last_checkpoint), task="detect")
                model.train(trainer=SeededDetectionTrainer, resume=True)
            else:
                _archive_checkpointless_attempt(run_dir)
                _set_global_seed(seed)
                model = YOLO(str(base_checkpoint), task="detect")
                print(
                    f"TRAIN: {model_name} seed {seed} -> {run_dir}",
                    flush=True)
                model.train(
                    trainer=SeededDetectionTrainer,
                    data=str(Path(data_yaml).resolve()),
                    seed=seed,
                    deterministic=True,
                    project=str(project),
                    name=f"seed_{seed}",
                    exist_ok=True,
                    resume=False,
                    **training_args,
                )

            actual_run_dir = Path(model.trainer.save_dir).resolve()
            if actual_run_dir != run_dir.resolve():
                raise ValueError(
                    f"unexpected Ultralytics save directory: {actual_run_dir}")
            best_checkpoint = run_dir / "weights" / "best.pt"
            final_checkpoint = (
                best_checkpoint if best_checkpoint.is_file()
                else run_dir / "weights" / "last.pt")
            if not final_checkpoint.is_file():
                raise FileNotFoundError(
                    f"training produced no checkpoint: {run_dir}")
            audit_path = run_dir / "seed_audit.json"
            if not audit_path.is_file():
                raise FileNotFoundError(
                    f"first-batch seed audit is missing: {audit_path}")
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            checkpoint = torch.load(
                final_checkpoint, map_location="cpu", weights_only=False)
            recorded_seed = (checkpoint.get("train_args") or {}).get("seed")
            if recorded_seed != seed:
                raise ValueError(
                    f"checkpoint seed mismatch: expected {seed}, "
                    f"found {recorded_seed}")
            record = dict(
                status="complete",
                seed_training_version=SEED_TRAINING_VERSION,
                suite_fingerprint=plan["suite_fingerprint"],
                model=model_name,
                seed=seed,
                checkpoint=str(final_checkpoint.resolve()),
                checkpoint_file_sha256=_file_hash(final_checkpoint),
                parameter_sha256=checkpoint_parameter_hash(final_checkpoint),
                first_batch_sha256=audit["first_batch_sha256"],
                loader_seed=audit["loader_seed"],
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            _atomic_json(completion_path, record)
            completed.append(record)
            del model
            torch.cuda.empty_cache()

    for model_name in models:
        rows = [row for row in completed if row["model"] == model_name]
        if len(rows) != len(tuple(seeds)):
            raise AssertionError(f"incomplete seed group: {model_name}")
        parameter_hashes = {row["parameter_sha256"] for row in rows}
        batch_hashes = {row["first_batch_sha256"] for row in rows}
        if len(parameter_hashes) != len(rows):
            raise AssertionError(
                f"duplicate model parameters remain: {model_name}")
        if len(batch_hashes) != len(rows):
            raise AssertionError(
                f"training batch order did not change by seed: {model_name}")

    complete = dict(
        status="complete",
        purpose=purpose,
        suite_fingerprint=plan["suite_fingerprint"],
        completed_runs=len(completed),
        distinct_parameter_hashes=len({
            row["parameter_sha256"] for row in completed}),
        distinct_first_batch_hashes=len({
            row["first_batch_sha256"] for row in completed}),
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    _atomic_json(output / "suite_complete.json", complete)
    print(json.dumps(complete, ensure_ascii=False), flush=True)
    print(f"SUCCESS: seed-correct training suite complete: {output}", flush=True)
    return output
