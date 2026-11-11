"""Seed-correct, resumable RT-DETR training for the COCO five-seed study."""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import torch
import ultralytics
from ultralytics import RTDETR
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import torch_distributed_zero_first

from .case_analysis import _file_hash
from .coco import COCO_NAMES
from .seeded_training import (
    _archive_checkpointless_attempt,
    _atomic_json,
    _dataset_cache_signature,
    _fingerprint,
    _set_global_seed,
    build_seeded_dataloader,
    checkpoint_parameter_hash,
    loader_seed,
)
from .vc_smoke import _model_names


RTDETR_SEED_TRAINING_VERSION = "coco-rtdetr-seed-training-0.1.0"


class SeededRTDETRTrainer(RTDETRTrainer):
    """RT-DETR trainer whose shuffle and worker RNG include ``args.seed``."""

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


def preflight_rtdetr_seed_training(
        *, base_checkpoint, seeds, data_yaml, output, training_args):
    """Validate and fingerprint one RT-DETR five-seed training protocol."""
    base_checkpoint = Path(base_checkpoint).resolve()
    data_yaml = Path(data_yaml).resolve()
    output = Path(output).resolve()
    seeds = tuple(seeds)
    if seeds != tuple(range(5)) and len(seeds) != 2:
        raise ValueError("seeds must be 0--4 or a two-seed smoke pair")
    if len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be unique")
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)
    if not data_yaml.is_file():
        raise FileNotFoundError(data_yaml)
    dataset_root = data_yaml.parent
    required = (
        dataset_root / "images" / "train",
        dataset_root / "images" / "val",
        dataset_root / "labels" / "train",
        dataset_root / "labels" / "val",
    )
    if not all(path.is_dir() for path in required):
        raise FileNotFoundError("COCO train/val images or labels are missing")

    model = RTDETR(str(base_checkpoint))
    _model_names(model, COCO_NAMES)
    trainer = model._smart_load("trainer")
    predictor = model._smart_load("predictor")
    if trainer.__name__ != "RTDETRTrainer":
        raise ValueError(f"unexpected RT-DETR trainer: {trainer.__name__}")
    if predictor.__name__ != "RTDETRPredictor":
        raise ValueError(f"unexpected RT-DETR predictor: {predictor.__name__}")
    del model

    cache_signatures = {
        "train": _dataset_cache_signature(
            dataset_root / "labels" / "train.cache"),
        "val": _dataset_cache_signature(
            dataset_root / "labels" / "val.cache"),
    }
    plan = dict(
        rtdetr_seed_training_version=RTDETR_SEED_TRAINING_VERSION,
        base_checkpoint=str(base_checkpoint),
        base_checkpoint_sha256=_file_hash(base_checkpoint),
        data_yaml=str(data_yaml),
        data_yaml_sha256=_file_hash(data_yaml),
        dataset_cache_signatures=cache_signatures,
        seeds=list(seeds),
        training_args=training_args,
        source_sha256={
            "region_eval/rtdetr_seeded_training.py": _file_hash(Path(__file__)),
        },
        environment=dict(
            torch_version=torch.__version__,
            ultralytics_version=ultralytics.__version__,
            ultralytics_source=str(Path(ultralytics.__file__).resolve()),
        ),
    )
    plan["suite_fingerprint"] = _fingerprint(plan)
    return output, plan


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


def run_rtdetr_seed_training_suite(
        *, base_checkpoint, seeds, data_yaml, output, training_args, purpose,
        expected_suite_fingerprint=None):
    """Train or resume RT-DETR and prove that all seed paths are distinct."""
    output, plan = preflight_rtdetr_seed_training(
        base_checkpoint=base_checkpoint,
        seeds=seeds,
        data_yaml=data_yaml,
        output=output,
        training_args=training_args,
    )
    if (expected_suite_fingerprint is not None
            and plan["suite_fingerprint"] != expected_suite_fingerprint):
        raise ValueError(
            "RT-DETR seed-training suite differs from its frozen fingerprint")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "suite_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("suite_fingerprint") != plan["suite_fingerprint"]:
            raise ValueError("training protocol changed; use a new output name")
    else:
        if any(output.iterdir()):
            raise ValueError(f"nonempty output has no manifest: {output}")
        _atomic_json(manifest_path, dict(
            status="started",
            purpose=purpose,
            created_at=datetime.now(timezone.utc).isoformat(),
            **plan,
        ))

    completed = []
    project = output / "runs" / "rtdetr_l"
    for seed in seeds:
        run_dir = project / f"seed_{seed}"
        completion_path = run_dir / "seed_training_complete.json"
        prior = _validated_completion(
            completion_path, plan["suite_fingerprint"])
        if prior is not None:
            completed.append(prior)
            print(f"SKIP: RT-DETR-L seed {seed} already complete", flush=True)
            continue

        last_checkpoint = run_dir / "weights" / "last.pt"
        if last_checkpoint.is_file():
            print(
                f"RESUME: RT-DETR-L seed {seed} from {last_checkpoint}",
                flush=True,
            )
            model = RTDETR(str(last_checkpoint))
            model.train(trainer=SeededRTDETRTrainer, resume=True)
        else:
            _archive_checkpointless_attempt(run_dir)
            _set_global_seed(seed)
            model = RTDETR(str(Path(base_checkpoint).resolve()))
            print(f"TRAIN: RT-DETR-L seed {seed} -> {run_dir}", flush=True)
            model.train(
                trainer=SeededRTDETRTrainer,
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
        best = run_dir / "weights" / "best.pt"
        final_checkpoint = best if best.is_file() else last_checkpoint
        if not final_checkpoint.is_file():
            raise FileNotFoundError(
                f"training produced no checkpoint: {run_dir}")
        audit_path = run_dir / "seed_audit.json"
        if not audit_path.is_file():
            raise FileNotFoundError(audit_path)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(
            final_checkpoint, map_location="cpu", weights_only=False)
        recorded_seed = (checkpoint.get("train_args") or {}).get("seed")
        if recorded_seed != seed:
            raise ValueError(
                f"checkpoint seed mismatch: expected {seed}, found {recorded_seed}")
        record = dict(
            status="complete",
            rtdetr_seed_training_version=RTDETR_SEED_TRAINING_VERSION,
            suite_fingerprint=plan["suite_fingerprint"],
            model="rtdetr_l",
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

    if len(completed) != len(tuple(seeds)):
        raise AssertionError("incomplete RT-DETR seed group")
    if len({row["parameter_sha256"] for row in completed}) != len(completed):
        raise AssertionError("duplicate RT-DETR model parameters remain")
    if len({row["first_batch_sha256"] for row in completed}) != len(completed):
        raise AssertionError("RT-DETR training batch order did not change by seed")
    complete = dict(
        status="complete",
        purpose=purpose,
        suite_fingerprint=plan["suite_fingerprint"],
        completed_runs=len(completed),
        distinct_parameter_hashes=len({
            row["parameter_sha256"] for row in completed}),
        distinct_first_batch_hashes=len({
            row["first_batch_sha256"] for row in completed}),
        results=sorted(completed, key=lambda row: row["seed"]),
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    _atomic_json(output / "suite_complete.json", complete)
    print(json.dumps(complete, ensure_ascii=False), flush=True)
    print(f"SUCCESS: RT-DETR five-seed training complete: {output}", flush=True)
    return output
