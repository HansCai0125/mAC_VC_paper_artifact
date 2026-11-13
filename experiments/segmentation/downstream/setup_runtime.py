"""Prepare public dependencies locally; never upgrade the user's torch/CUDA."""
from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import zipfile

from .common import atomic_json, read_json, sha256_file
from .protocol import SAM_COMMIT, SAM_URL


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "mAC-downstream-experiment/1.0"})
    written, last = 0, time.monotonic()
    with urllib.request.urlopen(request, timeout=90) as response, partial.open("wb") as stream:
        expected = int(response.headers.get("Content-Length", 0))
        for block in iter(lambda: response.read(1024 * 1024), b""):
            stream.write(block)
            written += len(block)
            if time.monotonic() - last >= 5:
                print(f"DOWNLOAD {destination.name}: {written / 1024**2:.1f} MB"
                      + (f" / {expected / 1024**2:.1f} MB" if expected else ""), flush=True)
                last = time.monotonic()
        if written == 0 or (expected and written != expected):
            raise IOError(f"Incomplete download: {url}. Run again to retry preparation.")
    os.replace(partial, destination)


def prepare_runtime(root, *, download_weights=True):
    root = Path(root).resolve()
    vendor = root / "third_party"
    vendor.mkdir(parents=True, exist_ok=True)
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    for name in ("numpy", "torch", "torchvision", "PIL", "matplotlib"):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError(f"Missing {name}. Use the existing yolo interpreter; "
                               "this script does not replace torch/CUDA.")
    marker_path = vendor / "runtime_manifest.json"
    previous = read_json(marker_path) if marker_path.exists() else {}
    if not (vendor / "pycocotools/__init__.py").is_file():
        print("SETUP: installing pycocotools 2.0.11 into experiment/third_party", flush=True)
        local_temp = root / "_setup_tmp"
        local_temp.mkdir(exist_ok=True)
        env = dict(os.environ, TEMP=str(local_temp), TMP=str(local_temp),
                   PIP_CACHE_DIR=str(vendor / "pip_cache"))
        subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                        "--only-binary=:all:", "--no-deps", "--target", str(vendor),
                        "pycocotools==2.0.11"], check=True, env=env)
        importlib.invalidate_caches()
    sam_package = vendor / "segment_anything"
    archive = vendor / f"segment-anything-{SAM_COMMIT}.zip"
    if not (sam_package / "__init__.py").is_file():
        if not archive.exists():
            download(f"https://codeload.github.com/facebookresearch/segment-anything/zip/{SAM_COMMIT}", archive)
        prefix = f"segment-anything-{SAM_COMMIT}/"
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                if not member.filename.startswith(prefix):
                    raise ValueError("Unexpected SAM archive root.")
                relative = Path(member.filename[len(prefix):])
                if ".." in relative.parts or relative.is_absolute():
                    raise ValueError("Unsafe archive member.")
                if not relative.parts or (relative.parts[0] != "segment_anything" and relative.name != "LICENSE"):
                    continue
                target = vendor / relative
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zipped.read(member))
    if not (sam_package / "predictor.py").is_file():
        raise RuntimeError("SAM source preparation incomplete. Inspect third_party before running.")
    source_hashes = {str(p.relative_to(vendor)): sha256_file(p) for p in sorted(sam_package.rglob("*.py"))}
    if previous.get("sam_source_sha256") and previous["sam_source_sha256"] != source_hashes:
        raise ValueError("Local SAM source changed; do not reuse the old experiment protocol.")
    checkpoint = root / "weights/sam_vit_b_01ec64.pth"
    checkpoint_sha = None
    if download_weights:
        if not checkpoint.is_file():
            print("SETUP: downloading official SAM ViT-B weights (about 375 MB).", flush=True)
            download(SAM_URL, checkpoint)
        if checkpoint.stat().st_size < 300_000_000:
            raise ValueError("SAM checkpoint appears incomplete; inspect the weight file.")
        checkpoint_sha = sha256_file(checkpoint)
        if previous.get("checkpoint_sha256") and previous["checkpoint_sha256"] != checkpoint_sha:
            raise ValueError("SAM checkpoint hash differs from the recorded local download.")
    manifest = {"sam_commit": SAM_COMMIT, "sam_source_sha256": source_hashes,
                "pycocotools": "2.0.11", "sam_weight_source": SAM_URL,
                "checkpoint_sha256": checkpoint_sha or previous.get("checkpoint_sha256"),
                "checkpoint_hash_kind": "locally recorded SHA256; not a publisher-signed checksum"}
    atomic_json(marker_path, manifest)
    print("SETUP READY: dependencies are local to this experiment.", flush=True)
    return manifest


def check_runtime_and_weights(root, *, download_weights=True):
    """Normal entrypoint: check user's packages, download ONLY the public weight if needed."""
    root = Path(root).resolve()
    vendor = root / "third_party"
    if vendor.exists() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    missing = [name for name in ("numpy", "torch", "torchvision", "PIL", "matplotlib",
                                 "pycocotools", "segment_anything") if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Missing packages: " + ", ".join(missing)
                           + ". Install experiments/segmentation/requirements.txt; no package was installed automatically.")
    sam_root = Path(importlib.util.find_spec("segment_anything").origin).parent
    source_hashes = {str(p.relative_to(sam_root)): sha256_file(p) for p in sorted(sam_root.rglob("*.py"))}
    checkpoint = root / "weights/sam_vit_b_01ec64.pth"
    if not checkpoint.is_file():
        if not download_weights:
            raise FileNotFoundError(f"Missing SAM weight: {checkpoint}. Run prepare_environment.py.")
        print("WEIGHTS: downloading official SAM ViT-B into this experiment/weights (about 375 MB).", flush=True)
        download(SAM_URL, checkpoint)
    if checkpoint.stat().st_size < 300_000_000:
        raise ValueError("SAM checkpoint appears incomplete.")
    checkpoint_sha = sha256_file(checkpoint)
    marker = root / "weights/weight_manifest.json"
    if marker.exists() and read_json(marker)["sha256"] != checkpoint_sha:
        raise ValueError("SAM weight changed since preparation; keep old results separate.")
    atomic_json(marker, {"url": SAM_URL, "sha256": checkpoint_sha,
                         "hash_kind": "locally measured; no publisher-signed checksum supplied"})
    return {"sam_source_directory": str(sam_root), "sam_source_sha256": source_hashes,
            "weight_sha256": checkpoint_sha, "requested_sam_commit": SAM_COMMIT,
            "source_pin_note": "Actual imported source hashes recorded; see requirements.txt for requested pin."}
