import os
import pickle
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT

import numpy as np
from PIL import Image


def load_cifar100_batch(pkl_path: Path):
    """Load CIFAR-100 pickled file (train/test/meta)."""
    with pkl_path.open("rb") as f:
        # CIFAR pickles sometimes need latin1 for compatibility
        data = pickle.load(f, encoding="latin1")
    return data


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_images_and_labels(
    images: np.ndarray,
    labels: np.ndarray,
    out_img_dir: Path,
    out_lbl_dir: Path,
    prefix: str,
    img_ext: str = ".jpg",
    jpeg_quality: int = 95,
):
    """
    Save images to out_img_dir and YOLO labels to out_lbl_dir.
    Each image gets ONE bbox covering the full image:
      class_id 0.5 0.5 1.0 1.0
    """
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    n = images.shape[0]
    for i in range(n):
        img = images[i]  # (H, W, 3) uint8
        cls = int(labels[i])

        stem = f"{prefix}_{i:06d}"
        img_path = out_img_dir / f"{stem}{img_ext}"
        lbl_path = out_lbl_dir / f"{stem}.txt"

        # save image
        pil_img = Image.fromarray(img)
        if img_ext.lower() in [".jpg", ".jpeg"]:
            pil_img.save(img_path, quality=jpeg_quality, subsampling=0)
        else:
            pil_img.save(img_path)

        # save label (full-image bbox)
        # YOLO format: class x_center y_center width height (normalized)
        lbl_path.write_text(f"{cls} 0.5 0.5 1.0 1.0\n", encoding="utf-8")


def write_dataset_yaml(out_root: Path, class_names: list, yaml_name: str = "cifar100_fullbox.yaml"):
    """
    Create Ultralytics YOLO detection dataset yaml.
    """
    yaml_path = out_root / yaml_name
    # Use forward slashes for Windows compatibility in YAML
    path_str = out_root.as_posix()

    # YAML content
    lines = []
    lines.append(f"path: {path_str}")
    lines.append("train: images/train")
    lines.append("val: images/val")
    lines.append(f"nc: {len(class_names)}")
    # Write names as a YAML list
    # Escape quotes if needed
    safe_names = [n.replace('"', '\\"') for n in class_names]
    names_list = ", ".join([f'"{n}"' for n in safe_names])
    lines.append(f"names: [{names_list}]")

    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def main():
    # ====== EDIT THESE TWO PATHS ======
    cifar_root = DATA_ROOT / "cifar100" / "cifar-100-python"
    out_root = DATA_ROOT / "cifar100" / "cifar100_yolo_det"
    # =================================

    train_pkl = cifar_root / "train"
    test_pkl = cifar_root / "test"
    meta_pkl = cifar_root / "meta"

    if not train_pkl.exists() or not test_pkl.exists() or not meta_pkl.exists():
        raise FileNotFoundError(
            "Cannot find CIFAR-100 python files. Expecting: train, test, meta under cifar-100-python."
        )

    # Load meta for class names
    meta = load_cifar100_batch(meta_pkl)
    # CIFAR-100 fine label names (100 classes)
    fine_names = meta["fine_label_names"]  # list[str]

    # Load train/test
    train_data = load_cifar100_batch(train_pkl)
    test_data = load_cifar100_batch(test_pkl)

    # CIFAR stores data as (N, 3072) with R,G,B blocks
    X_train = train_data["data"]  # (50000, 3072)
    y_train = np.array(train_data["fine_labels"], dtype=np.int64)

    X_test = test_data["data"]  # (10000, 3072)
    y_test = np.array(test_data["fine_labels"], dtype=np.int64)

    # Reshape to (N, 3, 32, 32) then transpose to (N, 32, 32, 3)
    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)

    # Output dirs like your YOLO detection dataset structure
    img_train_dir = out_root / "images" / "train"
    img_val_dir = out_root / "images" / "val"
    lbl_train_dir = out_root / "labels" / "train"
    lbl_val_dir = out_root / "labels" / "val"

    ensure_dir(out_root)

    print(f"[INFO] Saving train images/labels -> {img_train_dir} / {lbl_train_dir}")
    save_images_and_labels(X_train, y_train, img_train_dir, lbl_train_dir, prefix="train", img_ext=".jpg")

    print(f"[INFO] Saving val images/labels (from CIFAR test) -> {img_val_dir} / {lbl_val_dir}")
    save_images_and_labels(X_test, y_test, img_val_dir, lbl_val_dir, prefix="val", img_ext=".jpg")

    yaml_path = write_dataset_yaml(out_root, fine_names, yaml_name="cifar100_fullbox.yaml")
    print(f"[DONE] YOLO dataset yaml written: {yaml_path}")

    print("\nNOTE:")
    print("- CIFAR-100 has NO bounding boxes. This script creates a full-image bbox for each image.")
    print("- Suitable for testing pipeline / format conversion, not real object detection benchmarking.")


if __name__ == "__main__":
    main()
