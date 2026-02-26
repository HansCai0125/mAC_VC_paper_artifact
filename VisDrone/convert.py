import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT
import shutil
from PIL import Image

# ======== Path configuration ========
SRC_TRAIN_ROOT = str(DATA_ROOT / "VisDrone" / "VisDrone2019-DET-train")
SRC_VAL_ROOT = str(DATA_ROOT / "VisDrone" / "VisDrone2019-DET-val")

DST_IMG_TRAIN = r"/images\train"
DST_IMG_VAL = r"/images\val"
DST_LABEL_TRAIN = r"/labels\train"
DST_LABEL_VAL = r"/labels\val"

# Create destination directories
for d in [DST_IMG_TRAIN, DST_IMG_VAL, DST_LABEL_TRAIN, DST_LABEL_VAL]:
    os.makedirs(d, exist_ok=True)


def visdrone_line_to_yolo(fields, img_w, img_h):
    """
    Convert one VisDrone annotation line into YOLO format.
    Returns: (class_id, x_c, y_c, w, h) or None (ignore this box).
    """
    if len(fields) < 8:
        return None

    bbox_left = float(fields[0])
    bbox_top = float(fields[1])
    bbox_width = float(fields[2])
    bbox_height = float(fields[3])
    score = int(fields[4])
    category_id = int(fields[5])
    # fields[6], fields[7] are truncation / occlusion and are unused here

    # Ignore score=0 and ignored regions (0) / others (11)
    if score == 0 or category_id in (0, 11):
        return None

    if bbox_width <= 0 or bbox_height <= 0:
        return None

    # VisDrone classes 1-10 -> YOLO classes 0-9
    yolo_class = category_id - 1

    x_c = (bbox_left + bbox_width / 2.0) / img_w
    y_c = (bbox_top + bbox_height / 2.0) / img_h
    w = bbox_width / img_w
    h = bbox_height / img_h

    # Clamp to [0, 1] to handle occasional boundary overflow
    x_c = max(0.0, min(1.0, x_c))
    y_c = max(0.0, min(1.0, y_c))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return yolo_class, x_c, y_c, w, h


def process_split(src_root, dst_img_dir, dst_label_dir):
    """
    Process one split (train or val):
    - Read src_root/images and src_root/annotations
    - Copy images into dst_img_dir
    - Write YOLO labels into dst_label_dir
    """
    img_src_dir = os.path.join(src_root, "images")
    ann_src_dir = os.path.join(src_root, "annotations")

    ann_files = [f for f in os.listdir(ann_src_dir) if f.endswith(".txt")]
    ann_files.sort()

    print(f"Processing {src_root}, total annotation files: {len(ann_files)}")

    for idx, ann_name in enumerate(ann_files, 1):
        base_name = os.path.splitext(ann_name)[0]
        # VisDrone images are usually .jpg
        img_name = base_name + ".jpg"

        img_src_path = os.path.join(img_src_dir, img_name)
        ann_src_path = os.path.join(ann_src_dir, ann_name)

        if not os.path.isfile(img_src_path):
            # A few images may be .png; add a fallback check here if needed
            print(f"[WARN] Image not found for {ann_src_path}, skip.")
            continue

        # Destination paths
        img_dst_path = os.path.join(dst_img_dir, img_name)
        label_dst_path = os.path.join(dst_label_dir, base_name + ".txt")

        # Read image size
        with Image.open(img_src_path) as im:
            img_w, img_h = im.size

        yolo_lines = []

        with open(ann_src_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fields = line.split(",")
                result = visdrone_line_to_yolo(fields, img_w, img_h)
                if result is None:
                    continue
                cls_id, x_c, y_c, w, h = result
                yolo_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        # Even if there are no valid targets, write an empty label file (YOLO supports it)
        with open(label_dst_path, "w") as f_out:
            if yolo_lines:
                f_out.write("\n".join(yolo_lines))

        # Copy image
        if not os.path.isfile(img_dst_path):
            shutil.copy2(img_src_path, img_dst_path)

        if idx % 500 == 0:
            print(f"  processed {idx}/{len(ann_files)} files.")

    print(f"Finished split: {src_root}")


if __name__ == "__main__":
    process_split(SRC_TRAIN_ROOT, DST_IMG_TRAIN, DST_LABEL_TRAIN)
    process_split(SRC_VAL_ROOT, DST_IMG_VAL, DST_LABEL_VAL)
    print("All done.")
