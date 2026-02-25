import os
import shutil
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT

# ============ 路径配置 ============

# 1) Kaggle 数据集原始根目录（解压后的文件夹路径）
SRC_ROOT = str(DATA_ROOT / "MRI")

# 2) 目标根目录（整理成和 VisDrone 一样的格式）
DST_ROOT = str(DATA_ROOT / "MRI")

# 目标目录结构：
# E:\Project\Database\BrainTumor\
#   images\train
#   images\val
#   labels\train
#   labels\val

IMAGES_TRAIN = Path(DST_ROOT) / "images" / "train"
IMAGES_VAL   = Path(DST_ROOT) / "images" / "val"
LABELS_TRAIN = Path(DST_ROOT) / "labels" / "train"
LABELS_VAL   = Path(DST_ROOT) / "labels" / "val"

for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
    d.mkdir(parents=True, exist_ok=True)


def copy_split(split_name: str, dst_img_dir: Path, dst_label_dir: Path):
    """
    处理 Train 或 Val：
    - 遍历 SRC_ROOT/split_name 下所有类别子目录
    - 把 images/ 和 labels/ 里的文件复制到 dst_img_dir / dst_label_dir
    """
    split_root = Path(SRC_ROOT) / split_name
    if not split_root.exists():
        print(f"[WARN] split 目录不存在: {split_root}")
        return

    # 例如：Train 下的子目录 [Glioma, Meningioma, No Tumor, Pituitary]
    class_dirs = [p for p in split_root.iterdir() if p.is_dir()]

    total_imgs = 0
    total_labels = 0

    print(f"[INFO] 处理 {split_name} ...")
    for cls_dir in class_dirs:
        img_dir = cls_dir / "images"
        lbl_dir = cls_dir / "labels"

        if not img_dir.exists():
            print(f"  [WARN] {img_dir} 不存在，跳过该类。")
            continue
        if not lbl_dir.exists():
            print(f"  [WARN] {lbl_dir} 不存在，跳过该类。")
            continue

        # 遍历该类的所有图片
        for img_path in img_dir.glob("*.*"):
            if not img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
                continue

            base_name = img_path.stem  # 不含扩展名
            # label 文件假设是同名 .txt
            lbl_path = lbl_dir / f"{base_name}.txt"

            # 目标路径
            dst_img_path = dst_img_dir / img_path.name
            dst_lbl_path = dst_label_dir / f"{base_name}.txt"

            # 拷贝图片
            shutil.copy2(img_path, dst_img_path)
            total_imgs += 1

            # 拷贝/创建标签
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl_path)
                total_labels += 1
            else:
                # 没有标注则写空文件（YOLO 允许没有目标）
                with open(dst_lbl_path, "w", encoding="utf-8") as f:
                    pass
                print(f"  [WARN] 缺失 label: {lbl_path}，已创建空标注。")

    print(f"[DONE] {split_name}: 复制图片 {total_imgs} 张, 标签文件 {total_labels} 个")
    print(f"       输出目录: imgs -> {dst_img_dir}, labels -> {dst_label_dir}\n")


if __name__ == "__main__":
    copy_split("Train", IMAGES_TRAIN, LABELS_TRAIN)
    copy_split("Val",   IMAGES_VAL,   LABELS_VAL)
    print("[FINISHED] 所有数据整理完成。")
