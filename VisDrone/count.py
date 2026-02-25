# 统计 VisDrone 各类别的 图像数量 & label数量，并计算总数
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

# ================= 配置区域 =================
# 你的 labels 根目录
LABEL_ROOT = str(DATA_ROOT / "VisDrone" / "labels")
SPLITS = ["train", "val"]   # 如果名字不一样就改这里

# 注意顺序要和 YOLO 的 class id 一致！
class_names = [
    "pedestrian",        # 0
    "people",            # 1
    "bicycle",           # 2
    "car",               # 3
    "van",               # 4
    "truck",             # 5
    "tricycle",          # 6
    "awning-tricycle",   # 7
    "bus",               # 8
    "motor"              # 9
]

num_classes = len(class_names)

# 统计：
#   images_per_class  = 每个类别出现过的图像数量
#   labels_per_class  = 每个类别的 bbox/label 数量
images_per_class = Counter()
labels_per_class = Counter()

# 总数统计
total_images = 0      # train+val 一共有多少张图（=多少个 txt）
total_labels = 0      # train+val 一共有多少个 bbox（=所有 txt 的所有行）

for split in SPLITS:
    split_dir = Path(LABEL_ROOT) / split
    if not split_dir.exists():
        print(f"[WARN] 不存在目录: {split_dir}")
        continue

    for txt_path in split_dir.glob("*.txt"):
        with open(txt_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        # 每个 txt 对应一张图
        total_images += 1
        # 这一张图里有多少个 bbox
        total_labels += len(lines)

        # 记录该图片里出现过的类别（只计一次）
        classes_in_this_image = set()

        for line in lines:
            parts = line.split()
            if len(parts) == 0:
                continue

            try:
                cls_id = int(parts[0])
            except ValueError:
                # 如果有脏数据就跳过
                continue

            # 统计 label 数量：每一行就是一个 bbox
            labels_per_class[cls_id] += 1
            # 统计图像数量：该图像出现过这个类就记一次
            classes_in_this_image.add(cls_id)

        for cls_id in classes_in_this_image:
            images_per_class[cls_id] += 1

# 把 Counter 转成有序列表（保证 0~num_classes-1）
images_counts_list = [images_per_class[i] for i in range(num_classes)]
labels_counts_list = [labels_per_class[i] for i in range(num_classes)]

print("每个类别的图像数量:", images_counts_list)
print("每个类别的 label 数量:", labels_counts_list)

# 打印总数
print("图像总数 (train+val):", total_images)
print("label(bbox) 总数 (train+val):", total_labels)
print("sum(images_counts_list) =", sum(images_counts_list))
print("sum(labels_counts_list)  =", sum(labels_counts_list))

# ================= 画图：图一（图像数量） =================
fig1, ax1 = plt.subplots(figsize=(10, 5))
bars1 = ax1.bar(range(num_classes), images_counts_list)
ax1.set_xticks(range(num_classes))
ax1.set_xticklabels(class_names, rotation=45, ha="right")
ax1.set_ylabel("Number of images")
ax1.set_title("Number of images per class (VisDrone train+val)")

# 在每个柱子上标注数值
for bar in bars1:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()
plt.savefig("visdrone_images_per_class.png", dpi=300)

# ================= 画图：图二（label数量） =================
fig2, ax2 = plt.subplots(figsize=(10, 5))
bars2 = ax2.bar(range(num_classes), labels_counts_list)
ax2.set_xticks(range(num_classes))
ax2.set_xticklabels(class_names, rotation=45, ha="right")
ax2.set_ylabel("Number of labels (bboxes)")
ax2.set_title("Number of labels per class (VisDrone train+val)")

for bar in bars2:
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()
plt.savefig("visdrone_labels_per_class.png", dpi=300)

plt.show()
