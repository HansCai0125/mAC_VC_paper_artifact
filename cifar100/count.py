#数数量
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
LABEL_ROOT = str(DATA_ROOT / "cifar100" / "cifar100_yolo_det" / "labels")
SPLITS = ["train", "val"]   # 如果名字不一样就改这里

# 注意顺序要和你的 classes.txt 一致！
class_names = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm"
]


num_classes = len(class_names)

# 统计：images_per_class  = 每个类别出现过的图像数量
#      labels_per_class  = 每个类别的 bbox/label 数量
images_per_class = Counter()
labels_per_class = Counter()

for split in SPLITS:
    split_dir = Path(LABEL_ROOT) / split
    if not split_dir.exists():
        print(f"[WARN] 不存在目录: {split_dir}")
        continue

    for txt_path in split_dir.glob("*.txt"):
        with open(txt_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

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
print("每个类别的label数量:", labels_counts_list)

# ================= 画图：图一（图像数量） =================
fig1, ax1 = plt.subplots(figsize=(10, 5))
bars1 = ax1.bar(range(num_classes), images_counts_list)
ax1.set_xticks(range(num_classes))
ax1.set_xticklabels(class_names, rotation=45, ha="right")
ax1.set_ylabel("Number of images")
ax1.set_title("Number of images per weed class (train+val+test)")

# 在每个柱子上标注数值
for bar in bars1:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",          # 如果想加千分位，用 f"{int(height):,}"
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()
plt.savefig("weed_images_per_class.png", dpi=300)

# ================= 画图：图二（label数量） =================
fig2, ax2 = plt.subplots(figsize=(10, 5))
bars2 = ax2.bar(range(num_classes), labels_counts_list)
ax2.set_xticks(range(num_classes))
ax2.set_xticklabels(class_names, rotation=45, ha="right")
ax2.set_ylabel("Number of labels (bboxes)")
ax2.set_title("Number of labels per weed class (train+val+test)")

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
plt.savefig("weed_labels_per_class.png", dpi=300)

plt.show()
