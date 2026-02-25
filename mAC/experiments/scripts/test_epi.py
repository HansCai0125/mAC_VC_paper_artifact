import os
import sys
from ultralytics import YOLO
from PIL import Image

sys.path.append(r"E:\Project\mAC")
sys.path.append(r"E:\Project\mAC\experiments\scripts")
from callback_mac import MACCallback
from epi_callback import EPICallback
from vc_callback import VCCallback

def main():
    # 1️⃣ 模型与数据路径
    model_path = r"E:\Project\mAC\yolo11m.pt"
    data_path = r""
    model = YOLO(model_path)

    # 2️⃣ 初始化三个回调
    mac_cb = MACCallback(
        tau=0.50,
        score_thresh=0.25,
        class_aware=True,
        mac_out_csv=r"E:\Project\mAC\experiments\test_file\mac_epoch.csv"  # ★ 同一条路径
    )

    epi_cb = EPICallback(
        alpha=0.5, lam=0.5,
        out_csv=r"E:\Project\mAC\experiments\test_file\epi_results.csv",
        prefer_map50=True,
        mac_csv=r"E:\Project\mAC\experiments\test_file\mac_epoch.csv"  # ★ 同一条路径
    )

    vc_cb = VCCallback(epsilon=0.01,
                       out_csv=r"E:\Project\mAC\experiments\test_file\vc_epoch.csv",
                       mac_csv=r"E:\Project\mAC\experiments\test_file\mac_epoch.csv")

    model.add_callback("on_val_start", mac_cb.on_val_start)
    model.add_callback("on_val_end", mac_cb.on_val_end)
    model.add_callback("on_train_start", mac_cb.on_train_start)
    model.add_callback("on_train_end",   mac_cb.on_train_end)
    model.add_callback("on_fit_epoch_end", epi_cb.on_fit_epoch_end)
    model.add_callback("on_val_end", vc_cb.on_val_end)  # 新增 VC 每轮一行
    model.add_callback("on_train_end", vc_cb.on_train_end)

    # 4️⃣ 训练配置（完全模仿原模板）
    results = model.train(
        data=data_path,
        epochs=500,
        patience=100,
        batch=8,
        imgsz=640,
        pretrained=True,
        optimizer="SGD",
        cos_lr=True,
        close_mosaic=10,
        amp=True,
        rect=False,
        workers=12,
        freeze=0,
        val=True,             # 每个 epoch 都会验证 → 回调自动执行
        iou=0.7,
        max_det=500,
        plots=True,
        # Hyperparams
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        nbs=64,
        device=0,
        name="test",
    )

    # 5️⃣ 输出结果路径
    output_dir = model.trainer.save_dir
    print("\n✅ Training Finished")
    print(f"📦 Best Weights: {os.path.join(output_dir, 'weights', 'best.pt')}")
    print(f"📈 Results Plot: {os.path.join(output_dir, 'results.png')}")
    print(f"📊 mAC CSV: {os.path.join(output_dir, 'results_mac.csv')}")
    print(f"📊 EPI CSV: {r'E:/Project/mAC/experiments/test_file/epi_results.csv'}")

    try:
        Image.open(os.path.join(output_dir, "results.png")).show()
    except Exception:
        print("⚠️ results.png 无法在当前环境中打开。")


if __name__ == "__main__":
    main()
