import pandas as pd

data = {
    "model": ["YOLOv8n", "YOLOv11n", "YOLOv11s"],
    "avg_preds": [38.5, 42.2, 45.1],
#    "mAC": [0.66, 0.69, 0.72],
}
df = pd.DataFrame(data)
out = r"E:\Project\mAC\experiments\test_file\table_avgpreds.csv"
df.to_csv(out, index=False, encoding="utf-8")
print("✅ Saved:", out)
