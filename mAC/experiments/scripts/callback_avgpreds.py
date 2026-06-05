# -*- coding: utf-8 -*-
# AvgPredsCallback: count predicted boxes per image during validation to compute avg_preds
from dataclasses import dataclass, field
from typing import List
import os, csv

@dataclass
class AvgPredsCallback:
    out_csv: str = "outputs/avgpreds_epoch.csv"
    _counts: List[int] = field(default_factory=list, init=False)

    def _count_boxes_in_pred(self, pred):
        # Compatible with common Ultralytics structures: pred.boxes.boxes (N,6) or pred.boxes with .xyxy/.conf/.cls
        try:
            if hasattr(pred, "boxes"):
                b = getattr(pred.boxes, "boxes", None)
                if b is not None:
                    return int(b.shape[0])
                # YOLOv8/YOLOv11: pred.boxes.xyxy tensor
                if hasattr(pred.boxes, "xyxy"):
                    return int(pred.boxes.xyxy.shape[0])
            # Fallback: pred is ndarray/list
            if hasattr(pred, "shape"):
                return int(pred.shape[0])
            if isinstance(pred, (list, tuple)):
                return len(pred)
        except Exception:
            pass
        return 0

    def on_val_batch_end(self, trainer):
        # Try reading per-image predictions from validator/preds
        try:
            v = getattr(trainer, "validator", None)
            preds = getattr(v, "pred", None) or getattr(v, "preds", None)
            if preds is None:
                preds = getattr(trainer, "preds", None)
            if preds is None:
                return
            # preds is usually one element per image
            for p in preds:
                self._counts.append(self._count_boxes_in_pred(p))
        except Exception:
            return

    def on_val_end(self, trainer):
        # Compute avg_preds and write to trainer.state for EPI
        if not hasattr(trainer, "state"):
            trainer.state = {}
        if self._counts:
            avg_preds = sum(self._counts) / max(1, len(self._counts))
        else:
            avg_preds = 1.0  # fallback to avoid division by zero
        trainer.state["avg_preds"] = float(avg_preds)

        # Append save
        os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)
        hdr = ["epoch", "avg_preds", "n_images"]
        write_hdr = not os.path.exists(self.out_csv)
        with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            if write_hdr: w.writeheader()
            w.writerow({
                "epoch": getattr(trainer, "epoch", 0),
                "avg_preds": avg_preds,
                "n_images": len(self._counts)
            })
        # Clear buffer for next epoch
        self._counts.clear()
