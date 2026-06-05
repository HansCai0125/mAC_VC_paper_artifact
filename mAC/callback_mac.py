from typing import List, Dict, Any, Tuple, Union
import numpy as np
from mean_AC import Box, image_ac
import types
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv, os

Array = Union[np.ndarray, "torch.Tensor"]

class MACCallback:
    def __init__(self, tau: float = 0.5, score_thresh: float = 0.25, class_aware: bool = True,
                 mac_out_csv: str = "outputs/mac_epoch.csv"):
        self.tau = tau
        self.score_thresh = score_thresh
        self.class_aware = class_aware
        self.mac_out_csv = mac_out_csv  # Added: project-owned CSV output
        self.reset()

    def reset(self):
        self._all_cov: List[float] = []
        self._tp: int = 0
        self._total: int = 0
        self.last_mac: float = 0.0
        self.last_recall: float = 0.0
        self._orig_update_metrics = None  #Keep origin function
        self._pred_boxes_total = 0
        self._pred_images_total = 0

    @staticmethod
    def _np(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().numpy()
        try:
            return np.asarray(x)
        except Exception:
            return None

    def _init_epoch_counter(self):
        if not hasattr(self, "_epoch_counter"):
            self._epoch_counter = -1

    def _bump_epoch(self):
        self._init_epoch_counter()
        self._epoch_counter += 1

    def _get_epoch(self, validator) -> int:
        # 1) validator.epoch
        e = getattr(validator, "epoch", None)
        if isinstance(e, (int, float)):
            return int(e)
        # 2) trainer.epoch
        t = getattr(validator, "trainer", None)
        if t is not None:
            e = getattr(t, "epoch", None)
            if isinstance(e, (int, float)):
                return int(e)
        self._init_epoch_counter()
        return int(self._epoch_counter)

    def on_train_start(self, trainer):
        self._epoch_counter = -1

        os.makedirs(os.path.dirname(self.mac_out_csv), exist_ok=True)
        with open(self.mac_out_csv, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "mAC", "avg_preds"])
        print(f"[mAC] reset csv at train start -> {self.mac_out_csv}")

    # --------- GT: (g_cls, g_xywh_norm) ---------
    def _extract_pred_xyxy_pixel_for_image(self, preds_any, bi: int) -> List[Box]:
        boxes: List[Box] = []

        # Case 1：list/tuple
        if isinstance(preds_any, (list, tuple)):
            if bi >= len(preds_any):
                return boxes
            item = preds_any[bi]

            # 1a) list[Results]
            if hasattr(item, "boxes"):
                b = item.boxes
                if b is not None and len(b) > 0:
                    xyxy = self._np(b.xyxy)
                    conf = self._np(b.conf)
                    pcl = self._np(b.cls)
                    if xyxy is not None and conf is not None and pcl is not None:
                        for (x1, y1, x2, y2), s, c in zip(xyxy, conf, pcl):
                            boxes.append(Box(float(x1), float(y1), float(x2), float(y2),
                                             cls=int(c), conf=float(s)))
                return boxes

            # 1b) list[(N,6)]
            arr = self._np(item)
            if arr is not None and arr.ndim == 2 and arr.shape[1] >= 6:
                for x1, y1, x2, y2, s, c in arr:
                    boxes.append(Box(float(x1), float(y1), float(x2), float(y2),
                                     cls=int(c), conf=float(s)))
                return boxes

            # Others
            return boxes

        # Csae 2
        arr = self._np(preds_any)
        if arr is not None and arr.ndim == 2 and arr.shape[1] >= 6:
            for x1, y1, x2, y2, s, c in arr:
                boxes.append(Box(float(x1), float(y1), float(x2), float(y2),
                                 cls=int(c), conf=float(s)))
        return boxes


    def on_train_end(self, trainer):
        try:
            xs, macs, avgs = [], [], []
            if not os.path.exists(self.mac_out_csv):
                print(f"[mAC PLOT] skip, csv not found: {self.mac_out_csv}")
                return

            with open(self.mac_out_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    xs.append(int(float(row["epoch"])))
                    macs.append(float(row["mAC"]))
                    avgs.append(float(row.get("avg_preds", "1.0")))

            order = np.argsort(xs)
            xs = [xs[i] for i in order]
            macs = [macs[i] for i in order]
            avgs = [avgs[i] for i in order]

            last_idx = {}
            for i, e in enumerate(xs):
                last_idx[e] = i
            epochs = sorted(last_idx.keys())
            macs = [macs[last_idx[e]] for e in epochs]
            avgs = [avgs[last_idx[e]] for e in epochs]
            xs = epochs

            save_dir = getattr(trainer, "save_dir", None)
            out_dir = save_dir if save_dir else os.path.dirname(self.mac_out_csv)
            os.makedirs(out_dir, exist_ok=True)
            out_png = os.path.join(out_dir, "mac_curves.png")

            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(xs, macs, marker="o")
            plt.title("mAC")
            plt.xlabel("epoch")
            plt.ylabel("mAC")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.ylim(0.0, 1.0)

            plt.subplot(1, 2, 2)
            plt.plot(xs, avgs, marker="o")
            plt.title("avg_preds")
            plt.xlabel("epoch")
            plt.ylabel("avg preds per image")
            plt.grid(True, linestyle="--", alpha=0.3)

            plt.tight_layout()
            plt.savefig(out_png, dpi=200)
            plt.close()
            print(f"[mAC PLOT] saved: {out_png}")
        except Exception as e:
            print(f"[mAC PLOT] failed: {e}")


    # ================== Core：hook update_metrics ==================
    def on_val_start(self, validator):
        self._all_cov = []
        self._tp = 0
        self._total = 0
        self._pred_boxes_total = 0
        self._pred_images_total = 0
        self._bump_epoch()

        self._trainer_ref = getattr(validator, "trainer", None)

        validator._mac_debug_printed_this_epoch = False

        if self._orig_update_metrics is not None:
            return
        self._orig_update_metrics = validator.update_metrics

        def _patched_update_metrics(self_validator, preds, batch):
            if not getattr(self_validator, "_mac_debug_printed_this_epoch", False):
                print(f"[mAC DEBUG] ✅ update_metrics is active! "
                      f"preds type={type(preds)}, batch keys={list(batch.keys())}")
                self_validator._mac_debug_printed_this_epoch = True

            out = self._orig_update_metrics(preds, batch)

            try:
                #Img size(B,C,H,W)
                img = batch.get("img", None)
                if img is None or not hasattr(img, "shape"):
                    return out
                bs, _, H, W = img.shape

                # GT（xywh-norm）
                to_np = (lambda x: x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x))
                bboxes_all = to_np(batch["bboxes"])
                cls_all = to_np(batch["cls"])
                if cls_all.ndim == 2 and cls_all.shape[-1] == 1:
                    cls_all = cls_all.squeeze(-1)
                bidx_all = to_np(batch["batch_idx"])

                for bi in range(bs):
                    mask = (bidx_all == bi)
                    g_xywh = bboxes_all[mask]
                    g_cls = cls_all[mask]

                    # GT: xywh-norm → xyxy-pixel
                    gts = []
                    for (cx, cy, bw, bh), c in zip(g_xywh, g_cls):
                        x1 = (cx - bw / 2.0) * W
                        y1 = (cy - bh / 2.0) * H
                        x2 = (cx + bw / 2.0) * W
                        y2 = (cy + bh / 2.0) * H
                        gts.append(Box(float(x1), float(y1), float(x2), float(y2), cls=int(c)))

                    pred_boxes = []
                    n_boxes_this_img = 0

                    if isinstance(preds, (list, tuple)) and bi < len(preds):
                        item = preds[bi]

                        if hasattr(item, "boxes"):  # Ultralytics Results branch
                            b = item.boxes
                            if b is not None and len(b) > 0:
                                xyxy = to_np(b.xyxy)
                                conf = to_np(b.conf)
                                pcl = to_np(b.cls)
                                if xyxy is not None and conf is not None and pcl is not None:
                                    keep = (conf >= self.score_thresh).astype(bool).ravel()
                                    n_boxes_this_img = int(keep.sum())
                                    for (x1, y1, x2, y2), s, c, k in zip(xyxy, conf, pcl, keep):
                                        if not k:
                                            continue
                                        pred_boxes.append(Box(float(x1), float(y1), float(x2), float(y2),
                                                              cls=int(c), conf=float(s)))
                        else:
                            arr = to_np(item)
                            if arr is not None and arr.ndim == 2 and arr.shape[1] >= 6:
                                keep = arr[:, 4] >= self.score_thresh
                                n_boxes_this_img = int(keep.sum())
                                for x1, y1, x2, y2, s, c in arr[keep]:
                                    pred_boxes.append(Box(float(x1), float(y1), float(x2), float(y2),
                                                          cls=int(c), conf=float(s)))
                    _, covs, det = image_ac(
                        gts, pred_boxes,
                        tau=self.tau,
                        score_thresh=self.score_thresh,
                        class_aware=self.class_aware
                    )
                    self._all_cov.extend(covs)
                    self._tp += int(sum(det))
                    self._total += int(len(gts))

                    self._pred_boxes_total += n_boxes_this_img
                    self._pred_images_total += 1

            except Exception as e:
                print(f"[mAC DEBUG]: Exception in patched update_metrics: {e}")

            return out
        validator.update_metrics = types.MethodType(_patched_update_metrics, validator)

    def on_val_end(self, validator):
        mac = float(np.mean(self._all_cov)) if self._all_cov else 0.0
        recall = (self._tp / self._total) if self._total else 0.0
        avg_preds = (self._pred_boxes_total / self._pred_images_total) if self._pred_images_total > 0 else 1.0

        self.last_mac = mac
        self.last_recall = recall

        print(
            f"[mAC] mAC={mac:.4f}  Recall@AC{self.tau:.2f}={recall:.4f}  N_gt={self._total}  avg_preds={avg_preds:.2f}")

        epoch_val = self._get_epoch(validator)

        try:
            os.makedirs(os.path.dirname(self.mac_out_csv), exist_ok=True)
            exists2 = os.path.exists(self.mac_out_csv)
            with open(self.mac_out_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if not exists2:
                    w.writerow(["epoch", "mAC", "avg_preds"])
                w.writerow([epoch_val, mac, avg_preds])
            if not hasattr(self, "_mac_csv_notice"):
                print(f"[mAC INFO] wrote epoch metrics to: {self.mac_out_csv}")
                self._mac_csv_notice = True
        except Exception as e:
            print(f"[mAC ERROR] write mac_out_csv failed: {e}")
