import os, csv, math, shutil
from pathlib import Path
import numpy as np

class OverallBestSaver:
    """
    Realtime choose best checkpoint by composite score:
        score = w_map * mAP + w_mac * mAC + w_epi * EPI*

    EPI* normalization: 'none' | 'minmax' | 'zscore'
    """

    def __init__(self,
                 w_map: float = 0.4,
                 w_mac: float = 0.4,
                 w_epi: float = 0.2,
                 epi_norm: str = "minmax",           # 'none' | 'minmax' | 'zscore'
                 log_csv_name: str = "overall_best_log.csv",
                 keep_snapshots: bool = False,        # False=只保留一个 overall_best.pt
                 verbose: bool = True,
                 alpha_for_epi: float = 0.5,          # 临时计算EPI时的 α, λ
                 lam_for_epi: float = 0.5,
                 prefer_source: str = "best"):        # 'best' | 'last' 复制哪份权重
        self.w_map = float(w_map)
        self.w_mac = float(w_mac)
        self.w_epi = float(w_epi)
        self.epi_norm = epi_norm
        self.log_csv_name = log_csv_name
        self.keep_snapshots = keep_snapshots
        self.verbose = verbose
        self.alpha_for_epi = float(alpha_for_epi)
        self.lam_for_epi = float(lam_for_epi)
        self.prefer_source = prefer_source

        # state
        self.best_score = -float("inf")
        self.best_epoch = None
        self._epi_vals = []     # for zscore
        self._epi_min = None    # for minmax
        self._epi_max = None

        # 这些在 on_train_start 里动态设置
        self.exp_dir = None
        self.metrics_dir = None
        self.mac_csv = None
        self._log_path = None

    # -------- helpers --------
    @staticmethod
    def _safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    def _read_mac_row(self, epoch: int):
        """
        从当前 run 的 metrics/mac_epoch.csv 精确匹配 epoch
        return (mAC, avg_preds, latency_ms or None)
        """
        if not self.mac_csv or not os.path.exists(self.mac_csv):
            return None, None, None
        try:
            hit = None
            with open(self.mac_csv, "r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    try:
                        e = int(float(row.get("epoch", -1)))
                    except Exception:
                        continue
                    if e == epoch:
                        hit = row
                        break
            if not hit:
                return None, None, None
            mAC = self._safe_float(hit.get("mAC"), None)
            avg = self._safe_float(hit.get("avg_preds"), 1.0)
            lat_ms = None
            if "latency_ms" in hit:
                lat_ms = self._safe_float(hit.get("latency_ms"), None)
            return mAC, avg, lat_ms
        except Exception:
            return None, None, None

    def _get_map(self, validator) -> float:
        metrics = getattr(validator, "metrics", None)
        if metrics is None:
            return 0.0
        # 新结构优先
        box = getattr(metrics, "box", None)
        if box is not None:
            if hasattr(box, "map50"):
                try: return float(box.map50)
                except: pass
            if hasattr(box, "map"):
                try: return float(box.map)
                except: pass
        # 老结构
        for k in ("map50", "mAP50", "map"):
            if hasattr(metrics, k):
                try: return float(getattr(metrics, k))
                except: pass
        # results_dict 兜底
        rd = getattr(metrics, "results_dict", None)
        if isinstance(rd, dict):
            for k in ("metrics/mAP50", "map50", "mAP50", "metrics/mAP50-95", "map"):
                if k in rd and rd[k] is not None:
                    try: return float(rd[k])
                    except: pass
        return 0.0

    def _get_epoch(self, trainer, validator) -> int:
        for obj in (trainer, validator):
            if obj is None:
                continue
            e = getattr(obj, "epoch", None)
            if isinstance(e, (int, float)):
                return int(e)
        st = getattr(trainer, "state", {}) if trainer is not None else {}
        if isinstance(st, dict) and "epoch" in st:
            try: return int(st["epoch"])
            except: pass
        return 0

    def _latency_ms(self, validator) -> float | None:
        spd = getattr(validator, "speed", None)
        if isinstance(spd, dict):
            v = spd.get("inference", None)
            if v is not None:
                try: return float(v)
                except: pass
        return None

    def _epi_norm(self, epi_raw: float) -> float:
        if self.epi_norm == "none":
            return epi_raw
        if self.epi_norm == "minmax":
            if self._epi_min is None or epi_raw < self._epi_min:
                self._epi_min = epi_raw
            if self._epi_max is None or epi_raw > self._epi_max:
                self._epi_max = epi_raw
            if self._epi_max == self._epi_min:
                return 0.5
            return (epi_raw - self._epi_min) / (self._epi_max - self._epi_min)
        # zscore
        self._epi_vals.append(epi_raw)
        if len(self._epi_vals) < 5:
            return 0.5
        arr = np.asarray(self._epi_vals, dtype=float)
        mu, sigma = float(arr.mean()), float(arr.std() + 1e-12)
        z = (epi_raw - mu) / sigma
        return 1.0 / (1.0 + math.exp(-z))

    def _ensure_log_header(self, log_path: Path):
        if not log_path.exists():
            with open(log_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["epoch","mAP","mAC","EPI_raw","EPI_norm","score"])

    def _save_overall_best(self, exp_dir: Path, epoch: int) -> tuple[bool, str]:
        """返回 (ok, msg)。只保留一份 overall_best.pt；若 keep_snapshots=True，则额外保留 epoch 快照。"""
        wdir = exp_dir / "weights"
        wdir.mkdir(exist_ok=True, parents=True)

        # 选择源权重
        src = wdir / (f"{self.prefer_source}.pt")
        if not src.exists():
            # 兜底换另一个
            alt = "last" if self.prefer_source == "best" else "best"
            src = wdir / f"{alt}.pt"
            if not src.exists():
                return False, f"no source weights found (best.pt/last.pt) in {wdir}"

        # 删除旧的 overall_best*（只保留最新）
        for f in wdir.glob("overall_best*.pt"):
            try: f.unlink()
            except: pass

        # 复制为 overall_best.pt
        dst = wdir / "overall_best.pt"
        shutil.copy2(src, dst)

        # 如需保留快照（可选）
        if self.keep_snapshots:
            snap = wdir / f"overall_best_epoch{epoch}.pt"
            shutil.copy2(src, snap)

        return True, str(dst)

    # -------- callbacks --------
    def on_train_start(self, trainer):
        # reset state per run
        self.best_score = -float("inf")
        self.best_epoch = None
        self._epi_vals.clear()
        self._epi_min = None
        self._epi_max = None

        # 当前 run 的目录
        self.exp_dir = Path(getattr(trainer, "save_dir", "."))
        self.metrics_dir = self.exp_dir / "metrics"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.mac_csv = str(self.metrics_dir / "mac_epoch.csv")  # ✅ 动态定位本 run 的 mAC

        # 准备日志
        self._log_path = self.exp_dir / self.log_csv_name
        self._ensure_log_header(self._log_path)

    def on_fit_epoch_end(self, trainer):
        validator = getattr(trainer, "validator", None)
        epoch = self._get_epoch(trainer, validator)

        # --- gather metrics ---
        mAP = self._get_map(validator)

        # mAC / avg / latency from mac csv (exact epoch)
        mAC, avg_preds, lat_ms_csv = self._read_mac_row(epoch)
        if mAC is None:
            # fallback: results_dict
            rd = getattr(getattr(validator, "metrics", None), "results_dict", None)
            if isinstance(rd, dict) and "metrics/mAC" in rd:
                try: mAC = float(rd["metrics/mAC"])
                except: mAC = None
        if mAC is None:
            mAC = 0.0
        if avg_preds is None:
            avg_preds = 1.0
        lat_ms = lat_ms_csv if lat_ms_csv is not None else (self._latency_ms(validator) or 1.0)

        # EPI: prefer EPICallback state
        epi_raw = None
        st = getattr(trainer, "state", {})
        if isinstance(st, dict) and "EPI" in st:
            try: epi_raw = float(st["EPI"])
            except: epi_raw = None
        if epi_raw is None:
            # 临时估算：Q / (avg_preds * latency)
            Q = self.alpha_for_epi * (mAP or 0.0) + self.lam_for_epi * (mAC or 0.0)
            denom = max(1e-9, float(avg_preds) * float(lat_ms))
            epi_raw = Q / denom

        epi_norm = self._epi_norm(epi_raw)

        # 综合分
        score = (self.w_map * (mAP or 0.0)) + (self.w_mac * (mAC or 0.0)) + (self.w_epi * epi_norm)

        # 记录日志
        with open(self._log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, mAP, mAC, epi_raw, epi_norm, score])

        if self.verbose:
            print(f"[OVERALL] epoch={epoch}  mAP={mAP:.4f}  mAC={mAC:.4f}  "
                  f"EPI*={epi_norm:.4f}  score={score:.4f}  (best={self.best_score:.4f})")

        # 若更优则保存
        if score > self.best_score:
            ok, msg = self._save_overall_best(self.exp_dir, epoch)   # ✅ 这里传 exp_dir，且接收 (ok,msg)
            if self.verbose:
                if ok:
                    print(f"[OVERALL] ↑ improved → saved overall_best.pt @ epoch {epoch} -> {msg}")
                else:
                    print(f"[OVERALL] WARN: {msg}")
            self.best_score = score
            self.best_epoch = epoch

    def on_train_end(self, trainer):
        if self.verbose:
            print(f"[OVERALL] training finished. Best epoch = {self.best_epoch}, Best score = {self.best_score:.4f}")
