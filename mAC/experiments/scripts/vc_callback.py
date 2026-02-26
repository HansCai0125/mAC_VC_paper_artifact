import os, csv
import math
import matplotlib
matplotlib.use("Agg")           # Save figures in headless environments
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


class VCCallback:
    """
    VC proxy (Plan-1)
    VC50 = 2*mAP50*mAC/(mAP50+mAC+eps)
    VC50_95 = 2*mAP50_95*mAC/(mAP50_95+mAC+eps)

    Read:
      - mac_epoch.csv (last row): mAC, avg_preds
      - validator.metrics: mAP50 / mAP50-95 (compatible across versions)
      - validator.speed['inference']: latency/ms (optional)
    Write:
      - vc_epoch.csv：epoch, mAC, mAP50, mAP50-95, VC50, VC50_95, avg_preds, latency_ms
    """

    def __init__(self,
                 epsilon: float = 0.0,
                 out_csv: str = r"E:\Project\mAC\experiments\test_file\vc_epoch.csv",
                 mac_csv: str = r"E:\Project\mAC\experiments\test_file\mac_epoch.csv",
                 prefer_map50: bool = True):
        self.epsilon = float(epsilon)
        self.out_csv = out_csv
        self.mac_csv = mac_csv
        self.prefer_map50 = bool(prefer_map50)
        self._debug_once = False
        self._eps = 1e-12  # numerical stability for harmonic mean

    # ---------- helpers ----------
    def _read_last_from_mac_csv(self):
        """Read last row from mac_epoch.csv and return (epoch, mAC, avg_preds)."""
        epoch, mac, avg = 0, 0.0, 1.0
        try:
            if not os.path.exists(self.mac_csv):
                if not self._debug_once:
                    self._debug_once = True
                    print(f"[VC DEBUG] mac_csv not found: {self.mac_csv}")
                return epoch, mac, avg
            last = None
            with open(self.mac_csv, "r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    last = row
            if last is None:
                if not self._debug_once:
                    self._debug_once = True
                    print(f"[VC DEBUG] mac_csv empty: {self.mac_csv}")
                return epoch, mac, avg
            try:
                epoch = int(float(last.get("epoch", 0)))
            except Exception:
                epoch = 0
            mac = float(last.get("mAC", last.get("mac", 0.0)) or 0.0)
            avg = float(last.get("avg_preds", last.get("avg#preds", 1.0)) or 1.0)
        except Exception as e:
            if not self._debug_once:
                self._debug_once = True
                print(f"[VC DEBUG] read mac_csv failed: {e}")
        return epoch, mac, max(1.0, avg)

    def _latency_ms(self, validator) -> float:
        """Ultralytics validator.speed['inference'] (ms/img)."""
        spd = getattr(validator, "speed", None)
        if isinstance(spd, dict):
            try:
                return float(spd.get("inference", 0.0))
            except Exception:
                return 0.0
        return 0.0

    def _safe_maps(self, validator):
        """
        Get mAP with multi-version compatible structure:
        - metrics.box.map50 / metrics.box.map
        - metrics.map50 / metrics.map
        - metrics.results_dict['metrics/mAP50'], ['metrics/mAP50-95'] / 'map50' / 'map'
        """
        metrics = getattr(validator, "metrics", None)
        m50 = m5095 = None

        if metrics is not None:
            box = getattr(metrics, "box", None)
            if box is not None:
                if hasattr(box, "map50"):
                    try: m50 = float(box.map50)
                    except: pass
                if hasattr(box, "map"):
                    try: m5095 = float(box.map)
                    except: pass

            if m50 is None and hasattr(metrics, "map50"):
                try: m50 = float(metrics.map50)
                except: pass
            if m5095 is None and hasattr(metrics, "map"):
                try: m5095 = float(metrics.map)
                except: pass

            rd = getattr(metrics, "results_dict", None)
            if isinstance(rd, dict):
                if m50 is None:
                    for k in ("metrics/mAP50","map50","mAP50"):
                        if k in rd and rd[k] is not None:
                            try: m50 = float(rd[k]); break
                            except: pass
                if m5095 is None:
                    for k in ("metrics/mAP50-95","map","mAP50-95"):
                        if k in rd and rd[k] is not None:
                            try: m5095 = float(rd[k]); break
                            except: pass

        # Fallback
        if m50 is None:   m50 = 0.0
        if m5095 is None: m5095 = 0.0
        return m50, m5095

    def _get_epoch(self, validator) -> int:
        """Robust epoch lookup (mirrors EPICallback behavior)."""
        # 1) trainer.epoch
        t = getattr(validator, "trainer", None)
        if t is not None:
            e = getattr(t, "epoch", None)
            if isinstance(e, (int, float)):
                return int(e)

        # 2) validator.epoch
        e = getattr(validator, "epoch", None)
        if isinstance(e, (int, float)):
            return int(e)

        # 3) trainer.state["epoch"]
        st = getattr(t, "state", {}) if t is not None else {}
        if isinstance(st, dict) and "epoch" in st:
            try:
                return int(st["epoch"])
            except Exception:
                pass

        # 4) trainer.metrics dict
        tm = getattr(t, "metrics", None) if t is not None else None
        if isinstance(tm, dict) and "epoch" in tm:
            try:
                return int(tm["epoch"])
            except Exception:
                pass

        return 0

    # ---------- main hook ----------
    def on_val_end(self, validator):
        # 1) mAC & avg_preds
        epoch_from_mac, mAC, avg_preds = self._read_last_from_mac_csv()
        # 2) mAP50 / mAP50-95
        m50, m5095 = self._safe_maps(validator)
        # 3) latency
        lat_ms = self._latency_ms(validator)
        # 4) VC (harmonic mean)
        VC50 = (2.0 * m50 * mAC) / (m50 + mAC + self._eps)
        VC50_95 = (2.0 * m5095 * mAC) / (m5095 + mAC + self._eps)

        epoch = epoch_from_mac if epoch_from_mac is not None else self._get_epoch(validator)

        print(f"[VC] epoch={epoch} mAC={mAC:.3f} VC50={VC50:.3f}  "
              f"mAP50={m50:.3f} mAP50-95={m5095:.3f} VC50_95={VC50_95:.3f}  "
              f"avg={avg_preds:.2f} lat={lat_ms:.2f}ms")

        # 5) Append CSV (with epoch)
        try:
            os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)
            write_hdr = not os.path.exists(self.out_csv)
            with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_hdr:
                    w.writerow([
                        "epoch","mAC","mAP50","mAP50-95","VC50","VC50_95",
                        "avg_preds","latency_ms"
                    ])
                w.writerow([
                    epoch, mAC, m50, m5095, VC50, VC50_95,
                    avg_preds, lat_ms
                ])
        except Exception as e:
            print(f"[VC DEBUG] write CSV failed: {e}")



    def on_train_end(self, trainer):
        """
        After all training/validation is finished, generate plots from vc_epoch.csv
        (and optional mac_epoch.csv, epi_results.csv).
        Output files are saved in the same directory as vc_epoch.csv.
        """
        out_dir = os.path.dirname(self.out_csv) if self.out_csv else "."
        os.makedirs(out_dir, exist_ok=True)

        # Read vc_epoch.csv (required)
        try:
            df_vc = pd.read_csv(self.out_csv)
        except Exception as e:
            print(f"[VC PLOT] read {self.out_csv} failed: {e}")
            return

        # Optional: try reading mac_epoch.csv (for trend-length alignment)
        df_mac = None
        if os.path.exists(self.mac_csv):
            try:
                df_mac = pd.read_csv(self.mac_csv)
            except Exception:
                df_mac = None

        # Optional: try reading EPI (if you want to overlay it in trend plots)
        epi_csv_guess = os.path.join(out_dir, "epi_results.csv")
        df_epi = None
        if os.path.exists(epi_csv_guess):
            try:
                df_epi = pd.read_csv(epi_csv_guess)
            except Exception:
                df_epi = None

        # Align lengths using vc row count as the reference
        n = len(df_vc)
        if df_mac is not None and len(df_mac) >= n:
            df_mac = df_mac.iloc[:n].reset_index(drop=True)
        if df_epi is not None and len(df_epi) >= n:
            df_epi = df_epi.iloc[:n].reset_index(drop=True)

        # ---- Plot 1: trend comparison (mAC / VC / mAP50 / mAP50-95; optional EPI overlay) ----
        try:
            self._plot_trend(df_vc, df_epi, out_dir)
        except Exception as e:
            print(f"[VC PLOT] trend failed: {e}")

        # ---- Plot 2: correlation scatter (mAC vs mAP50, VC vs mAP50) ----
        try:
            self._plot_scatter(df_vc, out_dir)
        except Exception as e:
            print(f"[VC PLOT] scatter failed: {e}")

        try:
            self._plot_vc_compare(df_vc, out_dir)
        except Exception as e:
            print(f"[VC PLOT] vc-compare failed: {e}")

        print(f"[VC PLOT] all figures saved to {out_dir}")

    # --------- Trend Plot ----------
    def _plot_trend(self, df_vc: 'pd.DataFrame', df_epi: 'pd.DataFrame|None', out_dir: str):
        xs = range(len(df_vc))
        plt.figure(figsize=(10, 6))
        plt.plot(xs, df_vc["mAC"], marker="o", label="mAC")
        # Use VC50 here (not VC)
        plt.plot(xs, df_vc["VC50"], marker="s", label="VC50")
        plt.plot(xs, df_vc["mAP50"], marker="^", label="mAP@50")
        # Tolerant handling for mAP50-95 column names
        for cand in ("mAP50-95", "mAP50_95", "map50_95", "map5095", "mAP50-90"):
            if cand in df_vc.columns:
                plt.plot(xs, df_vc[cand], marker="x", label=cand.replace("_", "@").replace("map", "mAP"))
                break

        if df_epi is not None and "EPI" in df_epi.columns and len(df_epi):
            epi = df_epi["EPI"].astype(float).values[:len(df_vc)]
            if epi.max() != epi.min():
                epi_norm = (epi - epi.min()) / (epi.max() - epi.min())
                plt.plot(xs, epi_norm, linestyle="--", label="EPI (normalized)")

        plt.title("Trend: mAC / VC50 / mAP", fontsize=14)
        plt.xlabel("Epoch (row index)")
        plt.ylabel("Score")
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "vc_trend.png"), dpi=200)
        plt.close()

    def _plot_scatter(self, df_vc: 'pd.DataFrame', out_dir: str):
        x = pd.to_numeric(df_vc["mAP50"], errors="coerce")
        y_mac = pd.to_numeric(df_vc["mAC"], errors="coerce")
        # Use VC50 here as well
        y_vc = pd.to_numeric(df_vc["VC50"], errors="coerce")

        mask_mac = x.notna() & y_mac.notna()
        mask_vc = x.notna() & y_vc.notna()

        r_mac = float(pd.Series(x[mask_mac]).corr(pd.Series(y_mac[mask_mac]))) if mask_mac.sum() >= 2 else float("nan")
        r_vc = float(pd.Series(x[mask_vc]).corr(pd.Series(y_vc[mask_vc]))) if mask_vc.sum() >= 2 else float("nan")

        plt.figure(figsize=(6, 6))
        plt.scatter(x[mask_mac], y_mac[mask_mac], alpha=0.7, label=f"mAC vs mAP50 (r={r_mac:.3f})")
        plt.scatter(x[mask_vc], y_vc[mask_vc], alpha=0.7, label=f"VC50 vs mAP50 (r={r_vc:.3f})")

        try:
            if mask_mac.sum() >= 2:
                xs = np.linspace(x[mask_mac].min(), x[mask_mac].max(), 100)
                coef = np.polyfit(x[mask_mac], y_mac[mask_mac], 1)
                plt.plot(xs, np.polyval(coef, xs), linestyle="--")
        except:
            pass
        try:
            if mask_vc.sum() >= 2:
                xs = np.linspace(x[mask_vc].min(), x[mask_vc].max(), 100)
                coef = np.polyfit(x[mask_vc], y_vc[mask_vc], 1)
                plt.plot(xs, np.polyval(coef, xs), linestyle=":")
        except:
            pass

        plt.xlabel("mAP@50")
        plt.ylabel("mAC / VC50")
        plt.title("Correlation: mAC & VC50 vs mAP@50", fontsize=13)
        plt.grid(True, linestyle="--", alpha=0.35)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "vc_scatter_corr.png"), dpi=200)
        plt.close()

    def _plot_vc_compare(self, df_vc, out_dir: str):
        # Backward compatibility: if VC50 is missing, use VC
        vc_mac = pd.to_numeric(df_vc.get("VC50", df_vc.get("VC_mAC", df_vc.get("VC"))), errors="coerce").fillna(0.0)
        vc_m50 = pd.to_numeric(df_vc.get("VC50_95", df_vc.get("VC_mAP95", pd.Series([np.nan] * len(df_vc)))), errors="coerce")
        vc_m95 = vc_m50
        xs = np.arange(len(df_vc))

        # Summary stats displayed in scatter title
        valid_mask = ~(vc_mac.isna() | vc_m50.isna())
        r = float(np.corrcoef(vc_m50[valid_mask], vc_mac[valid_mask])[0, 1]) if valid_mask.sum() >= 2 else float("nan")
        mae = float(np.mean(np.abs(vc_mac[valid_mask] - vc_m50[valid_mask]))) if valid_mask.sum() >= 1 else float("nan")

        # 3 subplots: trend / scatter (equal axis) / delta-vs-epoch
        fig = plt.figure(figsize=(15, 5))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1])

        # (a) Trend
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(xs, vc_mac, marker="o", label="VC50")
        ax0.plot(xs, vc_m50, marker="s", label="VC50_95")
        if "VC50_95" in df_vc.columns: ax0.plot(xs, vc_m95, marker="^", label="VC50_95")
        ax0.set_title("VC Trend (mAC vs mAP)")
        ax0.set_xlabel("Epoch (row index)")
        ax0.set_ylabel("VC score")
        ax0.set_ylim(0, 1)
        ax0.grid(True, linestyle="--", alpha=0.35)
        ax0.legend()

        # (b) Scatter: VC50_95 vs VC50 (equal axis + y=x)
        ax1 = fig.add_subplot(gs[0, 1])
        ax1.scatter(vc_m50, vc_mac, alpha=0.85)
        ax1.plot([0, 1], [0, 1], linestyle="--", label="y = x")
        ax1.set_aspect('equal', 'box')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_xlabel("VC50_95")
        ax1.set_ylabel("VC50")
        ax1.set_title(f"VC50_95 vs VC50 (r={r:.3f}, MAE={mae:.3f})")
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend()

        # (c) Gap curve: Delta=VC50-VC50_95
        ax2 = fig.add_subplot(gs[0, 2])
        delta = (vc_mac - vc_m50).astype(float)
        ax2.plot(xs, delta, marker="o")
        ax2.axhline(0.0, linestyle="--")
        ax2.set_xlabel("Epoch (row index)")
        ax2.set_ylabel("Δ (VC50 - VC50_95)")
        ax2.set_title("Gap over Epochs")
        ax2.grid(True, linestyle="--", alpha=0.35)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "vc_compare.png"), dpi=200)
        plt.close()
