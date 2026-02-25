import os, csv

class EPICallback:
    def __init__(self, alpha: float = 0.5, lam: float = 0.5,
                 out_csv: str = r"E:\Project\mAC\experiments\test_file\epi_results.csv",
                 prefer_map50: bool = True,
                 mac_csv: str = r"E:\Project\mAC\experiments\test_file\mac_epoch.csv"):
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.out_csv = out_csv
        self.prefer_map50 = bool(prefer_map50)
        self.mac_csv = mac_csv  # ★ 新增：只读这个
        self._debug_once = False

    # ---- 统一入口：把 on_val_end 传入的 obj 拆成 (trainer, validator) ----
    def _split(self, obj):
        # 如果 obj 看起来像 Validator（有 metrics），就当作 validator
        if hasattr(obj, "metrics") and not hasattr(obj, "optimizer"):
            v = obj
            t = getattr(v, "trainer", None)
        else:
            t = obj
            v = getattr(t, "validator", None)
        return t, v

    # ---- 读取 mAP：依次尝试 box.map50/map → metrics.map50/map → trainer.metrics(dict) → state ----
    def _get_map(self, obj) -> float:
        t, v = self._split(obj)

        # 1) validator.metrics 优先
        metrics = getattr(v, "metrics", None)
        if metrics is not None:
            # 1a) 新结构：metrics.box.map50 / metrics.box.map
            box = getattr(metrics, "box", None)
            if box is not None:
                if self.prefer_map50 and hasattr(box, "map50"):
                    try: return float(box.map50)
                    except Exception: pass
                if hasattr(box, "map"):
                    try: return float(box.map)
                    except Exception: pass
            # 1b) 旧结构：metrics.map50 / metrics.map
            if self.prefer_map50 and hasattr(metrics, "map50"):
                try: return float(metrics.map50)
                except Exception: pass
            if hasattr(metrics, "map"):
                try: return float(metrics.map)
                except Exception: pass
            # 1c) 某些版本把数据放 results_dict
            rd = getattr(metrics, "results_dict", None)
            if isinstance(rd, dict):
                keys_50 = ["metrics/mAP50", "map50", "mAP50"]
                keys_5095 = ["metrics/mAP50-95", "mAP50-95", "map"]
                if self.prefer_map50:
                    for k in keys_50:
                        if k in rd and rd[k] is not None:
                            try: return float(rd[k])
                            except Exception: pass
                for k in keys_5095:
                    if k in rd and rd[k] is not None:
                        try: return float(rd[k])
                        except Exception: pass

        # 2) trainer.metrics 可能是 dict
        tm = getattr(t, "metrics", None)
        if isinstance(tm, dict):
            keys_50 = ["metrics/mAP50", "map50", "mAP50"]
            keys_5095 = ["metrics/mAP50-95", "map", "mAP50-95"]
            if self.prefer_map50:
                for k in keys_50:
                    if k in tm and tm[k] is not None:
                        try: return float(tm[k])
                        except Exception: pass
            for k in keys_5095:
                if k in tm and tm[k] is not None:
                    try: return float(tm[k])
                    except Exception: pass

        # 3) 兜底：state
        st = getattr(t, "state", {}) if t is not None else {}
        if isinstance(st, dict):
            if "mAP" in st:
                try: return float(st["mAP"])
                except Exception: pass
            # 有些人会把 map50 写成这个 key
            if "map50" in st:
                try: return float(st["map50"])
                except Exception: pass

        # 首次拿不到时，打印一次字段调试，帮助定位
        if not self._debug_once:
            self._debug_once = True
            try:
                print("[EPI DEBUG] Could not read mAP. "
                      f"metrics={type(metrics)} dir={dir(metrics) if metrics else None} "
                      f"trainer.metrics keys={list(tm.keys()) if isinstance(tm, dict) else None} "
                      f"state={st}")
            except Exception:
                pass
        return 0.0

    def _get_mac(self, obj) -> float:
        t, v = self._split(obj)
        rd = getattr(getattr(v, "metrics", None), "results_dict", None)
        if isinstance(rd, dict) and "metrics/mAC" in rd:
            try:
                return float(rd["metrics/mAC"])
            except Exception:
                pass
        st = getattr(t, "state", {}) if t is not None else {}
        try:
            return float(st.get("mAC", 0.0))
        except Exception:
            return 0.0

    def _get_avg_preds(self, obj) -> float:
        t, v = self._split(obj)
        rd = getattr(getattr(v, "metrics", None), "results_dict", None)
        if isinstance(rd, dict) and "metrics/avg_preds" in rd:
            try:
                return max(1.0, float(rd["metrics/avg_preds"]))
            except Exception:
                pass
        st = getattr(t, "state", {}) if t is not None else {}
        try:
            return max(1.0, float(st.get("avg_preds", 1.0)))
        except Exception:
            return 1.0

    def _get_latency_s(self, obj) -> float:
        t, v = self._split(obj)
        # Ultralytics: validator.speed['inference'] (ms/img)
        spd = getattr(v, "speed", None)
        if isinstance(spd, dict):
            inf = spd.get("inference", None)
            if inf is not None:
                try:
                    return float(inf) / 1000.0
                except Exception:
                    pass
        # 兜底 trainer.state
        st = getattr(t, "state", {}) if t is not None else {}
        if isinstance(st, dict):
            if "latency_s" in st:
                try: return float(st["latency_s"])
                except Exception: pass
            if "latency_ms" in st:
                try: return float(st["latency_ms"]) / 1000.0
                except Exception: pass
        return 1e-3  # 最后兜底，避免除零

    def _get_epoch(self, obj) -> int:
        """尽可能稳健地获取当前 epoch。优先 trainer.epoch，其次 validator.epoch，再兜底 state/metrics。"""
        # 先拆 trainer / validator
        t, v = self._split(obj)

        # 1) 直接从 trainer.epoch
        if t is not None:
            e = getattr(t, "epoch", None)
            if isinstance(e, (int, float)):
                return int(e)

        # 2) 再从 validator.epoch
        if v is not None:
            e = getattr(v, "epoch", None)
            if isinstance(e, (int, float)):
                return int(e)

        # 3) 某些版本把 epoch 放进 trainer.state
        st = getattr(t, "state", {}) if t is not None else {}
        if isinstance(st, dict) and "epoch" in st:
            try:
                return int(st["epoch"])
            except Exception:
                pass

        # 4) 极端兜底：若 trainer.metrics 是 dict 也试一下
        tm = getattr(t, "metrics", None)
        if isinstance(tm, dict) and "epoch" in tm:
            try:
                return int(tm["epoch"])
            except Exception:
                pass

        # 5) 实在拿不到，就返回 0（避免 None）
        return 0

    def _read_mac_from_csv(self, epoch: int):
        """从 mac_csv 精确读取指定 epoch 的 mAC 和 avg_preds。找不到返回 (None, None)。"""
        try:
            if not os.path.exists(self.mac_csv):
                return None, None
            import csv
            with open(self.mac_csv, "r", newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                last_row = None
                for row in r:
                    # 允许 epoch 列是字符串
                    try:
                        e = int(float(row.get("epoch", 0)))
                    except Exception:
                        continue
                    if e == epoch:
                        # 命中当前 epoch
                        try:
                            mac = float(row.get("mAC", 0.0))
                        except Exception:
                            mac = None
                        try:
                            ap = float(row.get("avg_preds", 1.0))
                        except Exception:
                            ap = None
                        return mac, ap
                    # 没命中，记录一下，必要时可用最近一行兜底
                    last_row = row
                # 精确匹配失败：可选策略——返回最后一行（如果你希望“最近值兜底”）
                if last_row is not None:
                    try:
                        mac = float(last_row.get("mAC", 0.0))
                    except Exception:
                        mac = None
                    try:
                        ap = float(last_row.get("avg_preds", 1.0))
                    except Exception:
                        ap = None
                    return mac, ap
        except Exception as e:
            print(f"[EPI DEBUG] read mac_csv failed: {e}")
        return None, None

    def on_fit_epoch_end(self, trainer):
        epoch = int(getattr(trainer, "epoch", 0))
        mAP = self._get_map(trainer)

        # ★ 先从我们自己的 mAC CSV 读
        mAC, avg_preds = self._read_mac_from_csv(epoch)

        # 读不到再温和回退（可选）
        if mAC is None or avg_preds is None:
            v = getattr(trainer, "validator", None)
            rd = getattr(getattr(v, "metrics", None), "results_dict", None)
            if isinstance(rd, dict):
                try:
                    mAC = float(rd.get("metrics/mAC", mAC if mAC is not None else 0.0))
                except Exception:
                    pass
                try:
                    avg_preds = float(rd.get("metrics/avg_preds", avg_preds if avg_preds is not None else 1.0))
                except Exception:
                    pass
        if mAC is None:
            mAC = 0.0
        if avg_preds is None or avg_preds <= 0:
            avg_preds = 1.0

        latency_s = self._get_latency_s(trainer)
        latency_ms = latency_s * 1000.0
        combined = self.alpha * mAP + self.lam * mAC
        denom = max(1e-12, avg_preds * latency_ms)
        epi = combined / denom

        if not hasattr(trainer, "state") or trainer.state is None:
            trainer.state = {}
        trainer.state["EPI"] = float(epi)

        print(f"[EPI] epoch={epoch} mAP={mAP:.3f} mAC={mAC:.3f} avg={avg_preds:.2f} "
              f"lat={latency_s * 1000:.2f}ms EPI={epi:.4f} (α={self.alpha}, λ={self.lam})")

        try:
            os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)
            write_hdr = not os.path.exists(self.out_csv)
            with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
                import csv
                w = csv.writer(f)
                if write_hdr:
                    w.writerow(["epoch", "alpha", "lambda", "mAP", "mAC", "avg_preds", "latency_s", "EPI"])
                w.writerow([epoch, self.alpha, self.lam, mAP, mAC, avg_preds, latency_s, epi])
        except Exception as e:
            print(f"[EPI DEBUG] write CSV failed: {e}")

    # ---- 事件：验证结束，计算并落盘（支持 obj=validator 或 trainer） ----
    def on_val_end(self, obj):
        mAP = self._get_map(obj)
        mAC = self._get_mac(obj)
        avg_preds = self._get_avg_preds(obj)
        latency_s = self._get_latency_s(obj)

        combined = self.alpha * mAP + self.lam * mAC
        denom = max(1e-12, avg_preds * latency_s)
        epi = combined / denom

        # 对外暴露
        t, _ = self._split(obj)
        if t is not None:
            if not hasattr(t, "state"):
                t.state = {}
            t.state["EPI"] = float(epi)

        # ✅ 统一用稳健的 epoch
        epoch = self._get_epoch(obj)

        # 打印
        print(f"[EPI] epoch={epoch} mAP={mAP:.3f} mAC={mAC:.3f} avg={avg_preds:.2f} "
              f"lat={latency_s * 1000:.2f}ms EPI={epi:.4f} (α={self.alpha}, λ={self.lam})")

        # 追加 CSV
        try:
            os.makedirs(os.path.dirname(self.out_csv), exist_ok=True)
            write_hdr = not os.path.exists(self.out_csv)
            with open(self.out_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_hdr:
                    w.writerow(["epoch","alpha","lambda","mAP","mAC","avg_preds","latency_s","EPI"])
                w.writerow([epoch, self.alpha, self.lam, mAP, mAC, avg_preds, latency_s, epi])
        except Exception as e:
            print(f"[EPI DEBUG] write CSV failed: {e}")
