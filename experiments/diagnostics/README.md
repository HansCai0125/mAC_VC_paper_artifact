# mAC / VC 最后一轮真实输出补充实验

所有新代码和结果集中在本文件夹。旧预测、SAM mask、旧统计结果及论文均只读。

## 直接运行

在 PyCharm 中选择现有 `yolo` 环境，运行本目录的 `run_experiment.py`。

默认设置已经是：

```python
MODE = 'formal'
RUN_NAME = 'main'
```

也可以在项目根目录终端运行：

```powershell
conda activate yolo
python experiments/diagnostics/run_experiment.py
```

无需训练，也不会加载检测器或 SAM 权重。使用既有预测、连续 polygon 分析及 SAM 输出缓存。
依赖为 numpy、pandas、pycocotools；当前 yolo 环境已经通过导入检查。无需安装 sklearn 或 shapely。

## 四部分工作

1. **A：实例匹配未成功时，是否仍存在有用的前景覆盖。** 两数据集、三种 YOLO、各五个 seed；COCO 使用 polygon 真值验证，VisDrone 只作 box 层面的描述。
2. **B：额外覆盖是否富集真实前景。** COCO 三种模型、各五个 seed，每个可用 GT 使用 200 次确定性随机平移，同时计算面积参照、平移参照和密度差。
3. **C：VC 是否提供现有几何与实例指标以外的信息。** 复用已冻结的 4,000 张 held-out SAM 图像，固定 theta=0.25、每图至多 20 prompts。比较扩展基线与扩展基线+VC 的分组交叉验证 R²，另输出 owner-independent foreground recall 及两种 ownership 的机制检查。
4. **D2：受控标注合并。** 保留全部单例，以同类 GT 的连通分量形成区域，delta=0 为主分析，0.25/0.5 为敏感性分析。记录组大小、链式合并、背景占比、box-mask 覆盖差和 VC。

真实 crowd 分析 D1 没有用已过滤预测强行补做；当前注册缓存应用过 crowd 过滤，按方案选用 D2。E1/E2 和 F 是备选扩展，未自动扩大本次计算。

## 输出位置

正式输出：`results/main/`

- `protocol.json`：固定参数、输入身份与代码哈希。
- `diagnostics.sqlite`：每图事务保存的中间结果，支持断点续跑。
- `per_run/`：每个数据集/模型/seed 的目标级、区域级及 image-class 表。
- `reports/阅读结果.md`：结果阅读说明。
- `reports/main_results.csv`：主要效应与图像级区间。
- `reports/primary_tests_holm.csv`：12 个预定主比较的校正结果。
- `reports/A_seed_mean_SD.csv`、`B_C_seed_mean_SD.csv`：跨 seed 均值与样本标准差；统计区间与 seed SD 分开。
- `reports/A3_strata_*.csv`：分层边界、样本量、排除与各 seed 对比。
- `reports/C_oof_*.csv`、`C_folds_*.csv`：交叉验证审计。
- `reports/D2_per_seed.csv`、`exclusions.csv`：合并结果与排除原因。
- `complete.json`：全部完成后才生成。

看到 `COMPLETE: ... Ready for result review.` 后，把该目录保留，我们再一起解释结果。

## 中断与恢复

意外中断后，使用相同代码、参数和 RUN_NAME 重新运行即可。已保存的图像会跳过；不会删除旧结果。
程序会重新核对输入校验和。若修改了代码或参数，应使用新的 RUN_NAME；程序会拒绝把不同方案混入同一输出。
不要同时运行两个相同 RUN_NAME。程序使用操作系统锁，异常退出后锁会自动释放。

## 检查模式

```powershell
python experiments/diagnostics/run_checks.py
python experiments/diagnostics/run_experiment.py --mode preflight
python experiments/diagnostics/run_experiment.py --mode smoke --run-name my_smoke_v1
```

smoke 是每个模型 seed0 的 12 张固定图像、3 次平移、50 次 bootstrap、2 折×2 次 CV，只检查执行链，不能解释成论文结果。正式模式恢复 200 次平移、2,000 次 bootstrap、5 折×10 次 CV。

主要耗时在 B 的精确 polygon 平移计算，使用 CPU。暂不根据小样本估算完整耗时，因为 polygon 复杂度和预测数量差别很大。每张图完成即保存，并持续显示进度。

科学协议见 `PLAN.md`。不因结果方向改变阈值、分层、模型或主终点。
