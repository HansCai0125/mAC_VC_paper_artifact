# Mean Area Coverage (mAC) and Visual Completeness (VC)

Code and results for **Complementing Instance Matching with Collective Regional Coverage: Mean Area Coverage and Visual Completeness**, by Han Cai, Huiyu Zhan and Mehrdad R. Kermani.

mAC and VC complement instance-matching metrics such as mAP, F1 and LRP:

- **mAC** measures target coverage by the union of same-class predictions.
- **VC** assigns each prediction to one target, measures their collective Dice agreement and penalizes orphan predictions.

The VC implementation is in [`region_eval/vc_v1.py`](region_eval/vc_v1.py). Earlier versions used a different VC definition and included EPI; those versions remain in Git history.

## Quick start

From the repository root, run the synthetic region-diagnostics example:

```bash
python -m region_eval --input examples/region_eval/synthetic_predictions.json
```

This example uses only the Python standard library and requires no dataset, model or GPU. It prints summary diagnostics and writes CSV reports under `results/region_eval/`. It demonstrates mAC and related region diagnostics; use `evaluate_vc` in `region_eval/vc_v1.py` for ownership-aware VC.

For the full test suite and experiments, use Python 3.11 and install the relevant dependencies: NumPy, PyTorch, Ultralytics, pandas, matplotlib, OpenCV, pycocotools and pytest. Then run:

```bash
python -m pytest tests
```

See the [reproduction guide](docs/reproduction.md) for recorded environment versions, data and model requirements, and the code and results corresponding to each paper table and figure.

## Repository

| Directory | Contents |
|---|---|
| `region_eval/` | Metrics, geometry, dataset loaders and analysis code |
| `experiments/` | Experiment entry points, including diagnostics and SAM evaluation |
| `results/` | Summary tables and figures |
| `training_logs/` | Configurations and logs for 35 detector training runs |
| `scripts/` | Figure generation |
| `docs/` | Experiment instructions and protocols |
| `tests/`, `examples/` | Tests and synthetic examples |

## Reproducing results

The experiments use MS-COCO 2017 and VisDrone2019-DET. Dataset images, detector checkpoints, SAM weights and per-image caches are not included. Full experiment reruns require these inputs and local dataset/weight paths to be configured.

Start with the [paper result index and setup instructions](docs/reproduction.md). Individual experiment notes are in [`docs/`](docs/). Archived result records retain their original paths and code hashes as provenance.

## License and citation

Code is released under the [MIT License](LICENSE). Datasets and third-party software retain their respective licenses. Citation details will be added on publication.
