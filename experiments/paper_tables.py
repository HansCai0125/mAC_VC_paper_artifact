"""Entry point: merge completed RT-DETR results into paper tables."""

from pathlib import Path as _Path
import sys as _sys

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import json

from region_eval.rtdetr_paper_results import build_rtdetr_paper_results


if __name__ == "__main__":
    output = build_rtdetr_paper_results()
    print(json.dumps({"status": "complete", "output": str(output)}, ensure_ascii=False))
    print("SUCCESS: RT-DETR four-model paper tables and theta figure generated.")

