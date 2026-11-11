"""PyCharm entry point: merge completed RT-DETR results into paper tables."""

import json

from region_eval.rtdetr_paper_results import build_rtdetr_paper_results


if __name__ == "__main__":
    output = build_rtdetr_paper_results()
    print(json.dumps({"status": "complete", "output": str(output)}, ensure_ascii=False))
    print("SUCCESS: RT-DETR four-model paper tables and theta figure generated.")

