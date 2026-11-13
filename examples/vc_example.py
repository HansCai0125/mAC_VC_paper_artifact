"""Minimal mAC / VC example on hand-made boxes (standard library only).

Run from the repository root::

    python examples/vc_example.py

Each case has one 10 x 10 ground-truth box and a few same-class predictions.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from region_eval.vc_v1 import evaluate_vc


def document(predictions):
    """Wrap one image with a single ground-truth box in the cache format."""
    return dict(
        schema_version=1,
        metadata=dict(
            dataset="synthetic", split="example", model_id="hand_made",
            checkpoint_id="none", coordinate_frame="original_image_xyxy",
            ignore_crowd_policy="already_filtered",
            class_ids=[0], class_names=["object"],
            postprocessing=dict(stage="synthetic", confidence_floor=0.0,
                                nms_iou=None, max_det=100),
        ),
        images=[dict(
            image_id="image", width=100, height=100,
            gts=[dict(id="g", class_id=0, xyxy=[0, 0, 10, 10])],
            predictions=predictions,
        )],
    )


def box(name, xyxy):
    return dict(id=name, class_id=0, score=0.9, xyxy=xyxy)


CASES = {
    "one exact box": [box("p", [0, 0, 10, 10])],
    "two complementary halves": [box("left", [0, 0, 5, 10]),
                                 box("right", [5, 0, 10, 10])],
    "one half only": [box("left", [0, 0, 5, 10])],
    "exact box plus an orphan": [box("p", [0, 0, 10, 10]),
                                 box("stray", [50, 50, 60, 60])],
}


if __name__ == "__main__":
    print(f"{'case':28s} {'mAC':>6s} {'VC':>6s} {'F1':>6s} {'orphans':>8s}")
    for name, predictions in CASES.items():
        result, _, _ = evaluate_vc(document(predictions), theta=0.25)
        print(f"{name:28s} {result['mAC']:6.3f} {result['VC']:6.3f} "
              f"{result['standard_F1']:6.3f} {result['orphan_pred_count']:8d}")
