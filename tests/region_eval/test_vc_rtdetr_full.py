from pathlib import Path

from region_eval.offline import evaluate_document
from region_eval.vc_rtdetr_full import (
    EXPECTED_COMBINATIONS,
    _predict_rtdetr,
)


class _Array:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.value


class _Boxes:
    xyxy = _Array([[1.0, 2.0, 11.0, 12.0], [4.0, 5.0, 14.0, 15.0]])
    conf = _Array([0.9, 0.4])
    cls = _Array([2.0, 3.0])


class _Result:
    def __init__(self, path):
        self.path = str(path)
        self.orig_shape = (40, 60)
        self.boxes = _Boxes()


class _Model:
    def __init__(self, path):
        self.path = path
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return [_Result(self.path)]


def test_rtdetr_suite_has_one_coco_and_five_visdrone_runs():
    assert EXPECTED_COMBINATIONS == {
        ("mscoco", 0),
        ("visdrone", 0),
        ("visdrone", 1),
        ("visdrone", 2),
        ("visdrone", 3),
        ("visdrone", 4),
    }


def test_rtdetr_predict_uses_native_postprocessing_without_nms_arguments(tmp_path):
    image = tmp_path / "image.jpg"
    image.touch()
    model = _Model(image)

    rows = _predict_rtdetr(
        model,
        image,
        width=60,
        height=40,
        device="cpu",
        imgsz=640,
        confidence_floor=0.001,
        max_det=1000,
        output=Path(tmp_path),
    )

    assert "iou" not in model.kwargs
    assert "agnostic_nms" not in model.kwargs
    assert model.kwargs["conf"] == 0.001
    assert model.kwargs["max_det"] == 1000
    assert rows == [
        dict(
            id="pred_000000",
            xyxy=[1.0, 2.0, 11.0, 12.0],
            score=0.9,
            class_id=2,
        ),
        dict(
            id="pred_000001",
            xyxy=[4.0, 5.0, 14.0, 15.0],
            score=0.4,
            class_id=3,
        ),
    ]


def test_offline_evaluation_accepts_native_rtdetr_postprocessing():
    document = {
        "schema_version": 1,
        "metadata": {
            "dataset": "test",
            "split": "val",
            "model_id": "RT-DETR-L",
            "checkpoint_id": "seed0",
            "ignore_crowd_policy": "already_filtered",
            "coordinate_frame": "original_image_xyxy",
            "class_ids": [0],
            "postprocessing": {
                "stage": "native_rtdetr_postprocess_no_nms",
                "confidence_floor": 0.001,
                "nms_iou": None,
                "max_det": 1000,
            },
        },
        "images": [
            {
                "image_id": "image0",
                "width": 20,
                "height": 20,
                "gts": [
                    {"id": "gt0", "xyxy": [2.0, 2.0, 10.0, 10.0], "class_id": 0}
                ],
                "predictions": [],
            }
        ],
    }

    config, images, summary = evaluate_document(document, theta=0.25)

    assert config.theta == 0.25
    assert len(images) == 1
    assert summary["n_images"] == 1
