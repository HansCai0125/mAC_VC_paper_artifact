import math

import pytest

from region_eval.fragment_analysis import (
    REFERENCE_FRAGMENT_PRECISION,
    REFERENCE_UNIQUE_COVERAGE,
    _extended_counterfactual_documents,
    counterfactual_fragment_metrics,
    evaluate_fragment_aware,
)


def _rows_by_scenario():
    return {
        row["scenario"]: row
        for row in counterfactual_fragment_metrics()
    }


def test_complementary_halves_become_one_fragment_group_detection():
    row = _rows_by_scenario()["complementary_fragments"]

    assert row["standard_TP"] == 2
    assert row["standard_FP"] == 2
    assert row["fragment_TP"] == 2
    assert row["fragment_group_TP"] == 0
    assert row["fragment_support"] == 2
    assert row["fragment_FP"] == 0
    assert row["fragment_FN"] == 0
    assert row["fragment_F1"] == pytest.approx(1.0)


def test_exact_duplicate_still_counts_as_false_positive():
    row = _rows_by_scenario()["exact_duplicate"]

    assert row["fragment_TP"] == 2
    assert row["fragment_support"] == 0
    assert row["fragment_FP"] == 1
    assert row["fragment_FN"] == 0
    assert row["fragment_F1"] == pytest.approx(0.8)


def test_three_subthreshold_fragments_form_an_exact_group():
    document, _ = _extended_counterfactual_documents()[
        "three_fragment_group"
    ]
    metrics, gt_rows, decisions = evaluate_fragment_aware(
        document,
        confidence_threshold=0.25,
        unique_coverage_threshold=REFERENCE_UNIQUE_COVERAGE,
        fragment_precision_threshold=REFERENCE_FRAGMENT_PRECISION,
    )

    assert metrics["TP"] == 1
    assert metrics["fragment_group_TP"] == 1
    assert metrics["neutral_support_pred_count"] == 3
    assert metrics["FP"] == 0
    assert metrics["FN"] == 0
    assert metrics["F1"] == pytest.approx(1.0)
    assert gt_rows[0]["detection_type"] == "fragment_group"
    assert gt_rows[0]["group_iou"] == pytest.approx(1.0)
    assert {row["final_status"] for row in decisions} == {
        "group_support"
    }


def test_grazing_low_precision_extra_is_rejected():
    document, _ = _extended_counterfactual_documents()["grazing_extra"]
    metrics, _, decisions = evaluate_fragment_aware(
        document,
        confidence_threshold=0.25,
        unique_coverage_threshold=REFERENCE_UNIQUE_COVERAGE,
        fragment_precision_threshold=REFERENCE_FRAGMENT_PRECISION,
    )

    grazing = next(row for row in decisions if row["pred_id"] == "grazing")
    assert grazing["final_status"] == "rejected_fp"
    assert grazing["fragment_precision"] < REFERENCE_FRAGMENT_PRECISION
    assert metrics["TP"] == 2
    assert metrics["FP"] == 1
    assert metrics["FN"] == 0


def test_every_prediction_has_exactly_one_final_role():
    for document, _ in _extended_counterfactual_documents().values():
        metrics, _, decisions = evaluate_fragment_aware(
            document,
            confidence_threshold=0.25,
            unique_coverage_threshold=REFERENCE_UNIQUE_COVERAGE,
            fragment_precision_threshold=REFERENCE_FRAGMENT_PRECISION,
        )
        expected_predictions = sum(
            prediction["score"] >= 0.25
            for image in document["images"]
            for prediction in image["predictions"]
        )
        assert len(decisions) == expected_predictions
        assert metrics["n_pred"] == (
            metrics["single_anchor_TP"]
            + metrics["neutral_support_pred_count"]
            + metrics["FP"]
        )
        assert all(math.isfinite(row["score"]) for row in decisions)


@pytest.mark.parametrize(
    "keyword,value",
    [
        ("confidence_threshold", -0.01),
        ("unique_coverage_threshold", 1.01),
        ("fragment_precision_threshold", -0.01),
        ("group_iou_threshold", 1.01),
    ],
)
def test_invalid_thresholds_are_rejected(keyword, value):
    document, _ = _extended_counterfactual_documents()["perfect_base"]
    arguments = dict(
        confidence_threshold=0.25,
        unique_coverage_threshold=REFERENCE_UNIQUE_COVERAGE,
        fragment_precision_threshold=REFERENCE_FRAGMENT_PRECISION,
    )
    arguments[keyword] = value

    with pytest.raises(ValueError):
        evaluate_fragment_aware(document, **arguments)
