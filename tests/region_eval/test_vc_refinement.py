import pytest

from region_eval.vc_refinement import count_metrics, novelty_ratio


def test_count_metrics_matches_closed_form_f1():
    result = count_metrics(2, 1, 1)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(2 / 3)
    assert result["F1"] == pytest.approx(2 / 3)


def test_count_metrics_handles_empty_predictions():
    result = count_metrics(0, 0, 3)
    assert result["precision"] == 0
    assert result["recall"] == 0
    assert result["F1"] == 0


def test_count_metrics_rejects_negative_counts():
    with pytest.raises(ValueError):
        count_metrics(1, -1, 0)


def test_novelty_ratio_distinguishes_new_from_redundant_overlap():
    assert novelty_ratio(0, 50) == 0
    assert novelty_ratio(25, 50) == pytest.approx(.5)
    assert novelty_ratio(50, 50) == 1


def test_novelty_ratio_handles_zero_overlap_and_rounding_cap():
    assert novelty_ratio(0, 0) == 0
    assert novelty_ratio(50.0000000001, 50) == 1


def test_novelty_ratio_rejects_negative_area():
    with pytest.raises(ValueError):
        novelty_ratio(-1, 10)
    with pytest.raises(ValueError):
        novelty_ratio(1, -10)
