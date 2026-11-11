import pytest

from region_eval.fragment_audit import (
    rejection_boundary_deficit,
    support_boundary_margin,
)


def decision(unique, precision):
    return dict(
        marginal_unique_coverage=unique,
        fragment_precision=precision,
    )


def test_support_margin_is_zero_on_either_active_boundary():
    assert support_boundary_margin(decision(.02, .75), .02, .50) == pytest.approx(0)
    assert support_boundary_margin(decision(.04, .50), .02, .50) == pytest.approx(0)


def test_support_margin_uses_nearest_normalized_boundary():
    assert support_boundary_margin(decision(.03, .75), .02, .50) == pytest.approx(.5)


def test_rejection_deficit_adds_only_failed_thresholds():
    assert rejection_boundary_deficit(decision(.01, .75), .02, .50) == pytest.approx(.5)
    assert rejection_boundary_deficit(decision(.03, .25), .02, .50) == pytest.approx(.5)
    assert rejection_boundary_deficit(decision(.01, .25), .02, .50) == pytest.approx(1.0)


@pytest.mark.parametrize("function", [support_boundary_margin, rejection_boundary_deficit])
def test_boundary_helpers_reject_invalid_thresholds(function):
    with pytest.raises(ValueError):
        function(decision(.02, .50), -0.01, .50)
    with pytest.raises(ValueError):
        function(decision(.02, .50), .02, 1.01)
