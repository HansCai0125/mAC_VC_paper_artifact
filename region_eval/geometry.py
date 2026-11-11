"""Continuous rectangle geometry; no raster masks or inclusive-pixel '+1'."""

from math import fsum

Rect = tuple[float, float, float, float]


def area(rect: Rect) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def intersection(a: Rect, b: Rect) -> Rect | None:
    rect = (max(a[0], b[0]), max(a[1], b[1]),
            min(a[2], b[2]), min(a[3], b[3]))
    return rect if rect[2] > rect[0] and rect[3] > rect[1] else None


def union_area(rectangles: list[Rect]) -> float:
    """Exact geometric x-slabs with merged y-intervals (float roundoff remains).

    Input is finite, valid, continuous xyxy geometry from diagnostics.py.
    Worst-case time is O(k^2 log k), auxiliary space O(k).
    """
    rects = sorted(set(r for r in rectangles if area(r) > 0))
    xs = sorted({x for r in rects for x in (r[0], r[2])})
    slabs = []
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((r[1], r[3]) for r in rects
                           if r[0] < right and r[2] > left)
        lengths = []
        if not intervals:
            continue
        start, end = intervals[0]
        for lo, hi in intervals[1:]:
            if lo <= end:
                end = max(end, hi)
            else:
                lengths.append(end - start)
                start, end = lo, hi
        lengths.append(end - start)
        slabs.append((right - left) * fsum(lengths))
    return fsum(slabs)
