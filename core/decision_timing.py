"""Experimental nearest-frame decision cells; production supports stay unchanged.

Internal edges use actual center midpoints, including half samples. The outer
cells extend to audio edges and include the unanalysed incomplete frame tail.
Analysis windows are never changed by this timing convention.
"""
import math


def decision_cells(features: dict, duration: float) -> tuple[list[float], list[float]]:
    """Return one disjoint, positive cell per finite ordered in-audio center."""
    if not math.isfinite(duration) or duration < 0:
        raise ValueError('Duration must be finite and nonnegative')
    centers = features['centers']
    previous = -1.
    for center in centers:
        if not math.isfinite(center) or not 0 <= center < duration or center <= previous:
            raise ValueError('Centers must be finite, strictly ordered and inside audio')
        previous = center
    if not centers:
        return [], []
    edges = [0.] + [(a+b)/2 for a,b in zip(centers,centers[1:])] + [duration]
    if any(b <= a for a,b in zip(edges,edges[1:])):
        raise ValueError('Decision cells must have positive duration')
    return edges[:-1], edges[1:]
