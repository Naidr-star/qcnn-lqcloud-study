"""Application metrics and explicitly conditional paired uncertainty."""

import math
from statistics import NormalDist

import numpy as np

from .models import labels


def wilson_interval(correct, total, confidence=.95):
    if type(correct) is not int or type(total) is not int or not 0 <= correct <= total or total <= 0 or not 0 < confidence < 1:
        raise ValueError("Invalid count or confidence level")
    z = NormalDist().inv_cdf((1+confidence)/2)
    p = correct/total
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    width = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0., center-width), min(1., center+width)]


def accuracy_report(predictions, expected):
    expected = np.asarray(expected)
    y = labels(expected, len(expected))
    pred = labels(predictions, len(y))
    if not len(y):
        raise ValueError("Empty evaluation")
    correct = int(np.sum(pred == y))
    matrix = [[int(np.sum((y == a) & (pred == b))) for b in (0, 1)] for a in (0, 1)]
    return {"correct": correct, "total": len(y), "accuracy": correct/len(y),
            "wilson_95": wilson_interval(correct, len(y)), "confusion_matrix": matrix,
            "scope": "Fixed fitted model and acquired test conditions; not a guarantee of future hardware stability."}


def paired_exact(baseline, candidate, expected):
    """Descriptive exact paired test; failure to reject is not equivalence."""
    y = labels(expected, len(expected))
    a, b = labels(baseline, len(y)), labels(candidate, len(y))
    if not len(y):
        raise ValueError("Empty paired evaluation")
    correct_a, correct_b = a == y, b == y
    gains = int(np.sum(~correct_a & correct_b))
    losses = int(np.sum(correct_a & ~correct_b))
    n = gains + losses
    p = min(1., 2*sum(math.comb(n, i) for i in range(min(gains, losses)+1))/(1 << n)) if n else 1.
    return {"baseline_only_correct": losses, "candidate_only_correct": gains,
            "both_correct": int(np.sum(correct_a & correct_b)),
            "both_wrong": int(np.sum(~correct_a & ~correct_b)),
            "accuracy_difference": (gains-losses)/len(y), "exact_mcnemar_p": p,
            "scope": "Descriptive paired test conditional on these predictions; not an equivalence test or a causal attribution."}


def paired_comparison(baseline, candidate, expected, *, draws, seed):
    if type(draws) is not int or draws <= 0 or type(seed) is not int or seed < 0:
        raise ValueError("Supply a positive resample count and nonnegative seed")
    y = labels(expected, len(expected))
    a, b = labels(baseline, len(y)), labels(candidate, len(y))
    if not len(y):
        raise ValueError("Empty paired evaluation")
    gains = int(np.sum((a != y) & (b == y)))
    losses = int(np.sum((a == y) & (b != y)))
    n = gains+losses
    p = min(1., 2*sum(math.comb(n, i) for i in range(min(gains, losses)+1))/(1 << n)) if n else 1.
    delta = (b == y).astype(float)-(a == y).astype(float)
    rng = np.random.default_rng(seed)
    values = np.zeros(draws)
    for label in np.unique(y):
        group = np.flatnonzero(y == label)
        indices = rng.choice(group, size=(len(group), draws), replace=True)
        values += np.sum(delta[indices], axis=0)/len(y)
    return {"accuracy_difference": float(delta.mean()), "candidate_only_correct": gains,
            "baseline_only_correct": losses, "exact_mcnemar_p": p,
            "paired_image_bootstrap_95": np.quantile(values, [.025, .975]).tolist(),
            "scope": "Class-stratified paired image resampling, conditional on measured predictions; excludes calibration uncertainty and temporal drift."}
