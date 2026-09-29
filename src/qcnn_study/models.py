"""Standardized centroids and fixed equal-weight voting; no shipped weights."""

from dataclasses import dataclass

import numpy as np


def features(value):
    x = np.asarray(value, dtype=float)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError("Features require a finite nonempty sample-by-feature matrix")
    return x


def labels(value, size):
    y = np.asarray(value)
    if y.shape != (size,) or y.dtype.kind not in "iu" or not np.isin(y, (0, 1)).all():
        raise ValueError("Labels require one integer binary class per sample")
    return y.astype(int)


@dataclass(frozen=True)
class CentroidModel:
    mean: np.ndarray
    scale: np.ndarray
    centers: np.ndarray

    @classmethod
    def fit(cls, training_features, training_labels):
        x = features(training_features)
        y = labels(training_labels, len(x))
        if set(y) != {0, 1}:
            raise ValueError("Both classes are required in calibration")
        mean, scale = x.mean(axis=0), x.std(axis=0)
        if not np.isfinite(mean).all() or not np.isfinite(scale).all():
            raise ValueError("Feature magnitude exceeds numerical range")
        scale = np.where(scale == 0, 1., scale)
        standardized = (x-mean)/scale
        return cls(mean, scale, np.asarray([standardized[y == label].mean(axis=0) for label in (0, 1)]))

    def predict(self, value):
        x = features(value)
        if x.shape[1] != len(self.mean):
            raise ValueError("Feature dimensions changed")
        distances = np.sum((((x-self.mean)/self.scale)[:, None, :]-self.centers[None, :, :])**2, axis=2)
        return np.argmin(distances, axis=1)


def majority_predict(models, features_by_route):
    if not models or len(models) % 2 != 1 or len(models) != len(features_by_route):
        raise ValueError("Voting requires an odd, nonzero number of fixed routes")
    predictions = [model.predict(x) for model, x in zip(models, features_by_route, strict=True)]
    if len({len(p) for p in predictions}) != 1:
        raise ValueError("Routes must use the same image order and coverage")
    return (np.sum(predictions, axis=0) > len(models)//2).astype(int)


def evaluate_holdout(training, training_labels, holdout, holdout_labels, training_ids, holdout_ids):
    """No fitted value depends on holdout labels; IDs are checked before fitting."""
    from .statistics import accuracy_report

    if not training or len(training) != len(holdout):
        raise ValueError("Calibration and test route counts differ")
    train = [features(x) for x in training]
    test = [features(x) for x in holdout]
    for ids, size in ((training_ids, len(train[0])), (holdout_ids, len(test[0]))):
        if len(ids) != size or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != size:
            raise ValueError("IDs must be distinct nonempty strings with complete coverage")
    if set(training_ids).intersection(holdout_ids):
        raise ValueError("Calibration and test identities overlap")
    y_train = labels(training_labels, len(train[0]))
    y_test = labels(holdout_labels, len(test[0]))
    if any(len(x) != len(y_train) for x in train) or any(len(x) != len(y_test) for x in test):
        raise ValueError("Route sample coverage differs")
    models = [CentroidModel.fit(x, y_train) for x in train]
    predictions = majority_predict(models, test)
    return accuracy_report(predictions, y_test)
