"""Learn what a normal 'good forehand' feature vector looks like from a
reference library and score new swings against it."""
from dataclasses import dataclass
from typing import List
import pickle

import numpy as np
from sklearn.covariance import EmpiricalCovariance

from .features import FEATURE_NAMES, SwingFeatures

FEATURE_Z_THRESHOLD = 1.5
MIN_REFERENCE_CLIPS = 10


@dataclass
class Finding:
    feature: str
    value: float
    reference_mean: float
    z_score: float
    severity: str


@dataclass
class ScoreResult:
    overall_mahalanobis_distance: float
    findings: List[Finding]


@dataclass
class ReferenceModel:
    means: np.ndarray
    stds: np.ndarray
    covariance: EmpiricalCovariance

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ReferenceModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_reference_model(feature_vectors: List[List[float]]) -> ReferenceModel:
    if len(feature_vectors) < MIN_REFERENCE_CLIPS:
        raise ValueError(
            f"Need at least {MIN_REFERENCE_CLIPS} reference clips to fit a model, "
            f"got {len(feature_vectors)}"
        )

    matrix = np.array(feature_vectors)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1e-6

    standardized = (matrix - means) / stds
    covariance = EmpiricalCovariance().fit(standardized)

    return ReferenceModel(means=means, stds=stds, covariance=covariance)


def score_swing(model: ReferenceModel, features: SwingFeatures) -> ScoreResult:
    vector = np.array(features.to_vector())
    z_scores = (vector - model.means) / model.stds

    squared_distance = model.covariance.mahalanobis(z_scores.reshape(1, -1))[0]
    overall_distance = float(squared_distance ** 0.5)

    findings = []
    for name, value, mean, z in zip(FEATURE_NAMES, vector, model.means, z_scores):
        if abs(z) >= FEATURE_Z_THRESHOLD:
            severity = "significant" if abs(z) >= 2.5 else "moderate"
            findings.append(
                Finding(
                    feature=name, value=float(value), reference_mean=float(mean),
                    z_score=float(z), severity=severity,
                )
            )

    return ScoreResult(overall_mahalanobis_distance=overall_distance, findings=findings)
