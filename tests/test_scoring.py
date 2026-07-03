import random

import pytest

from app.features import SwingFeatures
from app.scoring import fit_reference_model, score_swing, ReferenceModel


def make_reference_vectors(n=20, seed=1):
    rng = random.Random(seed)
    # 6 features, clustered tightly around known means with small noise
    means = [10.0, 160.0, 0.3, 0.15, 150.0, 0.01]
    return [[m + rng.uniform(-1.0, 1.0) for m in means] for _ in range(n)]


def test_fit_reference_model_requires_minimum_clips():
    with pytest.raises(ValueError):
        fit_reference_model(make_reference_vectors(n=5))


def test_fit_reference_model_succeeds_with_enough_clips():
    model = fit_reference_model(make_reference_vectors(n=20))
    assert isinstance(model, ReferenceModel)
    assert len(model.means) == 6


def test_score_swing_flags_no_findings_for_in_range_swing():
    model = fit_reference_model(make_reference_vectors(n=20))
    in_range_features = SwingFeatures(
        shoulder_rotation_deg=10.2, elbow_angle_deg=160.1, contact_height=0.31,
        contact_depth=0.16, knee_bend_deg=150.3, head_stability=0.011,
    )

    result = score_swing(model, in_range_features)

    assert result.findings == []
    assert result.overall_mahalanobis_distance < 3.0


def test_score_swing_flags_outlier_feature():
    model = fit_reference_model(make_reference_vectors(n=20))
    outlier_features = SwingFeatures(
        shoulder_rotation_deg=10.0, elbow_angle_deg=120.0,  # elbow way off reference
        contact_height=0.3, contact_depth=0.15, knee_bend_deg=150.0, head_stability=0.01,
    )

    result = score_swing(model, outlier_features)

    flagged = {f.feature for f in result.findings}
    assert "elbow_angle_deg" in flagged
    assert result.overall_mahalanobis_distance > 3.0


def test_reference_model_save_and_load_round_trip(tmp_path):
    model = fit_reference_model(make_reference_vectors(n=20))
    path = str(tmp_path / "model.pkl")

    model.save(path)
    loaded = ReferenceModel.load(path)

    assert (loaded.means == model.means).all()
