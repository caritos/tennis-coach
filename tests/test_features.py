import pytest

from app.pose import Landmark, FrameLandmarks
from app.phases import SwingPhases
from app.features import compute_features, FEATURE_NAMES, SwingFeatures

NUM_LANDMARKS = 33
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
RIGHT_ELBOW, RIGHT_WRIST = 14, 16
LEFT_HIP, RIGHT_HIP = 23, 24
RIGHT_KNEE, RIGHT_ANKLE = 26, 28
NOSE = 0


def make_landmarks(overrides):
    landmarks = [Landmark(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(NUM_LANDMARKS)]
    for idx, (x, y) in overrides.items():
        landmarks[idx] = Landmark(x=x, y=y, z=0.0, visibility=1.0)
    return landmarks


def test_compute_features_returns_expected_values():
    frame0 = FrameLandmarks(
        frame_index=0,
        timestamp_ms=0,
        landmarks=make_landmarks({
            LEFT_SHOULDER: (0.4, 0.5), RIGHT_SHOULDER: (0.6, 0.5), NOSE: (0.5, 0.5),
        }),
    )
    frame1 = FrameLandmarks(
        frame_index=1,
        timestamp_ms=100,
        landmarks=make_landmarks({
            RIGHT_SHOULDER: (0.6, 0.5), RIGHT_ELBOW: (0.6, 0.3), RIGHT_WRIST: (0.6, 0.1),
            LEFT_HIP: (0.45, 0.9), RIGHT_HIP: (0.55, 0.9),
            RIGHT_KNEE: (0.55, 1.0), RIGHT_ANKLE: (0.55, 1.2), NOSE: (0.5, 0.5),
        }),
    )
    frame2 = FrameLandmarks(
        frame_index=2, timestamp_ms=200, landmarks=make_landmarks({NOSE: (0.5, 0.5)})
    )
    phases = SwingPhases(ready_frame=0, backswing_frame=0, contact_frame=1, follow_through_frame=2)

    features = compute_features([frame0, frame1, frame2], phases)

    assert features.shoulder_rotation_deg == pytest.approx(0.0, abs=1e-6)
    assert features.elbow_angle_deg == pytest.approx(180.0, abs=1e-6)
    assert features.knee_bend_deg == pytest.approx(180.0, abs=1e-6)
    assert features.contact_height == pytest.approx(0.8, abs=1e-6)
    assert features.contact_depth == pytest.approx(0.1, abs=1e-6)
    assert features.head_stability == pytest.approx(0.0, abs=1e-6)

    vector = features.to_vector()
    assert len(vector) == len(FEATURE_NAMES) == 6


def test_swing_features_vector_round_trip():
    original = SwingFeatures(
        shoulder_rotation_deg=10.0, elbow_angle_deg=160.0, contact_height=0.3,
        contact_depth=0.15, knee_bend_deg=150.0, head_stability=0.01,
    )

    rebuilt = SwingFeatures.from_vector(original.to_vector())

    assert rebuilt == original
