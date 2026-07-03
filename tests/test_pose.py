from pathlib import Path

import pytest

from app.pose import extract_landmarks, FrameLandmarks, _monotonic_timestamp_ms

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_clip.mp4"


def test_monotonic_timestamp_ms_advances_normally():
    assert _monotonic_timestamp_ms(candidate_ms=100, last_ms=50) == 100


def test_monotonic_timestamp_ms_falls_back_when_not_increasing():
    assert _monotonic_timestamp_ms(candidate_ms=0, last_ms=50) == 51
    assert _monotonic_timestamp_ms(candidate_ms=50, last_ms=50) == 51


@pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason="Add tests/fixtures/sample_clip.mp4 (a short real forehand clip) to run this test",
)
def test_extract_landmarks_detects_person_in_most_frames():
    frames = extract_landmarks(str(FIXTURE_PATH))

    assert len(frames) > 0
    assert all(isinstance(f, FrameLandmarks) for f in frames)

    detected = [f for f in frames if f.landmarks is not None]
    assert len(detected) / len(frames) >= 0.5

    for f in detected:
        assert len(f.landmarks) == 33
