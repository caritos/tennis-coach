from pathlib import Path

import pytest

from app.pose import extract_landmarks, FrameLandmarks

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_clip.mp4"


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
