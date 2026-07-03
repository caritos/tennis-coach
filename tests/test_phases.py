import pytest

from app.pose import Landmark, FrameLandmarks
from app.phases import detect_phases, SwingPhases

NUM_LANDMARKS = 33


def make_landmarks(overrides):
    landmarks = [Landmark(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(NUM_LANDMARKS)]
    for idx, (x, y) in overrides.items():
        landmarks[idx] = Landmark(x=x, y=y, z=0.0, visibility=1.0)
    return landmarks


RIGHT_SHOULDER = 12
RIGHT_WRIST = 16


def make_swing_frames():
    # shoulder stays put; wrist goes ready -> backswing (far back) -> big forward
    # jump at contact -> settles into follow-through.
    wrist_positions = [0.5, 0.1, 0.3, 0.8, 0.85, 0.86]
    frames = []
    for i, wrist_x in enumerate(wrist_positions):
        landmarks = make_landmarks({RIGHT_SHOULDER: (0.4, 0.5), RIGHT_WRIST: (wrist_x, 0.5)})
        frames.append(FrameLandmarks(frame_index=i, timestamp_ms=i * 100, landmarks=landmarks))
    return frames


def test_detect_phases_finds_ready_backswing_contact_follow_through():
    frames = make_swing_frames()

    phases = detect_phases(frames)

    assert phases == SwingPhases(
        ready_frame=0, backswing_frame=1, contact_frame=3, follow_through_frame=5
    )


def test_detect_phases_skips_frames_with_no_detection():
    frames = make_swing_frames()
    frames[2] = FrameLandmarks(frame_index=2, timestamp_ms=200, landmarks=None)

    phases = detect_phases(frames)

    assert phases.ready_frame == 0
    assert phases.follow_through_frame == 5


def test_detect_phases_raises_with_too_few_detected_frames():
    frames = [
        FrameLandmarks(frame_index=0, timestamp_ms=0, landmarks=make_landmarks({})),
        FrameLandmarks(frame_index=1, timestamp_ms=100, landmarks=None),
        FrameLandmarks(frame_index=2, timestamp_ms=200, landmarks=None),
    ]

    with pytest.raises(ValueError):
        detect_phases(frames)
