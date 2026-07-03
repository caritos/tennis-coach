"""Pure functions that locate swing phases within a sequence of pose
landmarks. Assumes a single, complete right-handed forehand swing per clip
(v1 constraint)."""
from dataclasses import dataclass
from typing import List
import math

from .pose import FrameLandmarks
from .constants import RIGHT_WRIST, RIGHT_SHOULDER


@dataclass
class SwingPhases:
    ready_frame: int
    backswing_frame: int
    contact_frame: int
    follow_through_frame: int


def _valid_frames(frames: List[FrameLandmarks]) -> List[FrameLandmarks]:
    return [f for f in frames if f.landmarks is not None]


def _wrist_speed(a: FrameLandmarks, b: FrameLandmarks) -> float:
    wa = a.landmarks[RIGHT_WRIST]
    wb = b.landmarks[RIGHT_WRIST]
    dx = wb.x - wa.x
    dy = wb.y - wa.y
    dt_s = (b.timestamp_ms - a.timestamp_ms) / 1000.0
    if dt_s <= 0:
        return 0.0
    return math.hypot(dx, dy) / dt_s


def detect_phases(frames: List[FrameLandmarks]) -> SwingPhases:
    valid = _valid_frames(frames)
    if len(valid) < 4:
        raise ValueError("Not enough frames with a detected person to identify swing phases")

    speeds = [
        (valid[i + 1].frame_index, _wrist_speed(valid[i], valid[i + 1]))
        for i in range(len(valid) - 1)
    ]
    contact_frame = max(speeds, key=lambda item: item[1])[0]

    ready_frame = valid[0].frame_index

    pre_contact = [f for f in valid if f.frame_index < contact_frame]
    if not pre_contact:
        raise ValueError("No frames found before the detected contact point")
    backswing_frame = max(
        pre_contact,
        key=lambda f: abs(f.landmarks[RIGHT_WRIST].x - f.landmarks[RIGHT_SHOULDER].x),
    ).frame_index

    follow_through_frame = valid[-1].frame_index

    return SwingPhases(
        ready_frame=ready_frame,
        backswing_frame=backswing_frame,
        contact_frame=contact_frame,
        follow_through_frame=follow_through_frame,
    )
