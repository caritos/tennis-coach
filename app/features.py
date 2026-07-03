"""Pure functions computing biomechanical feature vectors from pose
landmarks at swing phases."""
from dataclasses import dataclass
import math
import statistics
from typing import List

from .pose import FrameLandmarks
from .phases import SwingPhases
from .constants import (
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE,
)

FEATURE_NAMES = [
    "shoulder_rotation_deg",
    "elbow_angle_deg",
    "contact_height",
    "contact_depth",
    "knee_bend_deg",
    "head_stability",
]


@dataclass
class SwingFeatures:
    shoulder_rotation_deg: float
    elbow_angle_deg: float
    contact_height: float
    contact_depth: float
    knee_bend_deg: float
    head_stability: float

    def to_vector(self) -> List[float]:
        return [getattr(self, name) for name in FEATURE_NAMES]

    @classmethod
    def from_vector(cls, vector: List[float]) -> "SwingFeatures":
        return cls(**dict(zip(FEATURE_NAMES, vector)))


def _angle_deg(a, b, c) -> float:
    """Angle at point b formed by points a-b-c, in degrees."""
    v1 = (a.x - b.x, a.y - b.y, a.z - b.z)
    v2 = (c.x - b.x, c.y - b.y, c.z - b.z)
    dot = sum(p * q for p, q in zip(v1, v2))
    mag1 = math.sqrt(sum(p * p for p in v1))
    mag2 = math.sqrt(sum(p * p for p in v2))
    if mag1 == 0 or mag2 == 0:
        raise ValueError("Cannot compute angle between coincident landmark points")
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def compute_features(frames: List[FrameLandmarks], phases: SwingPhases) -> SwingFeatures:
    by_index = {f.frame_index: f for f in frames if f.landmarks is not None}

    backswing = by_index[phases.backswing_frame].landmarks
    contact = by_index[phases.contact_frame].landmarks

    shoulder_rotation_deg = math.degrees(
        math.atan2(
            backswing[RIGHT_SHOULDER].y - backswing[LEFT_SHOULDER].y,
            backswing[RIGHT_SHOULDER].x - backswing[LEFT_SHOULDER].x,
        )
    )

    elbow_angle_deg = _angle_deg(contact[RIGHT_SHOULDER], contact[RIGHT_ELBOW], contact[RIGHT_WRIST])
    knee_bend_deg = _angle_deg(contact[RIGHT_HIP], contact[RIGHT_KNEE], contact[RIGHT_ANKLE])

    hip_mid_x = (contact[LEFT_HIP].x + contact[RIGHT_HIP].x) / 2
    contact_height = contact[RIGHT_HIP].y - contact[RIGHT_WRIST].y
    contact_depth = contact[RIGHT_WRIST].x - hip_mid_x

    swing_frames = [
        f for f in frames
        if f.landmarks is not None and phases.ready_frame <= f.frame_index <= phases.follow_through_frame
    ]
    nose_x_positions = [f.landmarks[NOSE].x for f in swing_frames]
    head_stability = statistics.pstdev(nose_x_positions) if len(nose_x_positions) > 1 else 0.0

    return SwingFeatures(
        shoulder_rotation_deg=shoulder_rotation_deg,
        elbow_angle_deg=elbow_angle_deg,
        contact_height=contact_height,
        contact_depth=contact_depth,
        knee_bend_deg=knee_bend_deg,
        head_stability=head_stability,
    )
