"""Draws a pose skeleton overlay onto video frames and extracts annotated
key phase frames for the coaching report."""
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from .pose import FrameLandmarks
from .phases import SwingPhases
from .constants import (
    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST,
    LEFT_HIP, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, LEFT_KNEE, LEFT_ANKLE,
)

SKELETON_CONNECTIONS: List[Tuple[int, int]] = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
    (NOSE, LEFT_SHOULDER),
    (NOSE, RIGHT_SHOULDER),
]

POINT_COLOR = (0, 255, 0)
LINE_COLOR = (255, 255, 0)


def draw_skeleton(frame, landmarks) -> None:
    height, width = frame.shape[:2]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

    for a, b in SKELETON_CONNECTIONS:
        cv2.line(frame, points[a], points[b], LINE_COLOR, 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, POINT_COLOR, -1)


def annotate_video(video_path: str, frames: List[FrameLandmarks], output_path: str) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        try:
            frames_by_index = {f.frame_index: f for f in frames}

            frame_index = 0
            while True:
                success, frame = cap.read()
                if not success:
                    break
                record = frames_by_index.get(frame_index)
                if record is not None and record.landmarks is not None:
                    draw_skeleton(frame, record.landmarks)
                writer.write(frame)
                frame_index += 1
        finally:
            writer.release()
    finally:
        cap.release()


def save_phase_frames(
    video_path: str, frames: List[FrameLandmarks], phases: SwingPhases, output_dir: str
) -> Dict[str, str]:
    phase_frame_indices = {
        "ready": phases.ready_frame,
        "backswing": phases.backswing_frame,
        "contact": phases.contact_frame,
        "follow_through": phases.follow_through_frame,
    }
    frames_by_index = {f.frame_index: f for f in frames}

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    output_paths = {}
    try:
        for phase_name, target_index in phase_frame_indices.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_index)
            success, frame = cap.read()
            if not success:
                continue
            record = frames_by_index.get(target_index)
            if record is not None and record.landmarks is not None:
                draw_skeleton(frame, record.landmarks)
            out_path = str(Path(output_dir) / f"{phase_name}.jpg")
            cv2.imwrite(out_path, frame)
            output_paths[phase_name] = out_path
    finally:
        cap.release()

    return output_paths
