from pathlib import Path

import cv2
import numpy as np

from app.pose import Landmark, FrameLandmarks
from app.phases import SwingPhases
from app.annotate import draw_skeleton, annotate_video, save_phase_frames

NUM_LANDMARKS = 33


def make_landmarks():
    # Distinct, in-bounds positions so skeleton lines/points land inside the frame.
    return [Landmark(x=0.3 + 0.01 * i, y=0.3 + 0.01 * i, z=0.0, visibility=1.0) for i in range(NUM_LANDMARKS)]


def make_test_video(path: str, num_frames=3, size=(64, 48)):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, size)
    for _ in range(num_frames):
        writer.write(np.zeros((size[1], size[0], 3), dtype=np.uint8))
    writer.release()


def test_draw_skeleton_modifies_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_skeleton(frame, make_landmarks())
    assert frame.sum() > 0


def test_annotate_video_produces_output_with_same_frame_count(tmp_path):
    video_path = str(tmp_path / "input.mp4")
    output_path = str(tmp_path / "output.mp4")
    make_test_video(video_path, num_frames=3)

    frames = [
        FrameLandmarks(frame_index=i, timestamp_ms=i * 100, landmarks=make_landmarks())
        for i in range(3)
    ]

    annotate_video(video_path, frames, output_path)

    assert Path(output_path).exists()
    cap = cv2.VideoCapture(output_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    assert frame_count == 3


def test_save_phase_frames_writes_all_four_phase_images(tmp_path):
    video_path = str(tmp_path / "input.mp4")
    output_dir = str(tmp_path / "phases")
    make_test_video(video_path, num_frames=3)

    frames = [
        FrameLandmarks(frame_index=i, timestamp_ms=i * 100, landmarks=make_landmarks())
        for i in range(3)
    ]
    phases = SwingPhases(ready_frame=0, backswing_frame=0, contact_frame=1, follow_through_frame=2)

    paths = save_phase_frames(video_path, frames, phases, output_dir)

    assert set(paths.keys()) == {"ready", "backswing", "contact", "follow_through"}
    for path in paths.values():
        assert Path(path).exists()
