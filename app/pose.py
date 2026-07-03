"""Wraps MediaPipe Pose Landmarker (Tasks API, VIDEO mode) to extract
per-frame body keypoints from a video file."""
from dataclasses import dataclass
from typing import List, Optional

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

DEFAULT_MODEL_PATH = "models/pose_landmarker_full.task"


@dataclass
class Landmark:
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class FrameLandmarks:
    frame_index: int
    timestamp_ms: int
    landmarks: Optional[List[Landmark]]


def _monotonic_timestamp_ms(candidate_ms: int, last_ms: int) -> int:
    """MediaPipe's Tasks API requires strictly increasing timestamps between
    detect_for_video calls. cap.get(CAP_PROP_POS_MSEC) can return 0 or a
    repeated value on some VFR/re-encoded files, so fall back to last_ms + 1
    whenever the candidate wouldn't advance the clock."""
    if candidate_ms > last_ms:
        return candidate_ms
    return last_ms + 1


def extract_landmarks(video_path: str, model_path: str = DEFAULT_MODEL_PATH) -> List[FrameLandmarks]:
    # num_poses=1 is how the spec's "pick the most prominent figure" error-handling
    # requirement is implemented: MediaPipe returns its single highest-confidence
    # detection per frame rather than every person in frame.
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    results: List[FrameLandmarks] = []

    try:
        with PoseLandmarker.create_from_options(options) as landmarker:
            frame_index = 0
            last_timestamp_ms = -1
            while True:
                success, frame = cap.read()
                if not success:
                    break

                candidate_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
                timestamp_ms = _monotonic_timestamp_ms(candidate_ms, last_timestamp_ms)
                last_timestamp_ms = timestamp_ms
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                detection = landmarker.detect_for_video(mp_image, timestamp_ms)

                if detection.pose_landmarks:
                    landmarks = [
                        Landmark(x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility)
                        for lm in detection.pose_landmarks[0]
                    ]
                else:
                    landmarks = None

                results.append(
                    FrameLandmarks(frame_index=frame_index, timestamp_ms=timestamp_ms, landmarks=landmarks)
                )
                frame_index += 1
    finally:
        cap.release()

    return results
