"""Orchestrates the full analysis flow: raw video in, annotated output +
coaching feedback out."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .pose import extract_landmarks
from .phases import detect_phases
from .features import compute_features
from .scoring import ReferenceModel, score_swing
from .feedback import generate_feedback
from .annotate import annotate_video, save_phase_frames

MIN_DETECTION_RATE = 0.5


class LowPoseConfidenceError(RuntimeError):
    pass


@dataclass
class AnalysisResult:
    feedback_text: str
    annotated_video_path: str
    phase_frame_paths: Dict[str, str]


def analyze_forehand(
    video_path: str,
    model_path: str,
    output_dir: str,
    pose_model_path: str = "models/pose_landmarker_full.task",
) -> AnalysisResult:
    frames = extract_landmarks(video_path, model_path=pose_model_path)

    detected_count = sum(1 for f in frames if f.landmarks is not None)
    if not frames or detected_count / len(frames) < MIN_DETECTION_RATE:
        raise LowPoseConfidenceError(
            "Couldn't get a clear read on this clip — make sure the student is "
            "clearly visible and well-lit throughout the swing."
        )

    phases = detect_phases(frames)
    features = compute_features(frames, phases)

    reference_model = ReferenceModel.load(model_path)
    score_result = score_swing(reference_model, features)

    feedback_text = generate_feedback(score_result)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    annotated_video_path = str(Path(output_dir) / "annotated.mp4")
    annotate_video(video_path, frames, annotated_video_path)
    phase_frame_paths = save_phase_frames(video_path, frames, phases, output_dir)

    return AnalysisResult(
        feedback_text=feedback_text,
        annotated_video_path=annotated_video_path,
        phase_frame_paths=phase_frame_paths,
    )
