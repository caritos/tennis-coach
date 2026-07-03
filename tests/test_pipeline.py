import pytest

from app.pose import FrameLandmarks, Landmark
from app.scoring import ScoreResult
import app.pipeline as pipeline
from app.pipeline import analyze_forehand, LowPoseConfidenceError, AnalysisResult


def make_frames(num_detected, num_missing):
    frames = []
    for i in range(num_detected):
        frames.append(FrameLandmarks(frame_index=i, timestamp_ms=i * 100, landmarks=[Landmark(0.5, 0.5, 0.0, 1.0)] * 33))
    for i in range(num_missing):
        frames.append(FrameLandmarks(frame_index=num_detected + i, timestamp_ms=(num_detected + i) * 100, landmarks=None))
    return frames


def install_fakes(monkeypatch, frames, calls):
    monkeypatch.setattr(pipeline, "extract_landmarks", lambda path, model_path=None: frames)
    monkeypatch.setattr(pipeline, "detect_phases", lambda f: calls.setdefault("phases_called", True) or "PHASES")
    monkeypatch.setattr(pipeline, "compute_features", lambda f, p: calls.setdefault("features_called", True) or "FEATURES")

    class FakeReferenceModel:
        @staticmethod
        def load(path):
            return "MODEL"

    monkeypatch.setattr(pipeline, "ReferenceModel", FakeReferenceModel)
    monkeypatch.setattr(
        pipeline, "score_swing",
        lambda model, features: ScoreResult(overall_mahalanobis_distance=1.0, findings=[]),
    )
    monkeypatch.setattr(pipeline, "generate_feedback", lambda result: "Nice swing!")
    monkeypatch.setattr(pipeline, "annotate_video", lambda video_path, frames, output_path: calls.setdefault("annotate_called", True))
    monkeypatch.setattr(
        pipeline, "save_phase_frames",
        lambda video_path, frames, phases, output_dir: {"ready": "r.jpg", "backswing": "b.jpg", "contact": "c.jpg", "follow_through": "f.jpg"},
    )


def test_analyze_forehand_happy_path(tmp_path, monkeypatch):
    calls = {}
    install_fakes(monkeypatch, make_frames(num_detected=10, num_missing=0), calls)

    result = analyze_forehand("video.mp4", "model.pkl", str(tmp_path))

    assert isinstance(result, AnalysisResult)
    assert result.feedback_text == "Nice swing!"
    assert result.annotated_video_path == str(tmp_path / "annotated.mp4")
    assert result.phase_frame_paths["contact"] == "c.jpg"
    assert calls.get("phases_called")
    assert calls.get("features_called")
    assert calls.get("annotate_called")


def test_analyze_forehand_rejects_low_confidence_clips(tmp_path, monkeypatch):
    calls = {}
    install_fakes(monkeypatch, make_frames(num_detected=2, num_missing=8), calls)

    with pytest.raises(LowPoseConfidenceError):
        analyze_forehand("video.mp4", "model.pkl", str(tmp_path))

    assert "phases_called" not in calls
