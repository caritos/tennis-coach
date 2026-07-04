import re
from pathlib import Path

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app
from app.pipeline import AnalysisResult, LowPoseConfidenceError

client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_upload_form_renders():
    response = client.get("/")
    assert response.status_code == 200
    assert "Upload a Forehand Clip" in response.text


def test_analyze_renders_results_on_success(monkeypatch, tmp_path):
    fake_result = AnalysisResult(
        feedback_text="Great extension through contact!",
        annotated_video_path=str(tmp_path / "annotated.mp4"),
        phase_frame_paths={
            "ready": str(tmp_path / "ready.jpg"),
            "backswing": str(tmp_path / "backswing.jpg"),
            "contact": str(tmp_path / "contact.jpg"),
            "follow_through": str(tmp_path / "follow_through.jpg"),
        },
    )
    for path in [fake_result.annotated_video_path, *fake_result.phase_frame_paths.values()]:
        with open(path, "wb") as f:
            f.write(b"fake bytes")

    monkeypatch.setattr(main_module, "analyze_forehand", lambda *a, **k: fake_result)

    response = client.post("/analyze", files={"video": ("clip.mp4", b"fake video bytes", "video/mp4")})

    assert response.status_code == 200
    assert "Great extension through contact!" in response.text


def test_analyze_shows_error_on_low_confidence(monkeypatch):
    def fake_analyze(*args, **kwargs):
        raise LowPoseConfidenceError("Couldn't get a clear read on this clip")

    monkeypatch.setattr(main_module, "analyze_forehand", fake_analyze)

    response = client.post("/analyze", files={"video": ("clip.mp4", b"fake video bytes", "video/mp4")})

    assert response.status_code == 200
    assert "Couldn&#39;t get a clear read on this clip" in response.text or "Couldn't get a clear read on this clip" in response.text


def test_analyze_shows_error_when_ollama_unavailable(monkeypatch):
    from app.feedback import OllamaUnavailableError

    def fake_analyze(*args, **kwargs):
        raise OllamaUnavailableError("Could not reach Ollama at localhost:11434")

    monkeypatch.setattr(main_module, "analyze_forehand", fake_analyze)

    response = client.post("/analyze", files={"video": ("clip.mp4", b"fake video bytes", "video/mp4")})

    assert response.status_code == 200
    assert "Could not reach Ollama" in response.text


def test_analyze_sanitizes_malicious_filename(monkeypatch, tmp_path):
    captured_paths = {}

    def fake_analyze(video_path, model_path, output_dir):
        captured_paths["video_path"] = video_path
        raise LowPoseConfidenceError("stop before real processing")

    monkeypatch.setattr(main_module, "analyze_forehand", fake_analyze)

    response = client.post(
        "/analyze",
        files={"video": ("../../../../etc/passwd", b"fake video bytes", "video/mp4")},
    )

    assert response.status_code == 200
    saved_path = Path(captured_paths["video_path"])
    assert saved_path.name == "upload.mp4"
    assert ".." not in saved_path.parts


def test_analyze_shows_error_when_no_clean_swing_detected(monkeypatch):
    def fake_analyze(*args, **kwargs):
        raise ValueError("Not enough frames with a detected person to identify swing phases")

    monkeypatch.setattr(main_module, "analyze_forehand", fake_analyze)

    response = client.post("/analyze", files={"video": ("clip.mp4", b"fake video bytes", "video/mp4")})

    assert response.status_code == 200
    assert "Not enough frames with a detected person" in response.text


def test_analyze_shows_error_when_reference_model_missing(monkeypatch):
    def fake_analyze(*args, **kwargs):
        raise FileNotFoundError("models/reference_model.pkl")

    monkeypatch.setattr(main_module, "analyze_forehand", fake_analyze)

    response = client.post("/analyze", files={"video": ("clip.mp4", b"fake video bytes", "video/mp4")})

    assert response.status_code == 200
    assert "No reference model found yet" in response.text


def test_analyze_isolates_results_across_requests(monkeypatch, tmp_path):
    def make_result(label):
        video_path = str(tmp_path / f"{label}_annotated.mp4")
        phase_path = str(tmp_path / f"{label}_ready.jpg")
        with open(video_path, "wb") as f:
            f.write(label.encode())
        with open(phase_path, "wb") as f:
            f.write(label.encode())
        return AnalysisResult(
            feedback_text=f"{label} feedback",
            annotated_video_path=video_path,
            phase_frame_paths={
                "ready": phase_path,
                "backswing": phase_path,
                "contact": phase_path,
                "follow_through": phase_path,
            },
        )

    results = iter([make_result("first"), make_result("second")])
    monkeypatch.setattr(main_module, "analyze_forehand", lambda *a, **k: next(results))

    response_first = client.post("/analyze", files={"video": ("clip.mp4", b"x", "video/mp4")})
    response_second = client.post("/analyze", files={"video": ("clip.mp4", b"x", "video/mp4")})

    assert "first feedback" in response_first.text
    assert "second feedback" in response_second.text

    url_first = re.search(r'src="(/static/results/[^"]*annotated\.mp4)"', response_first.text).group(1)
    url_second = re.search(r'src="(/static/results/[^"]*annotated\.mp4)"', response_second.text).group(1)

    # Each request must get its own storage location — otherwise the second
    # request's copy overwrites the first's file before/while the first
    # request's response is still being served.
    assert url_first != url_second
    assert client.get(url_first).content == b"first"
    assert client.get(url_second).content == b"second"
