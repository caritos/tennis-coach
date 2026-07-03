import pytest
import requests

from app.scoring import ScoreResult, Finding
from app.feedback import generate_feedback, OllamaUnavailableError, _build_prompt


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json


def make_result(findings=None):
    return ScoreResult(overall_mahalanobis_distance=1.2, findings=findings or [])


def test_build_prompt_includes_findings():
    result = make_result([
        Finding(feature="elbow_angle_deg", value=120.0, reference_mean=160.0, z_score=-2.0, severity="moderate")
    ])

    prompt = _build_prompt(result)

    assert "elbow_angle_deg" in prompt
    assert "120.0" in prompt


def test_build_prompt_handles_no_findings():
    prompt = _build_prompt(make_result([]))
    assert "No technique issues were flagged" in prompt


def test_generate_feedback_returns_response_text(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse({"response": "Great swing overall!"})
    )

    text = generate_feedback(make_result())

    assert text == "Great swing overall!"


def test_generate_feedback_raises_when_ollama_unreachable(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.exceptions.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(OllamaUnavailableError):
        generate_feedback(make_result())


def test_generate_feedback_raises_when_ollama_times_out(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr(requests, "post", fake_post)

    with pytest.raises(OllamaUnavailableError):
        generate_feedback(make_result())


def test_generate_feedback_raises_when_model_not_found(monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: FakeResponse({"response": "unused"}, status_code=404)
    )

    with pytest.raises(OllamaUnavailableError):
        generate_feedback(make_result())
