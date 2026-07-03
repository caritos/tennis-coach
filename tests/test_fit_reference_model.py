from pathlib import Path

import pytest

import scripts.fit_reference_model as fit_script


def test_build_feature_vectors_collects_and_skips(tmp_path, monkeypatch):
    for name in ["good1.mp4", "bad2.mp4", "ignored.txt"]:
        (tmp_path / name).write_bytes(b"fake video bytes")

    def fake_extract_landmarks(path, model_path=None):
        return [{"path": path}]

    def fake_detect_phases(frames):
        if "bad2" in frames[0]["path"]:
            raise ValueError("could not find a clean swing")
        return "phases"

    def fake_compute_features(frames, phases):
        class Features:
            def to_vector(self):
                return [1.0, 2.0, 3.0]
        return Features()

    monkeypatch.setattr(fit_script, "extract_landmarks", fake_extract_landmarks)
    monkeypatch.setattr(fit_script, "detect_phases", fake_detect_phases)
    monkeypatch.setattr(fit_script, "compute_features", fake_compute_features)

    vectors = fit_script.build_feature_vectors(tmp_path)

    assert vectors == [[1.0, 2.0, 3.0]]  # only good1.mp4 succeeds; bad2.mp4 skipped, .txt ignored


def test_build_feature_vectors_raises_when_no_videos_found(tmp_path):
    with pytest.raises(ValueError):
        fit_script.build_feature_vectors(tmp_path)


def test_holdout_check_runs_without_error(capsys):
    vectors = [[10.0 + 0.01 * i, 160.0, 0.3, 0.15, 150.0, 0.01] for i in range(20)]

    fit_script._holdout_check(vectors)

    captured = capsys.readouterr()
    assert "Hold-out sanity check" in captured.out


def test_holdout_check_skips_when_too_few_clips(capsys):
    fit_script._holdout_check([[1.0] * 6 for _ in range(5)])

    captured = capsys.readouterr()
    assert "Skipping hold-out" in captured.out
