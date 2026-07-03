"""Offline script: build the reference 'good forehand' scoring model from a
directory of reference video clips.

Usage:
    python scripts/fit_reference_model.py reference_clips/ models/reference_model.pkl
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.pose import extract_landmarks
from app.phases import detect_phases
from app.features import compute_features, SwingFeatures
from app.scoring import fit_reference_model, score_swing

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}
MIN_CLIPS_FOR_HOLDOUT_CHECK = 15


def build_feature_vectors(reference_dir: Path) -> list:
    clip_paths = sorted(p for p in Path(reference_dir).iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)

    if not clip_paths:
        raise ValueError(f"No video files found in {reference_dir}")

    vectors = []
    for clip_path in clip_paths:
        try:
            frames = extract_landmarks(str(clip_path))
            phases = detect_phases(frames)
            features = compute_features(frames, phases)
            vectors.append(features.to_vector())
            print(f"OK   {clip_path.name}")
        except ValueError as exc:
            # ValueError is the documented "expected" failure across the pipeline
            # (no clean swing found, too few detected frames, coincident landmark
            # points, etc.) — anything else is a real bug and should crash loudly
            # rather than being silently counted as a skipped clip.
            print(f"SKIP {clip_path.name}: {exc}")

    return vectors


def _holdout_check(vectors: list) -> None:
    """Sanity check: fit on 80% of reference clips, confirm the held-out 20%
    still mostly score as normal. Guards against the small reference-data
    size causing an overfit model that flags everything (including good
    swings) as an outlier."""
    if len(vectors) < MIN_CLIPS_FOR_HOLDOUT_CHECK:
        print(
            f"Skipping hold-out sanity check (need at least {MIN_CLIPS_FOR_HOLDOUT_CHECK} "
            f"clips for a meaningful split, got {len(vectors)})"
        )
        return

    split_index = int(len(vectors) * 0.8)
    train_vectors, holdout_vectors = vectors[:split_index], vectors[split_index:]

    holdout_model = fit_reference_model(train_vectors)
    outlier_count = 0
    for vector in holdout_vectors:
        features = SwingFeatures.from_vector(vector)
        result = score_swing(holdout_model, features)
        if result.findings:
            outlier_count += 1

    print(
        f"Hold-out sanity check: {outlier_count}/{len(holdout_vectors)} held-out reference "
        f"clips were flagged as outliers by a model trained on the rest (expect this to be low)."
    )


def main() -> None:
    if len(sys.argv) != 3:
        print("Usage: python scripts/fit_reference_model.py <reference_clips_dir> <output_model_path>")
        sys.exit(1)

    reference_dir = Path(sys.argv[1])
    output_path = sys.argv[2]

    vectors = build_feature_vectors(reference_dir)
    _holdout_check(vectors)

    print(f"Fitting final model on all {len(vectors)} reference clips...")
    model = fit_reference_model(vectors)
    model.save(output_path)
    print(f"Saved model to {output_path}")


if __name__ == "__main__":
    main()
