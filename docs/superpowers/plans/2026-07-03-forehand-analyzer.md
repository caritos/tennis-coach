# Forehand Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web app where a coach uploads a short forehand clip and gets back an annotated video (pose skeleton overlay) plus written coaching feedback, using only local compute — no paid AI APIs.

**Architecture:** FastAPI backend runs MediaPipe pose extraction over the uploaded clip, detects swing phases and computes biomechanical features, scores those features against a statistical model *fitted from a reference library of good-forehand clips* (learned, not hand-coded thresholds), and passes the resulting findings to a locally-running Ollama model to write up natural-language feedback. OpenCV draws the pose skeleton onto the video and key frames for the results page.

**Tech Stack:** Python 3.11+, FastAPI + Jinja2 templates, MediaPipe (Pose Landmarker, Tasks API), OpenCV, scikit-learn + numpy, Ollama (local HTTP API) via `requests`, pytest.

## Global Constraints

- All analysis runs on local compute — no paid AI APIs (spec: Goal).
- Single user, no accounts, no database (spec: Goal).
- Forehand only in v1 (spec: Goal, Non-goals).
- Each uploaded clip contains exactly one clean forehand swing (spec: Key assumptions).
- "Good technique" must come from a statistical model *fitted* on reference-clip feature data — never hand-coded thresholds (spec: Scoring model).
- Reference clips must come from a legitimate, non-copyright-infringing source (self-filmed or permissively-licensed) — never scraped broadcast footage (spec: Reference data).
- **Plan-level addition, not in spec:** v1 assumes a right-handed player (racket arm = right arm). The spec doesn't address handedness; this is the simplest correct default for a single-arm feature set. Revisit if the coach needs left-handed students supported.
- Python ≥3.9 required by `mediapipe`; this plan targets 3.11.

## Directory Structure

```
tennis-coach/
  app/
    __init__.py
    constants.py       # BlazePose landmark index constants
    pose.py            # MediaPipe pose extraction from a video file
    phases.py          # swing phase detection (ready/backswing/contact/follow-through)
    features.py        # biomechanical feature computation
    scoring.py          # reference model fit/save/load + scoring
    feedback.py         # Ollama-based coaching text generation
    annotate.py         # OpenCV skeleton drawing + key-frame extraction
    pipeline.py         # orchestrates the full analysis flow
    main.py             # FastAPI app + routes
  templates/
    upload.html
    results.html
  static/
    results/           # generated per-request output, gitignored
  scripts/
    __init__.py
    download_pose_model.sh
    fit_reference_model.py
  reference_clips/      # user-supplied reference videos, gitignored (dir tracked via .gitkeep)
  models/                # generated model artifacts, gitignored
  tests/
    __init__.py
    fixtures/
      README.md
    test_phases.py
    test_features.py
    test_scoring.py
    test_feedback.py
    test_annotate.py
    test_pipeline.py
    test_main.py
    test_fit_reference_model.py
    test_pose.py
  requirements.txt
  .gitignore
```

---

### Task 1: Project scaffolding + FastAPI health check

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `scripts/download_pose_model.sh`
- Create: `tests/__init__.py`
- Create: `tests/test_main.py`
- Create: `tests/fixtures/README.md`

**Interfaces:**
- Produces: FastAPI app instance `app.main.app`, importable by later tasks and by `uvicorn app.main:app`.

- [ ] **Step 1: Create the project scaffolding files**

`requirements.txt`:
```
fastapi>=0.115
uvicorn[standard]>=0.32
python-multipart>=0.0.9
jinja2>=3.1
opencv-python>=4.10
mediapipe>=0.10
scikit-learn>=1.5
numpy>=1.26
requests>=2.32
pytest>=8.3
```

`.gitignore`:
```
venv/
__pycache__/
*.pyc
.pytest_cache/
models/*.task
models/*.pkl
static/results/*
!static/results/.gitkeep
reference_clips/*
!reference_clips/.gitkeep
```

`app/__init__.py`:
```python
```

`tests/__init__.py`:
```python
```

`tests/fixtures/README.md`:
```markdown
# Test fixtures

Drop a short real forehand clip named `sample_clip.mp4` in this directory to
enable the pose-extraction smoke test in `tests/test_pose.py`. Without it,
that test is skipped (MediaPipe pose extraction can't be meaningfully unit
tested without a real video of a real person).
```

`scripts/download_pose_model.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
mkdir -p models
curl -L -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
echo "Downloaded models/pose_landmarker_full.task"
```

- [ ] **Step 2: Set up the virtualenv and install dependencies**

Run:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
chmod +x scripts/download_pose_model.sh
mkdir -p static/results reference_clips models
touch static/results/.gitkeep reference_clips/.gitkeep
```
Expected: dependencies install with no errors.

- [ ] **Step 3: Write the failing test for the health check**

`tests/test_main.py`:
```python
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `pytest tests/test_main.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.main'` (file doesn't exist yet).

- [ ] **Step 5: Implement the minimal FastAPI app**

`app/main.py`:
```python
"""FastAPI web app: upload a forehand clip, get back annotated video + coaching feedback."""
from fastapi import FastAPI

app = FastAPI()


@app.get("/health")
def health_check():
    return {"status": "ok"}
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `pytest tests/test_main.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .gitignore app/ scripts/download_pose_model.sh tests/ static/results/.gitkeep reference_clips/.gitkeep
git commit -m "Scaffold project with FastAPI health check"
```

---

### Task 2: Pose extraction module

**Files:**
- Create: `app/constants.py`
- Create: `app/pose.py`
- Create: `tests/test_pose.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `app.constants`: `NOSE=0, LEFT_SHOULDER=11, RIGHT_SHOULDER=12, LEFT_ELBOW=13, RIGHT_ELBOW=14, LEFT_WRIST=15, RIGHT_WRIST=16, LEFT_HIP=23, RIGHT_HIP=24, LEFT_KNEE=25, RIGHT_KNEE=26, LEFT_ANKLE=27, RIGHT_ANKLE=28`
  - `app.pose.Landmark` dataclass: `x: float, y: float, z: float, visibility: float`
  - `app.pose.FrameLandmarks` dataclass: `frame_index: int, timestamp_ms: int, landmarks: Optional[List[Landmark]]` (`None` when no person detected in that frame)
  - `app.pose.extract_landmarks(video_path: str, model_path: str = "models/pose_landmarker_full.task") -> List[FrameLandmarks]`

- [ ] **Step 1: Download the pose model**

Run: `./scripts/download_pose_model.sh`
Expected: `models/pose_landmarker_full.task` exists (tens of MB).

- [ ] **Step 2: Write the landmark index constants**

`app/constants.py`:
```python
"""BlazePose 33-point landmark indices used throughout the pipeline."""

NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
```

- [ ] **Step 3: Write the failing smoke test**

`tests/test_pose.py`:
```python
from pathlib import Path

import pytest

from app.pose import extract_landmarks, FrameLandmarks

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_clip.mp4"


@pytest.mark.skipif(
    not FIXTURE_PATH.exists(),
    reason="Add tests/fixtures/sample_clip.mp4 (a short real forehand clip) to run this test",
)
def test_extract_landmarks_detects_person_in_most_frames():
    frames = extract_landmarks(str(FIXTURE_PATH))

    assert len(frames) > 0
    assert all(isinstance(f, FrameLandmarks) for f in frames)

    detected = [f for f in frames if f.landmarks is not None]
    assert len(detected) / len(frames) >= 0.5

    for f in detected:
        assert len(f.landmarks) == 33
```

- [ ] **Step 4: Run the test to verify it fails (or skips)**

Run: `pytest tests/test_pose.py -v`
Expected: SKIPPED if no fixture clip present yet, or FAIL with `ModuleNotFoundError: No module named 'app.pose'` if you've already added a fixture clip.

- [ ] **Step 5: Implement pose extraction**

`app/pose.py`:
```python
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

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
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

    cap.release()
    return results
```

- [ ] **Step 6: Run the test**

Run: `pytest tests/test_pose.py -v`
Expected: SKIPPED (no fixture clip yet — that's fine for now) or PASS if you've supplied `tests/fixtures/sample_clip.mp4`.

- [ ] **Step 7: Commit**

```bash
git add app/constants.py app/pose.py tests/test_pose.py tests/fixtures/README.md
git commit -m "Add MediaPipe pose extraction module"
```

---

### Task 3: Phase detection module

**Files:**
- Create: `app/phases.py`
- Create: `tests/test_phases.py`

**Interfaces:**
- Consumes: `app.pose.Landmark`, `app.pose.FrameLandmarks` (Task 2).
- Produces:
  - `app.phases.SwingPhases` dataclass: `ready_frame: int, backswing_frame: int, contact_frame: int, follow_through_frame: int`
  - `app.phases.detect_phases(frames: List[FrameLandmarks]) -> SwingPhases` (raises `ValueError` if fewer than 4 frames have detected landmarks, or no frames precede the detected contact point)

- [ ] **Step 1: Write the failing tests**

`tests/test_phases.py`:
```python
import pytest

from app.pose import Landmark, FrameLandmarks
from app.phases import detect_phases, SwingPhases

NUM_LANDMARKS = 33


def make_landmarks(overrides):
    landmarks = [Landmark(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(NUM_LANDMARKS)]
    for idx, (x, y) in overrides.items():
        landmarks[idx] = Landmark(x=x, y=y, z=0.0, visibility=1.0)
    return landmarks


RIGHT_SHOULDER = 12
RIGHT_WRIST = 16


def make_swing_frames():
    # shoulder stays put; wrist goes ready -> backswing (far back) -> big forward
    # jump at contact -> settles into follow-through.
    wrist_positions = [0.5, 0.1, 0.3, 0.8, 0.85, 0.86]
    frames = []
    for i, wrist_x in enumerate(wrist_positions):
        landmarks = make_landmarks({RIGHT_SHOULDER: (0.4, 0.5), RIGHT_WRIST: (wrist_x, 0.5)})
        frames.append(FrameLandmarks(frame_index=i, timestamp_ms=i * 100, landmarks=landmarks))
    return frames


def test_detect_phases_finds_ready_backswing_contact_follow_through():
    frames = make_swing_frames()

    phases = detect_phases(frames)

    assert phases == SwingPhases(
        ready_frame=0, backswing_frame=1, contact_frame=3, follow_through_frame=5
    )


def test_detect_phases_skips_frames_with_no_detection():
    frames = make_swing_frames()
    frames[2] = FrameLandmarks(frame_index=2, timestamp_ms=200, landmarks=None)

    phases = detect_phases(frames)

    assert phases.ready_frame == 0
    assert phases.follow_through_frame == 5


def test_detect_phases_raises_with_too_few_detected_frames():
    frames = [
        FrameLandmarks(frame_index=0, timestamp_ms=0, landmarks=make_landmarks({})),
        FrameLandmarks(frame_index=1, timestamp_ms=100, landmarks=None),
        FrameLandmarks(frame_index=2, timestamp_ms=200, landmarks=None),
    ]

    with pytest.raises(ValueError):
        detect_phases(frames)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_phases.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.phases'`

- [ ] **Step 3: Implement phase detection**

`app/phases.py`:
```python
"""Pure functions that locate swing phases within a sequence of pose
landmarks. Assumes a single, complete right-handed forehand swing per clip
(v1 constraint)."""
from dataclasses import dataclass
from typing import List
import math

from .pose import FrameLandmarks
from .constants import RIGHT_WRIST, RIGHT_SHOULDER


@dataclass
class SwingPhases:
    ready_frame: int
    backswing_frame: int
    contact_frame: int
    follow_through_frame: int


def _valid_frames(frames: List[FrameLandmarks]) -> List[FrameLandmarks]:
    return [f for f in frames if f.landmarks is not None]


def _wrist_speed(a: FrameLandmarks, b: FrameLandmarks) -> float:
    wa = a.landmarks[RIGHT_WRIST]
    wb = b.landmarks[RIGHT_WRIST]
    dx = wb.x - wa.x
    dy = wb.y - wa.y
    dt_s = (b.timestamp_ms - a.timestamp_ms) / 1000.0
    if dt_s <= 0:
        return 0.0
    return math.hypot(dx, dy) / dt_s


def detect_phases(frames: List[FrameLandmarks]) -> SwingPhases:
    valid = _valid_frames(frames)
    if len(valid) < 4:
        raise ValueError("Not enough frames with a detected person to identify swing phases")

    speeds = [
        (valid[i + 1].frame_index, _wrist_speed(valid[i], valid[i + 1]))
        for i in range(len(valid) - 1)
    ]
    contact_frame = max(speeds, key=lambda item: item[1])[0]

    ready_frame = valid[0].frame_index

    pre_contact = [f for f in valid if f.frame_index < contact_frame]
    if not pre_contact:
        raise ValueError("No frames found before the detected contact point")
    backswing_frame = max(
        pre_contact,
        key=lambda f: abs(f.landmarks[RIGHT_WRIST].x - f.landmarks[RIGHT_SHOULDER].x),
    ).frame_index

    follow_through_frame = valid[-1].frame_index

    return SwingPhases(
        ready_frame=ready_frame,
        backswing_frame=backswing_frame,
        contact_frame=contact_frame,
        follow_through_frame=follow_through_frame,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_phases.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/phases.py tests/test_phases.py
git commit -m "Add swing phase detection from pose landmark sequences"
```

---

### Task 4: Feature computation module

**Files:**
- Create: `app/features.py`
- Create: `tests/test_features.py`

**Interfaces:**
- Consumes: `app.pose.FrameLandmarks` (Task 2), `app.phases.SwingPhases` (Task 3).
- Produces:
  - `app.features.FEATURE_NAMES: List[str]` = `["shoulder_rotation_deg", "elbow_angle_deg", "contact_height", "contact_depth", "knee_bend_deg", "head_stability"]`
  - `app.features.SwingFeatures` dataclass with those six fields plus `.to_vector() -> List[float]`
  - `app.features.compute_features(frames: List[FrameLandmarks], phases: SwingPhases) -> SwingFeatures`

- [ ] **Step 1: Write the failing test**

`tests/test_features.py`:
```python
import pytest

from app.pose import Landmark, FrameLandmarks
from app.phases import SwingPhases
from app.features import compute_features, FEATURE_NAMES

NUM_LANDMARKS = 33
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
RIGHT_ELBOW, RIGHT_WRIST = 14, 16
LEFT_HIP, RIGHT_HIP = 23, 24
RIGHT_KNEE, RIGHT_ANKLE = 26, 28
NOSE = 0


def make_landmarks(overrides):
    landmarks = [Landmark(x=0.5, y=0.5, z=0.0, visibility=1.0) for _ in range(NUM_LANDMARKS)]
    for idx, (x, y) in overrides.items():
        landmarks[idx] = Landmark(x=x, y=y, z=0.0, visibility=1.0)
    return landmarks


def test_compute_features_returns_expected_values():
    frame0 = FrameLandmarks(
        frame_index=0,
        timestamp_ms=0,
        landmarks=make_landmarks({
            LEFT_SHOULDER: (0.4, 0.5), RIGHT_SHOULDER: (0.6, 0.5), NOSE: (0.5, 0.5),
        }),
    )
    frame1 = FrameLandmarks(
        frame_index=1,
        timestamp_ms=100,
        landmarks=make_landmarks({
            RIGHT_SHOULDER: (0.6, 0.5), RIGHT_ELBOW: (0.6, 0.3), RIGHT_WRIST: (0.6, 0.1),
            LEFT_HIP: (0.45, 0.9), RIGHT_HIP: (0.55, 0.9),
            RIGHT_KNEE: (0.55, 1.0), RIGHT_ANKLE: (0.55, 1.2), NOSE: (0.5, 0.5),
        }),
    )
    frame2 = FrameLandmarks(
        frame_index=2, timestamp_ms=200, landmarks=make_landmarks({NOSE: (0.5, 0.5)})
    )
    phases = SwingPhases(ready_frame=0, backswing_frame=0, contact_frame=1, follow_through_frame=2)

    features = compute_features([frame0, frame1, frame2], phases)

    assert features.shoulder_rotation_deg == pytest.approx(0.0, abs=1e-6)
    assert features.elbow_angle_deg == pytest.approx(180.0, abs=1e-6)
    assert features.knee_bend_deg == pytest.approx(180.0, abs=1e-6)
    assert features.contact_height == pytest.approx(0.8, abs=1e-6)
    assert features.contact_depth == pytest.approx(0.1, abs=1e-6)
    assert features.head_stability == pytest.approx(0.0, abs=1e-6)

    vector = features.to_vector()
    assert len(vector) == len(FEATURE_NAMES) == 6


def test_swing_features_vector_round_trip():
    original = SwingFeatures(
        shoulder_rotation_deg=10.0, elbow_angle_deg=160.0, contact_height=0.3,
        contact_depth=0.15, knee_bend_deg=150.0, head_stability=0.01,
    )

    rebuilt = SwingFeatures.from_vector(original.to_vector())

    assert rebuilt == original
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_features.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.features'`

- [ ] **Step 3: Implement feature computation**

`app/features.py`:
```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add app/features.py tests/test_features.py
git commit -m "Add biomechanical feature computation from swing phases"
```

---

### Task 5: Scoring model (learned reference model)

**Files:**
- Create: `app/scoring.py`
- Create: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `app.features.FEATURE_NAMES`, `app.features.SwingFeatures` (Task 4).
- Produces:
  - `app.scoring.Finding` dataclass: `feature: str, value: float, reference_mean: float, z_score: float, severity: str`
  - `app.scoring.ScoreResult` dataclass: `overall_mahalanobis_distance: float, findings: List[Finding]`
  - `app.scoring.ReferenceModel` dataclass with `.save(path: str)` and static `.load(path: str) -> ReferenceModel`
  - `app.scoring.fit_reference_model(feature_vectors: List[List[float]]) -> ReferenceModel` (raises `ValueError` if fewer than 10 vectors)
  - `app.scoring.score_swing(model: ReferenceModel, features: SwingFeatures) -> ScoreResult`

- [ ] **Step 1: Write the failing tests**

`tests/test_scoring.py`:
```python
import random

import pytest

from app.features import SwingFeatures
from app.scoring import fit_reference_model, score_swing, ReferenceModel


def make_reference_vectors(n=20, seed=1):
    rng = random.Random(seed)
    # 6 features, clustered tightly around known means with small noise
    means = [10.0, 160.0, 0.3, 0.15, 150.0, 0.01]
    return [[m + rng.uniform(-1.0, 1.0) for m in means] for _ in range(n)]


def test_fit_reference_model_requires_minimum_clips():
    with pytest.raises(ValueError):
        fit_reference_model(make_reference_vectors(n=5))


def test_fit_reference_model_succeeds_with_enough_clips():
    model = fit_reference_model(make_reference_vectors(n=20))
    assert isinstance(model, ReferenceModel)
    assert len(model.means) == 6


def test_score_swing_flags_no_findings_for_in_range_swing():
    model = fit_reference_model(make_reference_vectors(n=20))
    in_range_features = SwingFeatures(
        shoulder_rotation_deg=10.2, elbow_angle_deg=160.1, contact_height=0.31,
        contact_depth=0.16, knee_bend_deg=150.3, head_stability=0.011,
    )

    result = score_swing(model, in_range_features)

    assert result.findings == []
    assert result.overall_mahalanobis_distance < 3.0


def test_score_swing_flags_outlier_feature():
    model = fit_reference_model(make_reference_vectors(n=20))
    outlier_features = SwingFeatures(
        shoulder_rotation_deg=10.0, elbow_angle_deg=120.0,  # elbow way off reference
        contact_height=0.3, contact_depth=0.15, knee_bend_deg=150.0, head_stability=0.01,
    )

    result = score_swing(model, outlier_features)

    flagged = {f.feature for f in result.findings}
    assert "elbow_angle_deg" in flagged
    assert result.overall_mahalanobis_distance > 3.0


def test_reference_model_save_and_load_round_trip(tmp_path):
    model = fit_reference_model(make_reference_vectors(n=20))
    path = str(tmp_path / "model.pkl")

    model.save(path)
    loaded = ReferenceModel.load(path)

    assert (loaded.means == model.means).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.scoring'`

- [ ] **Step 3: Implement the scoring model**

`app/scoring.py`:
```python
"""Learn what a normal 'good forehand' feature vector looks like from a
reference library and score new swings against it."""
from dataclasses import dataclass
from typing import List
import pickle

import numpy as np
from sklearn.covariance import EmpiricalCovariance

from .features import FEATURE_NAMES, SwingFeatures

FEATURE_Z_THRESHOLD = 1.5
MIN_REFERENCE_CLIPS = 10


@dataclass
class Finding:
    feature: str
    value: float
    reference_mean: float
    z_score: float
    severity: str


@dataclass
class ScoreResult:
    overall_mahalanobis_distance: float
    findings: List[Finding]


@dataclass
class ReferenceModel:
    means: np.ndarray
    stds: np.ndarray
    covariance: EmpiricalCovariance

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ReferenceModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def fit_reference_model(feature_vectors: List[List[float]]) -> ReferenceModel:
    if len(feature_vectors) < MIN_REFERENCE_CLIPS:
        raise ValueError(
            f"Need at least {MIN_REFERENCE_CLIPS} reference clips to fit a model, "
            f"got {len(feature_vectors)}"
        )

    matrix = np.array(feature_vectors)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds == 0] = 1e-6

    standardized = (matrix - means) / stds
    covariance = EmpiricalCovariance().fit(standardized)

    return ReferenceModel(means=means, stds=stds, covariance=covariance)


def score_swing(model: ReferenceModel, features: SwingFeatures) -> ScoreResult:
    vector = np.array(features.to_vector())
    z_scores = (vector - model.means) / model.stds

    squared_distance = model.covariance.mahalanobis(z_scores.reshape(1, -1))[0]
    overall_distance = float(squared_distance ** 0.5)

    findings = []
    for name, value, mean, z in zip(FEATURE_NAMES, vector, model.means, z_scores):
        if abs(z) >= FEATURE_Z_THRESHOLD:
            severity = "significant" if abs(z) >= 2.5 else "moderate"
            findings.append(
                Finding(
                    feature=name, value=float(value), reference_mean=float(mean),
                    z_score=float(z), severity=severity,
                )
            )

    return ScoreResult(overall_mahalanobis_distance=overall_distance, findings=findings)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_scoring.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app/scoring.py tests/test_scoring.py
git commit -m "Add learned reference-model scoring (Mahalanobis + per-feature z-scores)"
```

---

### Task 6: Reference model fitting script

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/fit_reference_model.py`
- Create: `tests/test_fit_reference_model.py`

**Interfaces:**
- Consumes: `app.pose.extract_landmarks` (Task 2), `app.phases.detect_phases` (Task 3), `app.features.compute_features`, `app.features.SwingFeatures.from_vector` (Task 4), `app.scoring.fit_reference_model`, `app.scoring.score_swing` (Task 5).
- Produces: `scripts.fit_reference_model.build_feature_vectors(reference_dir: Path) -> List[List[float]]`, `scripts.fit_reference_model._holdout_check(vectors: List[List[float]]) -> None`, `scripts.fit_reference_model.main() -> None` (CLI entry point).

- [ ] **Step 1: Write the failing tests**

`tests/test_fit_reference_model.py`:
```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_fit_reference_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.fit_reference_model'`

**Note on the hold-out sanity check:** the spec's Testing section requires validating reference-model quality "by holding out a few reference clips from fitting and checking they still score as normal." This script implements that as an automatic step (`_holdout_check`) that runs every time you fit a model, not just as a one-off manual check.

- [ ] **Step 3: Implement the fitting script**

Create `scripts/__init__.py` (empty file, makes `scripts` importable as a package):
```python
```

`scripts/fit_reference_model.py`:
```python
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
        except Exception as exc:
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_fit_reference_model.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/fit_reference_model.py tests/test_fit_reference_model.py
git commit -m "Add offline script to fit the reference model from a clip directory"
```

---

### Task 7: Local LLM feedback module (Ollama)

**Files:**
- Create: `app/feedback.py`
- Create: `tests/test_feedback.py`

**Interfaces:**
- Consumes: `app.scoring.ScoreResult`, `app.scoring.Finding` (Task 5).
- Produces:
  - `app.feedback.OllamaUnavailableError` exception class
  - `app.feedback.generate_feedback(result: ScoreResult, model: str = "llama3.1:8b") -> str`

- [ ] **Step 1: Write the failing tests**

`tests/test_feedback.py`:
```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_feedback.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.feedback'`

- [ ] **Step 3: Implement the feedback module**

`app/feedback.py`:
```python
"""Turns structured scoring findings into natural-language coaching feedback
using a locally running Ollama model. The LLM only rephrases findings that
were already decided by the scoring model — it does not judge technique itself."""
import requests

from .scoring import ScoreResult

OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1:8b"


class OllamaUnavailableError(RuntimeError):
    pass


def _build_prompt(result: ScoreResult) -> str:
    if not result.findings:
        findings_text = "No technique issues were flagged — this swing is close to the reference norm."
    else:
        findings_text = "\n".join(
            f"- {f.feature}: measured {f.value:.1f}, reference average {f.reference_mean:.1f} "
            f"({f.severity} deviation)"
            for f in result.findings
        )

    return (
        "You are a supportive tennis coach reviewing a student's forehand. "
        "Below are technique measurements compared to a reference model of good "
        "forehand form. Write 2-4 short paragraphs of encouraging, specific "
        "coaching feedback: mention what's working, explain each flagged issue "
        "in plain language, and suggest one concrete drill or cue per issue. "
        "Do not invent technique issues beyond what is listed below.\n\n"
        f"Measurements:\n{findings_text}\n"
    )


def generate_feedback(result: ScoreResult, model: str = DEFAULT_MODEL) -> str:
    prompt = _build_prompt(result)

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
    except requests.exceptions.ConnectionError as exc:
        raise OllamaUnavailableError(
            f"Could not reach Ollama at localhost:11434 — is `ollama serve` running "
            f"and has `{model}` been pulled?"
        ) from exc

    response.raise_for_status()
    return response.json()["response"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_feedback.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add app/feedback.py tests/test_feedback.py
git commit -m "Add Ollama-based coaching feedback generation"
```

---

### Task 8: Video annotation module

**Files:**
- Create: `app/annotate.py`
- Create: `tests/test_annotate.py`

**Interfaces:**
- Consumes: `app.pose.FrameLandmarks` (Task 2), `app.phases.SwingPhases` (Task 3).
- Produces:
  - `app.annotate.draw_skeleton(frame, landmarks) -> None` (mutates `frame` in place)
  - `app.annotate.annotate_video(video_path: str, frames: List[FrameLandmarks], output_path: str) -> None`
  - `app.annotate.save_phase_frames(video_path: str, frames: List[FrameLandmarks], phases: SwingPhases, output_dir: str) -> Dict[str, str]` (keys: `"ready", "backswing", "contact", "follow_through"`, values: file paths)

- [ ] **Step 1: Write the failing tests**

`tests/test_annotate.py`:
```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_annotate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.annotate'`

- [ ] **Step 3: Implement video annotation**

`app/annotate.py`:
```python
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
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
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

    cap.release()
    writer.release()


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
    output_paths = {}

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

    cap.release()
    return output_paths
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_annotate.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/annotate.py tests/test_annotate.py
git commit -m "Add OpenCV pose skeleton video/frame annotation"
```

---

### Task 9: Pipeline orchestration

**Files:**
- Create: `app/pipeline.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `app.pose.extract_landmarks`, `app.phases.detect_phases`, `app.features.compute_features`, `app.scoring.ReferenceModel`, `app.scoring.score_swing`, `app.feedback.generate_feedback`, `app.annotate.annotate_video`, `app.annotate.save_phase_frames` (Tasks 2-8).
- Produces:
  - `app.pipeline.LowPoseConfidenceError` exception class
  - `app.pipeline.AnalysisResult` dataclass: `feedback_text: str, annotated_video_path: str, phase_frame_paths: Dict[str, str]`
  - `app.pipeline.analyze_forehand(video_path: str, model_path: str, output_dir: str, pose_model_path: str = "models/pose_landmarker_full.task") -> AnalysisResult`

- [ ] **Step 1: Write the failing tests**

`tests/test_pipeline.py`:
```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.pipeline'`

- [ ] **Step 3: Implement the pipeline orchestrator**

`app/pipeline.py`:
```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/pipeline.py tests/test_pipeline.py
git commit -m "Add pipeline orchestration wiring pose through to feedback"
```

---

### Task 10: FastAPI upload/results endpoints + templates

**Files:**
- Modify: `app/main.py`
- Create: `templates/upload.html`
- Create: `templates/results.html`
- Modify: `tests/test_main.py`

**Interfaces:**
- Consumes: `app.pipeline.analyze_forehand`, `app.pipeline.LowPoseConfidenceError` (Task 9), `app.feedback.OllamaUnavailableError` (Task 7).
- Produces: `GET /` (upload form), `POST /analyze` (runs analysis, renders results or re-renders form with an error).

- [ ] **Step 1: Write the failing tests**

Replace the contents of `tests/test_main.py` with:
```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_main.py -v`
Expected: FAIL — `GET /` returns 404, `/analyze` doesn't exist, `OllamaUnavailableError` import error.

- [ ] **Step 3: Implement the upload/results routes and templates**

`templates/upload.html`:
```html
<!DOCTYPE html>
<html>
<head><title>Tennis Forehand Analyzer</title></head>
<body>
  <h1>Upload a Forehand Clip</h1>
  {% if error %}
    <p style="color: red;">{{ error }}</p>
  {% endif %}
  <form action="/analyze" method="post" enctype="multipart/form-data">
    <input type="file" name="video" accept="video/*" required>
    <button type="submit">Analyze</button>
  </form>
</body>
</html>
```

`templates/results.html`:
```html
<!DOCTYPE html>
<html>
<head><title>Forehand Analysis Results</title></head>
<body>
  <h1>Analysis Results</h1>
  <video src="{{ annotated_video_url }}" controls width="480"></video>

  <h2>Key Phases</h2>
  {% for phase_name, url in phase_urls.items() %}
    <figure>
      <img src="{{ url }}" width="240">
      <figcaption>{{ phase_name }}</figcaption>
    </figure>
  {% endfor %}

  <h2>Coaching Feedback</h2>
  <p>{{ feedback_text }}</p>

  <a href="/">Analyze another clip</a>
</body>
</html>
```

`app/main.py` (full replacement):
```python
"""FastAPI web app: upload a forehand clip, get back annotated video + coaching feedback."""
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .pipeline import analyze_forehand, LowPoseConfidenceError
from .feedback import OllamaUnavailableError

REFERENCE_MODEL_PATH = "models/reference_model.pkl"

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, video: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = str(Path(tmp_dir) / video.filename)
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)

        output_dir = str(Path(tmp_dir) / "output")

        try:
            result = analyze_forehand(video_path, REFERENCE_MODEL_PATH, output_dir)
        except (LowPoseConfidenceError, OllamaUnavailableError) as exc:
            return templates.TemplateResponse(
                "upload.html", {"request": request, "error": str(exc)}
            )

        served_dir = Path("static/results")
        served_dir.mkdir(parents=True, exist_ok=True)

        annotated_name = "annotated.mp4"
        shutil.copy(result.annotated_video_path, served_dir / annotated_name)

        phase_urls = {}
        for phase_name, path in result.phase_frame_paths.items():
            dest_name = f"{phase_name}.jpg"
            shutil.copy(path, served_dir / dest_name)
            phase_urls[phase_name] = f"/static/results/{dest_name}"

        return templates.TemplateResponse(
            "results.html",
            {
                "request": request,
                "feedback_text": result.feedback_text,
                "annotated_video_url": f"/static/results/{annotated_name}",
                "phase_urls": phase_urls,
            },
        )


app.mount("/static", StaticFiles(directory="static"), name="static")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_main.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app/main.py templates/ tests/test_main.py
git commit -m "Add upload and results web pages"
```

---

### Task 11: End-to-end wiring — real reference clips and manual verification

This task has no unit test of its own — it wires the real dependencies together and is verified by hand, matching the spec's testing section ("Pose extraction / phase detection... validated via manual spot-checks").

**Files:** none (no code changes — this is a setup + manual verification task).

- [ ] **Step 1: Install and start Ollama, pull the model**

Run (macOS example — adjust for your platform):
```bash
brew install ollama
ollama serve &
ollama pull llama3.1:8b
```
Expected: `ollama pull` completes; `curl http://localhost:11434/` returns `Ollama is running`.

- [ ] **Step 2: Gather reference "good forehand" clips**

Per the spec's "Reference data" section: film a strong player (club pro, advanced player) yourself, or source permissively-licensed instructional content — never scraped Grand Slam/broadcast footage. Place at least 10 short (one-swing) clips into `reference_clips/` as `.mp4`/`.mov`/`.m4v` files.

- [ ] **Step 3: Fit the real reference model**

Run: `python scripts/fit_reference_model.py reference_clips/ models/reference_model.pkl`
Expected: prints `OK` or `SKIP` per clip, then either a hold-out sanity check line (if you supplied 15+ clips) or a "Skipping hold-out sanity check" line, then `Fitting final model on all N reference clips...` and `Saved model to models/reference_model.pkl`. If it exits with `No video files found`, add clips to `reference_clips/` first. If it exits with a `ValueError` about needing 10 clips, add more clips.

- [ ] **Step 4: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass (pose smoke test will run for real now if you added `tests/fixtures/sample_clip.mp4`, otherwise it's skipped).

- [ ] **Step 5: Start the app and verify end-to-end manually**

Run: `uvicorn app.main:app --reload`

In a browser, visit `http://localhost:8000/`, upload a short trimmed clip of a student's forehand (per the spec's key assumption: one clean swing, few seconds, student clearly visible), and confirm:
- the results page loads with an annotated video showing the pose skeleton overlay
- four key-phase images (ready/backswing/contact/follow-through) are shown
- the written coaching feedback reads naturally and references measurements that plausibly match what you see in the video
- uploading a clip with no person clearly visible (e.g. filmed from too far away) shows the "couldn't get a clear read on this clip" error instead of bad output

- [ ] **Step 6: Commit any setup notes**

If you want to document the Ollama/reference-clip setup steps for your own future reference, add a `README.md` at the repo root summarizing Steps 1-3 above, then:
```bash
git add README.md
git commit -m "Document local setup steps for Ollama and reference model fitting"
```

---
