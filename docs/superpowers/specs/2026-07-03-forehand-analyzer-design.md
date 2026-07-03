# Local Forehand Stroke Analyzer (Web, v1) — Design

## Goal

A web app where a tennis coach uploads a short video clip of a student hitting a
forehand and receives:
- an annotated video/key frames showing pose overlay at key swing phases, and
- written coaching feedback (strengths, issues, drills/cues to fix them).

All analysis runs on local compute — no paid AI APIs. Single user, no accounts,
no database. Forehand only in v1.

This project is also an explicit vehicle for the user to learn practical AI/ML
— the "what does good technique look like" step should be genuinely learned
from data (not hand-coded rules), even though that costs more effort than a
pure rule-based v1 would.

## Non-goals (v1)

- Not analyzing backhand, serve, volley, or any stroke besides forehand.
- Not segmenting multiple swings out of a long match/rally video.
- Not supporting multiple users, logins, or per-student history tracking.
- Not comparing against a reference "pro" clip side-by-side.

## Key assumptions

- **Each uploaded clip contains exactly one clean forehand swing**, a few
  seconds long, with one student clearly visible. The coach trims footage
  before upload. Automatically detecting and segmenting multiple swings out
  of a longer video is a materially harder problem and is explicitly
  deferred to a future version.
- **"Good technique" is learned from a reference library of clean forehand
  clips**, not hand-coded thresholds. Building that reference library is
  required before the scoring model can work at all — see "Reference data"
  below for sourcing constraints.

## Architecture

```
Reference clips (good forehands) → same pose/feature pipeline → fit statistical
                                    model of "normal" good-forehand feature space
                                    (done once, offline, before v1 can be used)

Upload (browser) → FastAPI backend
                      ├─ 1. Pose extraction (MediaPipe, per-frame keypoints)
                      ├─ 2. Phase detection (ready / backswing / contact / follow-through)
                      ├─ 3. Metric computation (angles, contact point, balance, etc.)
                      ├─ 4. Scoring model (metrics → outlier findings, vs. learned reference)
                      ├─ 5. Local LLM (Ollama) turns findings into coaching prose
                      └─ 6. Annotated video/frames (OpenCV skeleton overlay)
                    → Results page: annotated video + written report
```

## Components

### 1. Pose extraction
MediaPipe Pose Landmarker runs per-frame over the uploaded clip, producing 33
body keypoints (with confidence) per frame. Runs locally on CPU, no API cost.

### 2. Phase detection
Heuristic signal processing over the keypoint sequence (no ML model) to
locate:
- **ready position** — start of clip / stable stance before backswing
- **backswing peak** — extreme point of racket-hand/wrist position before
  forward swing begins
- **contact point** — frame of peak wrist speed during the forward swing
- **follow-through** — settled position after contact

### 3. Metric computation
From keypoints at each detected phase, compute:
- Shoulder/hip rotation angle during backswing (unit turn)
- Elbow angle at contact
- Contact point height and horizontal distance relative to the body
- Knee bend / weight transfer at contact
- Balance / head stability across the swing

### 4. Scoring model (learned, not hand-coded)
Rather than hand-picked thresholds, "good technique" is learned from a
reference library of clean forehand clips:

1. Run the same pose extraction + metric computation (steps 1-3) over every
   reference clip, producing a feature vector per clip (shoulder rotation,
   elbow angle at contact, contact point position, knee bend, balance, etc.).
2. Fit a statistical model of the "normal" region of that feature space using
   only good examples — e.g. a Gaussian/Mahalanobis-distance model or a
   one-class SVM. This is a one-time offline step done before the app is
   usable, re-run only when the reference library changes.
3. At inference time, a student's clip's feature vector is scored against
   this fitted model. Features that fall outside the learned normal region
   become structured findings, e.g. `{issue: "contact_point_outlier",
   severity: "moderate", metric_value: ..., reference_range: ...}`.

This layer decides *what* is technically off, learned from data rather than
guessed thresholds. It is pure data-in/data-out logic, independent of the LLM
and independent of video — testable by fitting on synthetic "good" feature
fixtures and checking known-bad vectors are flagged as outliers.

### Reference data (required before the scoring model works at all)
The reference library of "good forehand" clips must come from a legitimate,
non-copyright-infringing source — this is a hard constraint, not a
nice-to-have:
- Grand Slam / professional match broadcast footage is copyrighted content
  owned by the tournaments/broadcasters. Scraping or downloading it (e.g. via
  YouTube-downloading tools) violates YouTube's Terms of Service and the
  broadcasters' copyright, even for a personal, non-commercial project. This
  approach is explicitly ruled out.
- Acceptable sources: footage the user films themselves (e.g. a club pro or
  strong player, ideally using the same camera setup/angle as student
  footage), or instructional content whose license/terms clearly permit this
  kind of personal/educational use.
- **Open item:** exact source(s) and target quantity (~30-100 clips) to be
  finalized before implementation of this component begins.

### 5. Local LLM feedback generation
Findings from the rule engine (not raw video or images) are passed to a
locally-running Ollama model (e.g. `llama3.1:8b`) via its local HTTP API,
with a coaching-persona prompt. The LLM's job is limited to *phrasing* —
turning structured findings into natural, encouraging written feedback with
concrete cues/drills. The LLM does not make its own technique judgments; it
narrates the rule engine's output. This keeps feedback grounded and avoids
hallucinated claims about technique the model didn't actually verify.

### 6. Video annotation
OpenCV draws the pose skeleton onto video frames and produces:
- an annotated version of the clip (skeleton overlay throughout), and
- highlighted still frames for each detected phase (ready/backswing/contact/
  follow-through) for use in the written report.

## Stack

- **Backend:** Python + FastAPI, single process.
- **Computer vision:** MediaPipe (pose estimation) + OpenCV (video I/O,
  drawing overlays).
- **Scoring model:** scikit-learn (e.g. `EmpiricalCovariance`/Mahalanobis
  distance or `OneClassSVM`) fitted on reference-clip feature vectors, plus
  numpy/pandas for feature handling. Model artifact is saved to disk and
  reloaded at inference time (no retraining per request).
- **LLM:** Ollama running locally (e.g. `llama3.1:8b`), called via its local
  HTTP API. No external API keys, no per-request cost.
- **Frontend:** simple server-rendered pages (upload form → results page).
  No SPA framework needed for v1.
- **Storage:** none persistent — uploaded video and generated artifacts live
  in a scratch directory for the duration of processing and are served back
  to the browser, then cleaned up. No database, no accounts.

## Error handling

- If pose-detection confidence is too low across too much of the clip (poor
  lighting, student not clearly framed, camera too far away), reject the
  clip with a clear message asking for a clearer recording, rather than
  producing an unreliable analysis.
- If more than one person is detected in frame, the pipeline picks the most
  central/largest figure as the subject; if the choice is ambiguous, warn the
  user rather than silently guessing.
- If Ollama is unreachable or the model isn't pulled locally, surface a clear
  setup error rather than a generic 500.

## Testing

- **Scoring model:** unit-testable by fitting on synthetic "good" feature
  fixtures and asserting that known-in-range vectors score as normal and
  known-out-of-range vectors are flagged as outliers with the expected
  feature attributed.
- **Pose extraction / phase detection:** not meaningfully unit-testable in
  the traditional sense — validated via manual spot-checks against a small
  set of real sample forehand clips provided by the user during development.
- **Reference model quality:** validated by holding out a few reference
  clips from fitting and checking they still score as "normal" (basic
  sanity/overfitting check given the small data size).
- **LLM feedback step:** validated by manual review of generated coaching
  text against the scoring model's findings for a handful of sample cases,
  checking the prose stays faithful to the findings and doesn't invent
  claims.

## Open questions / future versions (explicitly out of scope now)

- Exact source and quantity of reference "good forehand" clips (see
  "Reference data" above) — must be resolved before implementation of the
  scoring model begins.
- Multi-swing segmentation from longer videos.
- Additional strokes (backhand, serve, volley).
- Per-student accounts and progress history over time.
- Side-by-side comparison against reference/pro footage.
- Upgrading the scoring model to a trained neural approach (e.g. an
  autoencoder over pose sequences) once enough reference data/experience
  exists — a natural "more ML depth" follow-up to the v1 statistical model.
