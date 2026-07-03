# Local Forehand Stroke Analyzer (Web, v1) — Design

## Goal

A web app where a tennis coach uploads a short video clip of a student hitting a
forehand and receives:
- an annotated video/key frames showing pose overlay at key swing phases, and
- written coaching feedback (strengths, issues, drills/cues to fix them).

All analysis runs on local compute — no paid AI APIs. Single user, no accounts,
no database. Forehand only in v1.

## Non-goals (v1)

- Not analyzing backhand, serve, volley, or any stroke besides forehand.
- Not segmenting multiple swings out of a long match/rally video.
- Not supporting multiple users, logins, or per-student history tracking.
- Not comparing against a reference "pro" clip side-by-side.

## Key assumption

**Each uploaded clip contains exactly one clean forehand swing**, a few
seconds long, with one student clearly visible. The coach trims footage
before upload. Automatically detecting and segmenting multiple swings out of
a longer video is a materially harder problem and is explicitly deferred to
a future version.

## Architecture

```
Upload (browser) → FastAPI backend
                      ├─ 1. Pose extraction (MediaPipe, per-frame keypoints)
                      ├─ 2. Phase detection (ready / backswing / contact / follow-through)
                      ├─ 3. Metric computation (angles, contact point, balance, etc.)
                      ├─ 4. Rule engine (metrics → structured findings)
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

### 4. Rule engine
Deterministic thresholds (based on standard forehand technique cues) map
metrics to structured findings, e.g. `{issue: "late_contact_point", severity:
"moderate", metric_value: ...}`. This layer decides *what* is technically
right or wrong. It is pure data-in/data-out logic, independent of the LLM and
independent of video — testable with synthetic keypoint fixtures.

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

- **Rule engine:** unit-testable in isolation using synthetic keypoint
  fixtures representing known-good and known-bad forehand mechanics, with
  expected findings asserted directly.
- **Pose extraction / phase detection:** not meaningfully unit-testable in
  the traditional sense — validated via manual spot-checks against a small
  set of real sample forehand clips provided by the user during development.
- **LLM feedback step:** validated by manual review of generated coaching
  text against the rule engine's findings for a handful of sample cases,
  checking the prose stays faithful to the findings and doesn't invent
  claims.

## Open questions / future versions (explicitly out of scope now)

- Multi-swing segmentation from longer videos.
- Additional strokes (backhand, serve, volley).
- Per-student accounts and progress history over time.
- Side-by-side comparison against reference/pro footage.
