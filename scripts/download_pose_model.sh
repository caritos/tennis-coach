#!/usr/bin/env bash
set -euo pipefail
mkdir -p models
curl -L -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
echo "Downloaded models/pose_landmarker_full.task"
