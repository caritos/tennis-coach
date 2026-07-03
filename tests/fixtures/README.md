# Test fixtures

Drop a short real forehand clip named `sample_clip.mp4` in this directory to
enable the pose-extraction smoke test in `tests/test_pose.py`. Without it,
that test is skipped (MediaPipe pose extraction can't be meaningfully unit
tested without a real video of a real person).
