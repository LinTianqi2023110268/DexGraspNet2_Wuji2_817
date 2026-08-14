#!/usr/bin/env bash
set -euo pipefail

# Native Isaac Sim viewport capture.  No X11/window lookup/desktop recording
# and no external ffmpeg executable are used.
ROOT="/home/lin/Projects/DexGraspNet2_Wuji2"
CONTROL_ROOT="$ROOT/08_dual_arm_scene_layout/isaaclab_control"
VIDEO="$CONTROL_ROOT/outputs/full_pick_place_25s_dog_candidate3800/videos/full_pick_place_replay_24.85s.mp4"

exec bash "$CONTROL_ROOT/run_replay_25s.sh" \
  --export-mp4 \
  --video-output "$VIDEO" \
  --video-width 1920 \
  --video-height 1080 \
  --video-fps 15 \
  --video-frame-stride 2 \
  "$@"
