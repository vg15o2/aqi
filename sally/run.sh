#!/bin/bash
# Airport Queue Intelligence — launcher
#
# Usage:
#   ./run.sh --source video.mp4 --zones zones.json                 # run
#   ./run.sh --source video.mp4 --zones zones.json --draw-zones    # draw zones
#   ./run.sh --source video.mp4 --zones zones.json --conf 0.25     # custom conf

cd "$(dirname "$0")"
python3 run.py "$@"
