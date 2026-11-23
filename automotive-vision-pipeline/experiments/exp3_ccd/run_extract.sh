#!/bin/sh
# usage: run_extract.sh <shard i> <n shards> ; crash clips first, then normal clips (resumable)
cd "$(dirname "$0")"
HERE=$(pwd)
PY=$HERE/../../.venv/Scripts/python.exe
VID=$HERE/../../data/CarCrash/videos
mkdir -p results/logs results_normal/logs
EXP3_CCD_DIR=$VID/Crash-1500 EXP3_OUT=$HERE/results OMP_NUM_THREADS=1 $PY 01_extract_features.py --demo 0 --device 0 --shard $1/$2 >> results/logs/crash_$1.log 2>&1
EXP3_CCD_DIR=$VID/Normal EXP3_OUT=$HERE/results_normal OMP_NUM_THREADS=1 $PY 01_extract_features.py --demo 0 --device 0 --shard $1/$2 >> results_normal/logs/n_$1.log 2>&1
