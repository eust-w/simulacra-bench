#!/bin/sh
# Runs inside pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime on 1 GPU.
# Usage: sh job_gpu_eval.sh <submission-dir-name> [more...]
set -e
BASE=/data/personal/simulacra-bench
cd $BASE/code
python -m venv --system-site-packages /tmp/venv
/tmp/venv/bin/pip install -q --no-index --find-links $BASE/wheelhouse-py311 pandas==2.2.3 pyarrow==18.1.0 pyyaml==6.0.2
PY=/tmp/venv/bin/python
$PY -c "import torch,pandas,numpy;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'pandas',pandas.__version__,'numpy',numpy.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p $BASE/results
for SUB in "$@"; do
  for SB in _sandbox _sandbox_v2; do
    for PH in 1 2; do
      echo "=== $SUB $SB phase $PH"
      $PY tools/eval_local.py --submission submissions/$SUB --sandbox starter-kit/$SB --phase $PH --json $BASE/results/gpu_scores.jsonl
    done
  done
  echo "=== official score.py (phase 1, noised) $SUB"
  cd starter-kit
  for S in unicef world_bank unhcr; do $PY score.py --submission ../submissions/$SUB --data _sandbox/$S --schema data/$S.json --phase 1 --python $PY 2>&1 | tail -2; done
  cd ..
done
echo "[job] done"
