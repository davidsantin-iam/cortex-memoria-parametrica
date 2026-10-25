#!/bin/zsh
cd "$(dirname "$0")"
PY=${PY:-python}
./run_experiments.sh
$PY 07_sequential.py --variant structural --rank 32 2>&1 | tee -a logs/run_experiments.log
$PY 07_sequential.py --variant structural --rank 32 --no-replay 2>&1 | tee -a logs/run_experiments.log
echo "ALL DONE $(date)" >> logs/run_experiments.log
