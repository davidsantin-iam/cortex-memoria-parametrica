#!/bin/zsh
# Ablación adicional: KL literal del paper con la tasa de aprendizaje corregida (aísla el efecto de la KL del de Adam).
cd "$(dirname "$0")"
PY=${PY:-python}
until ! pgrep -f run_v2.sh > /dev/null; do sleep 120; done
[ -d checkpoints/synaptic_fullvocab_lr1e-4 ] || $PY 05_train.py --name synaptic_fullvocab_lr1e-4 --variant synaptic --rank 16 --objective kl_mse --kl-full-vocab --lr 1e-4 --epochs 6 --generic 260 2>&1 | grep -E "guardado" | tail -1
[ -f results/eval_synaptic_fullvocab_lr1e-4.json ] || $PY 03_eval.py --name synaptic_fullvocab_lr1e-4 --adapter checkpoints/synaptic_fullvocab_lr1e-4 2>&1 | grep -E "retain|target" | tail -2
echo "EXTRA DONE $(date)"
