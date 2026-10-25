#!/bin/zsh
# Experimento adicional: ensayo prospectivo guiado por facetas (6 preguntas por infracción).
cd "$(dirname "$0")"
PY=${PY:-python}
# (espera a la cadena principal eliminada: ya terminó)
until [ -f data/rehearsal_v2.jsonl ] && ! pgrep -f "02_rehearsal.py" > /dev/null; do sleep 60; done
mkdir -p cache_v2 && cp -n cache/G*.npz cache_v2/ 2>/dev/null
[ -f cache_v2/E00000.npz ] || $PY 04_cache_teacher.py --rehearsal rehearsal_v2.jsonl --cache-dir cache_v2 2>&1 | grep -v Fetching | tail -2
for v in "synaptic 16" "structural 64"; do
  var=${v%% *}; r=${v##* }; name=${var}_v2
  [ -d checkpoints/$name ] || $PY 05_train.py --name $name --variant $var --rank $r --objective kl_mse --lr 1e-4 --epochs 6 --generic 260 --rehearsal rehearsal_v2.jsonl --cache-dir cache_v2 2>&1 | grep -E "^ep|guardado" | tail -3
  [ -f results/eval_$name.json ] || $PY 03_eval.py --name $name --adapter checkpoints/$name 2>&1 | grep -v -E "Fetching|Finished" | tail -3
done
$PY 06_metrics.py
echo "V2 DONE $(date)"
