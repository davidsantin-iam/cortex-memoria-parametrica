#!/bin/zsh
# Variante v3: preguntas del 2B (por facetas, filtradas) + respuestas del 27B filtradas contra el XML.
cd "$(dirname "$0")"
PY=${PY:-python}
until grep -q "^FIN:" logs/08_outside_teacher.log 2>/dev/null; do sleep 120; done
lms unload --all > /dev/null 2>&1
[ -f cache_v3/X00000.npz ] || $PY 04_cache_teacher.py --rehearsal rehearsal_v3.jsonl --cache-dir cache_v3 2>&1 | grep -v Fetching | tail -2
for v in "synaptic 16" "structural 64"; do
  var=${v%% *}; r=${v##* }; name=${var}_v3
  [ -d checkpoints/$name ] || $PY 05_train.py --name $name --variant $var --rank $r --objective kl_mse --lr 1e-4 --epochs 6 --generic 260 --rehearsal rehearsal_v3.jsonl --cache-dir cache_v3 2>&1 | grep -E "guardado" | tail -1
  [ -f results/eval_$name.json ] || $PY 03_eval.py --name $name --adapter checkpoints/$name 2>&1 | grep -E "target_score|retain_score" | tail -2
done
$PY 06_metrics.py --seq seq_structural_ 2>&1 | tail -1
echo "V3 DONE $(date)"
