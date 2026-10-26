#!/bin/zsh
# v5: mucho volumen de preguntas propias (2 rondas de ensayo por facetas) filtradas contra el XML,
# contextos con el articulado explícito, y preguntas de articulado con respuesta de plantilla.
cd "$(dirname "$0")"
PY=${PY:-python}
set -e
[ -f data/contexts_v5.json ] || $PY 09_build_v5.py contexts
for seed in 101 102; do
  r=$((seed-100))
  [ -f data/rehearsal_v5_r$r.jsonl ] || $PY -u 02_rehearsal.py --facets --seed $seed --contexts contexts_v5.json --out rehearsal_v5_r$r.jsonl --rid-prefix R${r}_ --skip-generic 2>&1 | grep -v -E "Fetching|Finished processing" | grep -E "preguntas válidas|ejemplos de ensayo|recuperadas"
done
$PY 09_build_v5.py merge
[ -f cache_v5/G129.npz ] || $PY 04_cache_teacher.py --rehearsal rehearsal_v5.jsonl --cache-dir cache_v5 --contexts contexts_v5.json 2>&1 | grep -v Fetching | tail -1
train_eval() {  # nombre variante rango fichero_ensayo
  [ -d checkpoints/$1 ] || $PY 05_train.py --name $1 --variant $2 --rank $3 --objective kl_mse --lr 1e-4 --epochs 6 --generic 260 --rehearsal $4 --cache-dir cache_v5 2>&1 | grep -E "guardado|ejemplos/época" | tail -2
  [ -f results/eval_$1.json ] || $PY 03_eval.py --name $1 --adapter checkpoints/$1 2>&1 | grep -E "target_score|retain_score|ppl_wiki" | tail -3
}
train_eval synaptic_v5 synaptic 16 rehearsal_v5.jsonl
train_eval structural_v5 structural 64 rehearsal_v5.jsonl
train_eval synaptic_v5self synaptic 16 rehearsal_v5_self.jsonl
$PY 06_metrics.py --seq seq_structural_ 2>&1 | tail -1
echo "V5 DONE $(date)"
