#!/bin/zsh
# Orquestación del experimento principal (consolidación en lote de los 27 contextos).
# Ejecutar desde cortex_local/. Cada paso es idempotente salvo que se borren las salidas.
set -e
set -o pipefail
PY=${PY:-python}
cd "$(dirname "$0")"
mkdir -p logs results checkpoints figures

run() { echo "\n=== $* ==="; "$@" 2>&1 | grep -v -E "Fetching|Finished processing" | tee -a logs/run_experiments.log; }

# 1. líneas de referencia: modelo base sin contexto y con el contexto correcto (oráculo)
[ -f results/eval_vanilla.json ] || run $PY 03_eval.py --name vanilla
[ -f results/eval_context.json ] || run $PY 03_eval.py --name context --context --skip-retain

# 2. objetivos del profesor (una sola vez)
[ -f cache/G129.npz ] || run $PY 04_cache_teacher.py

# 3. Cortex: dos formas de plasticidad con el mismo presupuesto de parámetros (9.4M)
[ -d checkpoints/synaptic_r16 ]  || run $PY 05_train.py --name synaptic_r16 --lr 1e-4  --variant synaptic   --rank 16 --objective kl_mse --epochs 6 --generic 260 --save-epochs 3
[ -f results/eval_synaptic_r16.json ] || run $PY 03_eval.py --name synaptic_r16 --adapter checkpoints/synaptic_r16
[ -f results/eval_synaptic_r16_ep3.json ] || run $PY 03_eval.py --name synaptic_r16_ep3 --adapter checkpoints/synaptic_r16_ep3
[ -d checkpoints/structural_k64 ] || run $PY 05_train.py --name structural_k64 --lr 1e-4 --variant structural --rank 64 --objective kl_mse --epochs 6 --generic 260 --save-epochs 3
[ -f results/eval_structural_k64.json ] || run $PY 03_eval.py --name structural_k64 --adapter checkpoints/structural_k64
[ -f results/eval_structural_k64_ep3.json ] || run $PY 03_eval.py --name structural_k64_ep3 --adapter checkpoints/structural_k64_ep3

# 4. líneas base "ingenuas"
#    a) SFT clásico: LoRA en todo el modelo con entropía cruzada sobre las mismas respuestas
[ -d checkpoints/lora_all_ce ] || run $PY 05_train.py --name lora_all_ce --lr 1e-4 --variant lora_all --rank 16 --objective ce --epochs 6 --generic 260
[ -f results/eval_lora_all_ce.json ] || run $PY 03_eval.py --name lora_all_ce --adapter checkpoints/lora_all_ce
#    b) "seguir preentrenando" sobre el texto del catálogo
[ -d checkpoints/docs_ce ] || run $PY 05_train.py --name docs_ce --lr 1e-4 --variant lora_all --rank 16 --data docs --epochs 20 --accum 2
[ -f results/eval_docs_ce.json ] || run $PY 03_eval.py --name docs_ce --adapter checkpoints/docs_ce

# 5. ablaciones: sin MSE; LoRA FFN con CE (misma localización, sin destilación)
[ -d checkpoints/synaptic_kl ] || run $PY 05_train.py --name synaptic_kl --lr 1e-4 --variant synaptic --rank 16 --objective kl --epochs 6 --generic 260
[ -f results/eval_synaptic_kl.json ] || run $PY 03_eval.py --name synaptic_kl --adapter checkpoints/synaptic_kl
[ -d checkpoints/synaptic_ce ] || run $PY 05_train.py --name synaptic_ce --lr 1e-4 --variant synaptic --rank 16 --objective ce --epochs 6 --generic 260
[ -f results/eval_synaptic_ce.json ] || run $PY 03_eval.py --name synaptic_ce --adapter checkpoints/synaptic_ce

# 6. métricas y figuras
run $PY 06_metrics.py
echo "\nFIN experimento principal"
