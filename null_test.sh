#!/bin/zsh
# Prueba nula: solo ejemplos genéricos (profesor == alumno). La pérdida debe quedarse ~0 y la perplejidad no debe moverse.
PY=${PY:-python}
for cfg in "5e-4 1e-8" "1e-4 1e-8" "2e-5 1e-8" "5e-4 1e-6"; do
  lr=${cfg%% *}; eps=${cfg##* }; name=null_lr${lr}_eps${eps}
  $PY 05_train.py --name $name --variant synaptic --rank 16 --objective kl_mse --epochs 1 --generic 130 --ctx-filter NONE --lr $lr --adam-eps $eps 2>&1 | grep -E "guardado" | cut -c1-60
  last=$(tail -1 logs/train_$name.jsonl | $PY -c "import sys,json; r=json.loads(sys.stdin.read()); print(f'loss final {r[\"loss\"]:.4f} (kl {r[\"kl\"]:.3f} mse {r[\"mse\"]:.3f})')")
  ppl=$($PY - <<PYEOF 2>/dev/null
import importlib, mlx.core as mx
from common import *
ev = importlib.import_module("03_eval")
m, tok = load_model(); load_adapter(m, CKPT/"$name"); fold_adapters(m); mx.eval(m.parameters()); m.eval()
print(round(ev.perplexity(m, tok, read_jsonl(DATA/"eval_ppl.jsonl")), 3))
PYEOF
)
  echo "lr=$lr eps=$eps -> $last | PPL wiki = $ppl (base 9.085)"
done
