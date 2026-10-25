"""
Utilidades compartidas para la réplica local de Cortex (Zhang et al., 2026)
sobre Qwen3.5-2B con MLX.

Contiene:
  - rutas y constantes
  - carga del modelo / tokenizador
  - renderizado de contextos (catálogo de sanciones) y construcción de prompts
  - forward con captura de estados ocultos post-FFN por capa
  - plasticidad "sináptica" (LoRA solo en FFN) y "estructural" (canales nuevos en el FFN)
  - plegado (fold) de los adaptadores en los pesos base
  - guardado / carga de adaptadores
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Iterable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm import load as mlx_load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.tuner.lora import LoRALinear

# ----------------------------------------------------------------------------
# Rutas
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CKPT = ROOT / "checkpoints"
LOGS = ROOT / "logs"
for _d in (DATA, RESULTS, FIGURES, CKPT, LOGS):
    _d.mkdir(exist_ok=True)

NORMATIVA_XML = ROOT.parent / "normativa" / "getSanciones.txt"

MODEL_ID = "mlx-community/Qwen3.5-2B-bf16"

# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------
SYSTEM_PLAIN = (
    "Eres un asistente experto en la normativa de tráfico y en el catálogo de "
    "infracciones y sanciones del Ayuntamiento de Madrid. Responde en español, "
    "de forma directa y concisa."
)

# El profesor ve el contexto; se le pide que no lo mencione (filtro del paper, Appx. B.2)
SYSTEM_TEACHER = (
    SYSTEM_PLAIN
    + " Responde a la pregunta usando la información del catálogo que se te facilita, "
    "pero sin mencionar nunca que dispones de un texto, catálogo o contexto: contesta "
    "como si lo supieras de memoria. Máximo tres frases."
)

CONTEXT_HEADER = (
    "Catálogo de infracciones y sanciones de tráfico del Ayuntamiento de Madrid "
    "(extracto). Cada entrada indica la descripción de la infracción, el importe de la "
    "multa, el importe con reducción por pronto pago, los puntos del permiso que se "
    "retiran, la calificación de gravedad y el precepto que la tipifica.\n"
)

NORM_NAMES = {
    "LSV": "Ley de Seguridad Vial",
    "VEH": "Reglamento General de Vehículos",
    "OMS": "Ordenanza de Movilidad Sostenible",
    "CON": "Reglamento General de Conductores",
    "SOA": "Ley sobre responsabilidad civil y seguro (SOA)",
    "RGC": "Reglamento General de Circulación",
    "CIR": "Reglamento General de Circulación",
}


def fmt_eur(v: str | float) -> str:
    f = float(v)
    if f.is_integer():
        return f"{int(f)} €"
    return f"{f:.2f}".replace(".", ",") + " €"


def render_entry(s: dict) -> str:
    puntos = int(float(s["puntos"]))
    pts = "0 (no se retiran puntos)" if puntos == 0 else str(puntos)
    return (
        f"- Infracción: {s['descripcion']}\n"
        f"  Importe: {fmt_eur(s['cuantia'])}. Con reducción por pronto pago: {fmt_eur(s['reduccion'])}. "
        f"Puntos: {pts}. Gravedad: {s['gravedad']}. Precepto: {s['precepto']}."
    )


def render_context(sanciones: Iterable[dict]) -> str:
    return CONTEXT_HEADER + "\n".join(render_entry(s) for s in sanciones)


# ----------------------------------------------------------------------------
# Modelo y tokenizador
# ----------------------------------------------------------------------------
def load_model(model_id: str = MODEL_ID):
    model, tok = mlx_load(model_id)
    return model, tok


def chat_prompt_ids(tok, user: str, system: str | None = SYSTEM_PLAIN) -> list[int]:
    """Ids del prompt (hasta el inicio de la respuesta del asistente), sin 'thinking'."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
    return list(ids)


def answer_ids(tok, answer: str) -> list[int]:
    """Ids de la respuesta del asistente más el token de fin de turno."""
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    return list(tok.encode(answer, add_special_tokens=False)) + [im_end]


def text_model(model):
    """Devuelve el Qwen3_5TextModel interno (capas, embed, norm)."""
    return model.language_model.model


def forward_with_hidden(model, ids: mx.array):
    """Forward completo devolviendo logits y la lista de estados ocultos
    tras la actualización residual del FFN de cada capa (post-FFN)."""
    tm = text_model(model)
    h = tm.embed_tokens(ids)
    fa_mask = create_attention_mask(h, None)
    ssm_mask = create_ssm_mask(h, None)
    hiddens = []
    for layer in tm.layers:
        mask = ssm_mask if layer.is_linear else fa_mask
        h = layer(h, mask=mask, cache=None)
        hiddens.append(h)
    out = tm.norm(h)
    lm = model.language_model
    if lm.args.tie_word_embeddings:
        logits = tm.embed_tokens.as_linear(out)
    else:
        logits = lm.lm_head(out)
    return logits, hiddens


# ----------------------------------------------------------------------------
# Plasticidad estructural: canales nuevos en el FFN (Ec. 6 del paper)
#   F'(x) = F(x) + Ds[(Us x) ⊙ SiLU(Gs x)]
# ----------------------------------------------------------------------------
class StructuralMLP(nn.Module):
    def __init__(self, base: nn.Module, k: int, init_from_parent: bool = True):
        super().__init__()
        self.base = base
        dh = base.gate_proj.weight.shape[1]
        dff = base.gate_proj.weight.shape[0]
        self.k = k
        if init_from_parent:
            # Inicializa gate/up muestreando canales del FFN padre (Appx. B.3 del paper)
            idx = mx.random.permutation(dff)[:k]
            g = base.gate_proj.weight[idx]
            u = base.up_proj.weight[idx]
        else:
            scale = 1 / math.sqrt(dh)
            g = mx.random.uniform(-scale, scale, (k, dh))
            u = mx.random.uniform(-scale, scale, (k, dh))
        self.gate_s = g.astype(base.gate_proj.weight.dtype)
        self.up_s = u.astype(base.up_proj.weight.dtype)
        # down inicializada a cero: la red empieza siendo idéntica al modelo base
        self.down_s = mx.zeros((dh, k), dtype=base.down_proj.weight.dtype)

    def __call__(self, x):
        y = self.base(x)
        z = nn.silu(x @ self.gate_s.T) * (x @ self.up_s.T)
        return y + z @ self.down_s.T

    def fold(self) -> nn.Module:
        """Pliega los canales nuevos en un FFN más ancho (d_ff + k)."""
        b = self.base
        dh = b.gate_proj.weight.shape[1]
        dff = b.gate_proj.weight.shape[0]
        new = type(b)(dh, dff + self.k)
        new.gate_proj.weight = mx.concatenate([b.gate_proj.weight, self.gate_s], axis=0)
        new.up_proj.weight = mx.concatenate([b.up_proj.weight, self.up_s], axis=0)
        new.down_proj.weight = mx.concatenate([b.down_proj.weight, self.down_s], axis=1)
        return new


FFN_KEYS = ("gate_proj", "up_proj", "down_proj")


def _iter_layers(model):
    return text_model(model).layers


def apply_synaptic(model, r: int, alpha: float | None = None, dropout: float = 0.0):
    """LoRA solo en las tres proyecciones del FFN (Ec. 5). alpha = 2r por defecto."""
    alpha = 2 * r if alpha is None else alpha
    scale = alpha / r
    model.freeze()
    for layer in _iter_layers(model):
        mlp = layer.mlp
        for key in FFN_KEYS:
            lin = getattr(mlp, key)
            lora = LoRALinear.from_base(lin, r=r, dropout=dropout, scale=scale)
            setattr(mlp, key, lora)
    return model


def apply_lora_all(model, r: int, alpha: float | None = None, dropout: float = 0.0):
    """LoRA en atención + mezcla lineal + FFN (línea base 'LoRA-All')."""
    alpha = 2 * r if alpha is None else alpha
    scale = alpha / r
    model.freeze()
    for layer in _iter_layers(model):
        targets = []
        if layer.is_linear:
            targets += [(layer.linear_attn, k) for k in ("in_proj_qkv", "in_proj_z", "out_proj")]
        else:
            targets += [(layer.self_attn, k) for k in ("q_proj", "k_proj", "v_proj", "o_proj")]
        targets += [(layer.mlp, k) for k in FFN_KEYS]
        for mod, key in targets:
            lin = getattr(mod, key)
            setattr(mod, key, LoRALinear.from_base(lin, r=r, dropout=dropout, scale=scale))
    return model


def apply_structural(model, k: int):
    """Expansión estructural del FFN de todas las capas con k canales nuevos."""
    model.freeze()
    for layer in _iter_layers(model):
        layer.mlp = StructuralMLP(layer.mlp, k)
    # solo los parámetros nuevos son entrenables
    for layer in _iter_layers(model):
        layer.mlp.unfreeze(keys=["gate_s", "up_s", "down_s"], strict=False)
    return model


def fold_adapters(model):
    """Pliega LoRA (fuse) y canales estructurales (fold) en los pesos base."""
    for layer in _iter_layers(model):
        if isinstance(layer.mlp, StructuralMLP):
            layer.mlp = layer.mlp.fold()
        for mod in (layer.mlp, getattr(layer, "self_attn", None), getattr(layer, "linear_attn", None)):
            if mod is None:
                continue
            for name, child in list(mod.children().items()):
                if isinstance(child, LoRALinear):
                    setattr(mod, name, child.fuse())
    model.unfreeze()
    return model


def count_trainable(model) -> int:
    return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))


def save_adapter(model, out_dir: Path, config: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out_dir / "adapter.safetensors"), flat)
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False))


def apply_variant(model, variant: str, **kw):
    if variant == "synaptic":
        return apply_synaptic(model, r=kw["rank"])
    if variant == "structural":
        return apply_structural(model, k=kw["rank"])
    if variant == "lora_all":
        return apply_lora_all(model, r=kw["rank"])
    if variant == "none":
        return model
    raise ValueError(variant)


def load_adapter(model, adapter_dir: Path):
    """Reconstruye la estructura del adaptador y carga sus pesos.
    Soporta una lista de adaptadores secuenciales (se pliegan uno a uno)."""
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    apply_variant(model, cfg["variant"], rank=cfg["rank"])
    weights = mx.load(str(adapter_dir / "adapter.safetensors"))
    model.load_weights(list(weights.items()), strict=False)
    return model, cfg


def load_adapter_chain(model, adapter_dirs: list[Path]):
    """Aplica varios adaptadores en orden plegando cada uno (actualizaciones secuenciales)."""
    cfg = None
    for i, d in enumerate(adapter_dirs):
        model, cfg = load_adapter(model, d)
        if i < len(adapter_dirs) - 1:
            fold_adapters(model)
    return model, cfg


# ----------------------------------------------------------------------------
# Utilidades de texto / evaluación
# ----------------------------------------------------------------------------
NUM_RE = re.compile(r"(?<![\d,.])(\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,](\d{1,2}))?(?!\d)")


def extract_numbers(text: str) -> list[float]:
    """Extrae números (formato español o internacional) de un texto."""
    out = []
    for m in NUM_RE.finditer(text):
        whole = re.sub(r"[.,]", "", m.group(1))
        dec = m.group(2) or "0"
        try:
            out.append(float(f"{whole}.{dec}"))
        except ValueError:
            pass
    return out


def norm_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[áàä]", "a", t)
    t = re.sub(r"[éèë]", "e", t)
    t = re.sub(r"[íìï]", "i", t)
    t = re.sub(r"[óòö]", "o", t)
    t = re.sub(r"[úùü]", "u", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def read_jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p: Path, rows: Iterable[dict]):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
