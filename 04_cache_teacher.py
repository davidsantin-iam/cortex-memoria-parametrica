"""
Paso 4: objetivos del profesor (context-to-memory distillation, §3.3 del paper).

Para cada ejemplo de ensayo (C, q, y) el profesor es el modelo congelado M0 viendo
[sistema, C, q, y]; el estudiante verá [sistema, q, y]. Guardamos, en las posiciones
alineadas que predicen los tokens de la respuesta (y + <|im_end|>):
  - los K logits más altos del profesor (índices + valores)        -> KL truncada
  - los estados ocultos post-FFN de TODAS las capas (fp16)           -> MSE de representación
Para el banco genérico, profesor y estudiante reciben la misma entrada (sin contexto).

Los estados se calculan con la misma ruta numérica que usará el estudiante
(modo entrenamiento, regla delta por bloques) para que el objetivo no arrastre ruido bf16.

Salida: cache/<rid>.npz  y  cache/<gid>.npz
"""
from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import numpy as np

import fast_gdn
from common import DATA, ROOT, SYSTEM_PLAIN, SYSTEM_TEACHER, answer_ids, chat_prompt_ids, forward_with_hidden, load_model, read_jsonl

CACHE = ROOT / "cache"
CACHE.mkdir(exist_ok=True)


def teacher_targets(model, ids: list[int], n_sup: int, topk: int):
    """Logits top-K y estados ocultos en las últimas n_sup posiciones de predicción."""
    x = mx.array([ids])
    logits, hiddens = forward_with_hidden(model, x)
    S = len(ids)
    pos = mx.arange(S - 1 - n_sup, S - 1)  # posiciones que predicen los tokens de la respuesta
    lg = logits[0, pos].astype(mx.float32)  # [P, V]
    idx = mx.argpartition(-lg, kth=topk - 1, axis=-1)[:, :topk]
    val = mx.take_along_axis(lg, idx, axis=-1)
    hid = mx.stack([h[0, pos] for h in hiddens]).astype(mx.float16)  # [L, P, dh]
    mx.eval(idx, val, hid)
    return np.array(idx, dtype=np.int32), np.array(val, dtype=np.float16), np.array(hid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=1024)
    ap.add_argument("--max-answer-tokens", type=int, default=160)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--rehearsal", default="rehearsal.jsonl")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--contexts", default="contexts.json")
    args = ap.parse_args()
    global CACHE
    CACHE = ROOT / args.cache_dir
    CACHE.mkdir(exist_ok=True)

    model, tok = load_model()
    fast_gdn.patch_mlx_lm(True)
    model.train()  # misma ruta numérica (chunked) que en entrenamiento; sin gradientes
    contexts = {c["ctx_id"]: c["text"] for c in json.loads((DATA / args.contexts).read_text())}

    jobs = []
    for r in read_jsonl(DATA / args.rehearsal):
        t_user = f"{contexts[r['ctx_id']]}\n\nPregunta: {r['q']}"
        jobs.append((r["rid"], chat_prompt_ids(tok, t_user, SYSTEM_TEACHER), chat_prompt_ids(tok, r["q"], SYSTEM_PLAIN), r["y"]))
    for g in read_jsonl(DATA / "generic_answers.jsonl"):
        p = chat_prompt_ids(tok, g["q"], SYSTEM_PLAIN)
        jobs.append((g["gid"], p, p, g["y"]))

    t0, n_done, n_tok = time.time(), 0, 0
    for i, (rid, t_prompt, s_prompt, y) in enumerate(jobs):
        out = CACHE / f"{rid}.npz"
        if out.exists() and not args.force:
            continue
        y_ids = answer_ids(tok, y)[: args.max_answer_tokens]
        idx, val, hid = teacher_targets(model, t_prompt + y_ids, len(y_ids), args.topk)
        s_ids = np.array(s_prompt + y_ids, dtype=np.int32)
        sup_pos = np.arange(len(s_prompt) - 1, len(s_ids) - 1, dtype=np.int32)
        np.savez(out, ids=s_ids, sup_pos=sup_pos, t_idx=idx, t_val=val, t_hid=hid)
        n_done += 1
        n_tok += len(t_prompt) + len(y_ids)
        if n_done % 50 == 0:
            el = time.time() - t0
            print(f"{i+1}/{len(jobs)}  {n_tok/el:.0f} tok/s  {el/60:.1f} min")
    size = sum(f.stat().st_size for f in CACHE.glob("*.npz")) / 1e9
    print(f"caché: {len(list(CACHE.glob('*.npz')))} ficheros, {size:.1f} GB, {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
