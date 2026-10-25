"""
Paso 5: consolidación (entrenamiento del estudiante sin contexto).

Variantes de plasticidad (dónde escribir):
  --variant synaptic    LoRA solo en el FFN (Ec. 5)            [Cortex-Synaptic]
  --variant structural  k canales nuevos en el FFN (Ec. 6)     [Cortex-Structural]
  --variant lora_all    LoRA en atención + FFN                 [línea base]
Objetivos (cómo escribir):
  --objective kl_mse    KL top-K (T=2) + MSE de estados post-FFN  [Cortex]
  --objective kl        solo KL
  --objective ce        entropía cruzada sobre la respuesta      [SFT clásico]
Datos:
  --data rehearsal      ensayo prospectivo (+ banco genérico)   [Cortex]
  --data docs           texto crudo de los contextos con CE     [«seguir preentrenando»]

Actualizaciones secuenciales: --init-from adaptadores previos (se pliegan y congelan),
--ctx-filter para restringir a los contextos del paso, y --replay-from para re-generar
respuestas de consultas antiguas con el modelo pre-actualización (query replay, §3.3).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from mlx_lm import batch_generate
from mlx_lm.sample_utils import make_sampler

import fast_gdn
from common import (
    CKPT,
    DATA,
    LOGS,
    ROOT,
    SYSTEM_PLAIN,
    answer_ids,
    apply_variant,
    chat_prompt_ids,
    count_trainable,
    fold_adapters,
    forward_with_hidden,
    load_adapter_chain,
    load_model,
    read_jsonl,
    save_adapter,
)

CACHE = ROOT / "cache"


# ----------------------------------------------------------------------------
# Datos
# ----------------------------------------------------------------------------
def load_example(path: Path) -> dict:
    z = np.load(path)
    return dict(
        ids=mx.array(z["ids"])[None],
        sup_pos=mx.array(z["sup_pos"]),
        t_idx=mx.array(z["t_idx"]),
        t_val=mx.array(z["t_val"]).astype(mx.float32),
        t_hid=mx.array(z["t_hid"]),
    )


def docs_examples(tok, contexts: list[dict], max_len: int = 1024) -> list[dict]:
    out = []
    for c in contexts:
        ids = tok.encode(c["text"])
        for i in range(0, len(ids), max_len):
            chunk = ids[i : i + max_len]
            if len(chunk) < 32:
                continue
            out.append(dict(ids=mx.array(chunk)[None], sup_pos=mx.arange(0, len(chunk) - 1), raw=True))
    return out


def build_replay(model, tok, queries: list[str], topk: int) -> list[dict]:
    """Query replay: el modelo pre-actualización responde (sin contexto) a consultas antiguas
    y sus propias salidas se convierten en objetivos de destilación."""
    from importlib import import_module

    cache_mod = import_module("04_cache_teacher")
    model.eval()
    prompts = [chat_prompt_ids(tok, q, SYSTEM_PLAIN) for q in queries]
    resp = batch_generate(model, tok, prompts, max_tokens=140, sampler=make_sampler(temp=0.0), verbose=False)
    model.train()
    out = []
    for p, y in zip(prompts, resp.texts):
        y_ids = answer_ids(tok, y.strip())[:160]
        idx, val, hid = cache_mod.teacher_targets(model, p + y_ids, len(y_ids), topk)
        ids = np.array(p + y_ids, dtype=np.int32)
        out.append(dict(ids=mx.array(ids)[None], sup_pos=mx.arange(len(p) - 1, len(ids) - 1), t_idx=mx.array(idx), t_val=mx.array(val).astype(mx.float32), t_hid=mx.array(hid)))
    return out


# ----------------------------------------------------------------------------
# Pérdida (Ec. 8)
# ----------------------------------------------------------------------------
def make_loss(objective: str, T: float, kl_coef: float, mse_coef: float, full_vocab: bool = False):
    def loss_fn(model, ids, sup_pos, t_idx, t_val, t_hid):
        logits, hiddens = forward_with_hidden(model, ids)
        s_logits = logits[0, sup_pos].astype(mx.float32)  # [P, V]
        if objective == "ce":
            tgt = ids[0, sup_pos + 1]
            ce = nn.losses.cross_entropy(s_logits, tgt).mean()
            return ce, (ce, mx.zeros(()), mx.zeros(()))
        # KL truncada a los top-K del profesor, temperatura T, escalada por T^2
        log_ps = (s_logits / T) - mx.logsumexp(s_logits / T, axis=-1, keepdims=True)
        log_qt = (t_val / T) - mx.logsumexp(t_val / T, axis=-1, keepdims=True)
        log_ps_k = mx.take_along_axis(log_ps, t_idx, axis=-1)
        if not full_vocab:
            # Adaptación: el alumno también se renormaliza sobre el top-K del profesor.
            # Con vocabulario de 248K y T=2 el top-1024 solo cubre ~35% de la masa; la versión literal
            # del paper convierte la KL en un objetivo de "afilar" la distribución y destruye la retención.
            log_ps_k = log_ps_k - mx.logsumexp(log_ps_k, axis=-1, keepdims=True)
        kl = (mx.exp(log_qt) * (log_qt - log_ps_k)).sum(-1).mean() * (T * T)
        if objective == "kl" or mse_coef == 0:
            return kl_coef * kl, (mx.zeros(()), kl, mx.zeros(()))
        s_hid = mx.stack([h[0, sup_pos] for h in hiddens]).astype(mx.float32)  # [L, P, dh]
        mse = ((s_hid - t_hid.astype(mx.float32)) ** 2).mean()
        return kl_coef * kl + mse_coef * mse, (mx.zeros(()), kl, mse)

    return loss_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", default="synaptic", choices=["synaptic", "structural", "lora_all"])
    ap.add_argument("--rank", type=int, default=16, help="r (LoRA) o k (canales estructurales)")
    ap.add_argument("--objective", default="kl_mse", choices=["kl_mse", "kl", "ce"])
    ap.add_argument("--data", default="rehearsal", choices=["rehearsal", "docs"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--adam-eps", type=float, default=1e-8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--kl-temp", type=float, default=2.0)
    ap.add_argument("--kl-coef", type=float, default=1.0)
    ap.add_argument("--mse-coef", type=float, default=1.0)
    ap.add_argument("--kl-full-vocab", action="store_true", help="KL literal del paper (alumno normalizado sobre todo el vocabulario)")
    ap.add_argument("--generic", type=int, default=130, help="ejemplos genéricos mezclados por época (0 = ninguno)")
    ap.add_argument("--ctx-filter", default="", help="solo ejemplos de estos ctx_id (coma)")
    ap.add_argument("--init-from", default="", help="adaptadores previos a plegar (coma) - secuencial")
    ap.add_argument("--replay-from", default="", help="fichero jsonl con consultas antiguas para query replay")
    ap.add_argument("--replay-n", type=int, default=50)
    ap.add_argument("--save-epochs", default="", help="guardar checkpoints intermedios en estas épocas (coma)")
    ap.add_argument("--max-examples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--rehearsal", default="rehearsal.jsonl")
    ap.add_argument("--cache-dir", default="cache")
    args = ap.parse_args()
    global CACHE
    CACHE = ROOT / args.cache_dir

    random.seed(args.seed)
    mx.random.seed(args.seed)
    fast_gdn.patch_mlx_lm(True)
    model, tok = load_model()
    if args.init_from:
        model, _ = load_adapter_chain(model, [Path(p) for p in args.init_from.split(",")])
        fold_adapters(model)
        mx.eval(model.parameters())
        print("plegados adaptadores previos:", args.init_from)

    # ---- datos --------------------------------------------------------------------
    contexts = json.loads((DATA / "contexts.json").read_text())
    if args.data == "docs":
        if args.ctx_filter:
            keep = set(args.ctx_filter.split(","))
            contexts = [c for c in contexts if c["ctx_id"] in keep]
        base_examples = docs_examples(tok, contexts)
        generic_paths = []
    else:
        reh = read_jsonl(DATA / args.rehearsal)
        if args.ctx_filter:
            keep = set(args.ctx_filter.split(","))
            reh = [r for r in reh if r["ctx_id"] in keep]
        if args.max_examples:
            reh = reh[: args.max_examples]
        base_examples = [CACHE / f"{r['rid']}.npz" for r in reh]
        generic_paths = [CACHE / f"{g['gid']}.npz" for g in read_jsonl(DATA / "generic_answers.jsonl")]
    replay_examples = []
    if args.replay_from:
        qs = [r["q"] for r in read_jsonl(Path(args.replay_from))]
        random.shuffle(qs)
        qs = qs[: args.replay_n]
        print(f"query replay: regenerando {len(qs)} respuestas con el modelo pre-actualización ...")
        replay_examples = build_replay(model, tok, qs, topk=1024)

    # ---- modelo --------------------------------------------------------------------
    apply_variant(model, args.variant, rank=args.rank)
    model.train()
    n_train = count_trainable(model)
    print(f"variante {args.variant} (rank/k={args.rank}): {n_train/1e6:.2f}M parámetros entrenables")

    per_epoch = len(base_examples) + min(args.generic, len(generic_paths)) + len(replay_examples)
    total_updates = math.ceil(per_epoch * args.epochs / args.accum)
    warm = max(1, int(total_updates * args.warmup))
    sched = optim.join_schedules([optim.linear_schedule(0.0, args.lr, warm), optim.cosine_decay(args.lr, max(1, total_updates - warm))], [warm])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.0, eps=args.adam_eps)
    objective = "ce" if args.data == "docs" else args.objective
    loss_fn = make_loss(objective, args.kl_temp, args.kl_coef, args.mse_coef, args.kl_full_vocab)
    vg = nn.value_and_grad(model, loss_fn)

    log_path = LOGS / f"train_{args.name}.jsonl"
    log_f = open(log_path, "w")
    save_epochs = {int(e) for e in args.save_epochs.split(",") if e}
    out_dir = CKPT / args.name
    cfg = dict(vars(args), variant=args.variant, rank=args.rank, trainable=n_train, per_epoch=per_epoch, total_updates=total_updates)

    print(f"{per_epoch} ejemplos/época, {args.epochs} épocas, {total_updates} actualizaciones (accum {args.accum})")
    step, upd, t0 = 0, 0, time.time()
    acc_grads, acc_n = None, 0
    run = dict(loss=0.0, ce=0.0, kl=0.0, mse=0.0, n=0)
    for epoch in range(1, args.epochs + 1):
        order = list(base_examples) + random.sample(generic_paths, min(args.generic, len(generic_paths))) + list(replay_examples)
        random.shuffle(order)
        for ex in order:
            ex = ex if isinstance(ex, dict) else load_example(ex)
            if ex.get("raw"):  # CE sobre texto crudo: los objetivos del profesor no se usan
                dummy = (mx.zeros((1, 1), dtype=mx.int32), mx.zeros((1, 1)), mx.zeros((1, 1, 1)))
                (loss, (ce, kl, mse)), grads = vg(model, ex["ids"], ex["sup_pos"], *dummy)
            else:
                (loss, (ce, kl, mse)), grads = vg(model, ex["ids"], ex["sup_pos"], ex["t_idx"], ex["t_val"], ex["t_hid"])
            acc_grads = grads if acc_grads is None else tree_map(lambda a, b: a + b, acc_grads, grads)
            acc_n += 1
            mx.eval(loss, acc_grads)
            run["loss"] += float(loss); run["ce"] += float(ce); run["kl"] += float(kl); run["mse"] += float(mse); run["n"] += 1
            step += 1
            if acc_n == args.accum:
                grads = tree_map(lambda a: a / acc_n, acc_grads)
                grads, gnorm = optim.clip_grad_norm(grads, 1.0)
                opt.update(model, grads)
                mx.eval(model.trainable_parameters(), opt.state)
                acc_grads, acc_n = None, 0
                upd += 1
                if upd % 10 == 0:
                    n = run["n"]
                    rec = dict(epoch=epoch, step=step, update=upd, loss=run["loss"] / n, ce=run["ce"] / n, kl=run["kl"] / n, mse=run["mse"] / n, lr=float(sched(upd)), gnorm=float(gnorm), elapsed=time.time() - t0)
                    log_f.write(json.dumps(rec) + "\n"); log_f.flush()
                    if upd % 50 == 0:
                        print(f"ep {epoch} upd {upd}/{total_updates} loss {rec['loss']:.4f} (ce {rec['ce']:.3f} kl {rec['kl']:.3f} mse {rec['mse']:.3f}) lr {rec['lr']:.2e} {rec['elapsed']/60:.1f} min")
                    run = dict(loss=0.0, ce=0.0, kl=0.0, mse=0.0, n=0)
        if epoch in save_epochs and epoch < args.epochs:
            save_adapter(model, CKPT / f"{args.name}_ep{epoch}", dict(cfg, epoch=epoch))
            print(f"checkpoint época {epoch} guardado")
    if acc_n:
        grads = tree_map(lambda a: a / acc_n, acc_grads)
        grads, _ = optim.clip_grad_norm(grads, 1.0)
        opt.update(model, grads)
        mx.eval(model.trainable_parameters())
    save_adapter(model, out_dir, dict(cfg, epoch=args.epochs, minutes=(time.time() - t0) / 60))
    print(f"adaptador guardado en {out_dir}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
