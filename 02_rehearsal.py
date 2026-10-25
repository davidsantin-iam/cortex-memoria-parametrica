"""
Paso 2: ensayo prospectivo (Prospective Rehearsal, §3.1 del paper).

El propio modelo pre-actualización M0, viendo cada contexto C_i, genera:
  a) preguntas que anticipan usos futuros de esa información (consultas prospectivas), y
  b) sus respuestas condicionadas al contexto (el "profesor").
Además responde, SIN contexto, a un banco de consultas genéricas que servirá para
preservar las capacidades no relacionadas (retención).

No se usa ningún generador externo ni las preguntas de evaluación.

Salidas: data/rehearsal.jsonl, data/generic_answers.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re

import mlx.core as mx
from mlx_lm import batch_generate
from mlx_lm.sample_utils import make_sampler

from common import DATA, SYSTEM_TEACHER, SYSTEM_PLAIN, chat_prompt_ids, load_model, read_jsonl, write_jsonl

QUERY_INSTRUCTION = (
    "{ctx}\n\n"
    "Tarea: escribe {k} preguntas distintas y naturales que un ciudadano, un conductor o un agente "
    "podrían hacer en el futuro sobre la infracción «{desc}». Varía el enfoque: el importe de la multa, "
    "la reducción por pronto pago, los puntos que se retiran, la gravedad, el precepto aplicable o una "
    "situación concreta en la que alguien comete esa infracción. Cada pregunta debe entenderse por sí "
    "sola, sin referirse a ningún texto, catálogo o contexto. Devuelve solo las preguntas, una por línea, "
    "numeradas del 1 al {k}."
)

FACET_INSTRUCTION = (
    "{ctx}\n\n"
    "Tarea: escribe 6 preguntas distintas y naturales que un ciudadano, un conductor o un agente podrían "
    "hacer en el futuro sobre la infracción «{desc}», una por cada aspecto, en este orden: "
    "1) el importe de la multa; 2) cuánto se paga con la reducción por pronto pago; 3) cuántos puntos del "
    "permiso se retiran; 4) si es leve, grave o muy grave; 5) qué precepto (artículo y norma) la tipifica; "
    "6) una situación concreta contada en primera persona por alguien que acaba de cometerla y quiere saber "
    "qué le va a pasar. Cada pregunta debe entenderse por sí sola, sin referirse a ningún texto, catálogo o "
    "contexto, y debe describir la infracción con sus propias palabras. Devuelve solo las preguntas, una por "
    "línea, numeradas del 1 al 6."
)

BAD_WORDS = re.compile(
    r"catálogo|catalogo|contexto|texto (anterior|proporcionado|facilitado|que)|según el (texto|documento|extracto)|"
    r"extracto|en la lista|de la lista|arriba|anteriormente|proporcionad|se te ha|que me has|mencionad",
    re.I,
)


def parse_questions(txt: str, k: int) -> list[str]:
    qs = []
    for line in txt.splitlines():
        m = re.match(r"\s*(?:\d+[.)]|[-*•])\s*(.+)", line.strip())
        if not m:
            continue
        q = m.group(1).strip().strip('"«»')
        if len(q) < 15 or "?" not in q or BAD_WORDS.search(q):
            continue
        qs.append(q)
    # dedup conservando orden
    seen, out = set(), []
    for q in qs:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            out.append(q)
    return out[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="preguntas por infracción")
    ap.add_argument("--limit-contexts", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--facets", action="store_true", help="una pregunta por faceta (6 por infracción)")
    ap.add_argument("--out", default="rehearsal.jsonl")
    ap.add_argument("--skip-generic", action="store_true")
    args = ap.parse_args()
    if args.facets:
        args.k = 6
    random.seed(args.seed)
    mx.random.seed(args.seed)

    model, tok = load_model()
    sanciones = {s["uid"]: s for s in read_jsonl(DATA / "sanciones.jsonl")}
    contexts = json.loads((DATA / "contexts.json").read_text())
    if args.limit_contexts:
        contexts = contexts[: args.limit_contexts]

    # --- a) consultas prospectivas -------------------------------------------------
    jobs, prompts = [], []
    for c in contexts:
        for uid in c["uids"]:
            s = sanciones[uid]
            user = (FACET_INSTRUCTION if args.facets else QUERY_INSTRUCTION).format(ctx=c["text"], k=args.k, desc=s["descripcion"])
            jobs.append((c["ctx_id"], uid))
            prompts.append(chat_prompt_ids(tok, user, system=SYSTEM_PLAIN))
    print(f"generando preguntas para {len(prompts)} infracciones en {len(contexts)} contextos ...")
    sampler = make_sampler(temp=0.7, top_p=0.95)
    resp = batch_generate(model, tok, prompts, max_tokens=400 if args.facets else 220, sampler=sampler, verbose=True, prefill_batch_size=2, completion_batch_size=16)
    qa_jobs = []
    n_short = 0
    for (ctx_id, uid), txt in zip(jobs, resp.texts):
        qs = parse_questions(txt, args.k)
        if len(qs) < args.k:
            n_short += 1
        for q in qs:
            qa_jobs.append(dict(ctx_id=ctx_id, uid=uid, q=q))
    print(f"preguntas válidas: {len(qa_jobs)} ({n_short} infracciones con menos de {args.k})")

    # --- b) respuestas del profesor (con contexto) ---------------------------------
    ctx_text = {c["ctx_id"]: c["text"] for c in contexts}
    greedy = make_sampler(temp=0.0)

    def teacher_prompt(j):
        return chat_prompt_ids(tok, f"{ctx_text[j['ctx_id']]}\n\nPregunta: {j['q']}", system=SYSTEM_TEACHER)

    print("generando respuestas del profesor ...")
    resp = batch_generate(model, tok, [teacher_prompt(j) for j in qa_jobs], max_tokens=140, sampler=greedy, verbose=True, prefill_batch_size=2, completion_batch_size=16)
    rehearsal, retry = [], []
    for j, y in zip(qa_jobs, resp.texts):
        y = y.strip()
        if not y or BAD_WORDS.search(y):
            retry.append(j)
        else:
            rehearsal.append(dict(j, y=y))
    if retry:
        print(f"reintentando {len(retry)} respuestas que aludían al contexto ...")
        resp = batch_generate(model, tok, [teacher_prompt(j) for j in retry], max_tokens=140, sampler=make_sampler(temp=0.7, top_p=0.95), verbose=True, prefill_batch_size=2, completion_batch_size=16)
        kept = 0
        for j, y in zip(retry, resp.texts):
            y = y.strip()
            if y and not BAD_WORDS.search(y):
                rehearsal.append(dict(j, y=y))
                kept += 1
        print(f"  recuperadas {kept}/{len(retry)}")
    for i, r in enumerate(rehearsal):
        r["rid"] = f"E{i:05d}"
    write_jsonl(DATA / args.out, rehearsal)
    print(f"ejemplos de ensayo: {len(rehearsal)}")
    for r in random.sample(rehearsal, min(5, len(rehearsal))):
        print(" Q:", r["q"], "\n A:", r["y"][:200].replace("\n", " "), "\n")

    # --- c) banco genérico respondido por M0 sin contexto --------------------------
    if args.skip_generic:
        print("listo (sin regenerar genéricas)")
        return
    generic = read_jsonl(DATA / "generic_queries.jsonl")
    print(f"respondiendo {len(generic)} consultas genéricas sin contexto ...")
    resp = batch_generate(model, tok, [chat_prompt_ids(tok, g["question"], system=SYSTEM_PLAIN) for g in generic], max_tokens=200, sampler=greedy, verbose=True)
    write_jsonl(DATA / "generic_answers.jsonl", [dict(gid=g["gid"], q=g["question"], y=y.strip()) for g, y in zip(generic, resp.texts)])
    print("listo")


if __name__ == "__main__":
    main()
