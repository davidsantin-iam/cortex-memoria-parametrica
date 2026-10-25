"""
Paso 3: evaluación de una condición (modelo base, base+contexto "oráculo", o modelo
consolidado con adaptadores) sobre:
  - Q+ objetivo: preguntas sobre el catálogo (eval_target.jsonl + eval_target_para.jsonl)
  - Q- retención: MMMLU-ES, GSM8K (eval_retain.jsonl) y perplejidad en Wikipedia-es (eval_ppl.jsonl)

Todo el "juez" es determinista (extracción de números / categorías), sin LLM juez.

Uso:
  python 03_eval.py --name vanilla
  python 03_eval.py --name context --context           # Perf(M, Q+, C): contexto correcto en el prompt
  python 03_eval.py --name synaptic_r16 --adapter checkpoints/synaptic_r16
  python 03_eval.py --name seq_step3 --adapter ckpt/seq_1,ckpt/seq_2,ckpt/seq_3
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import batch_generate
from mlx_lm.sample_utils import make_sampler

from common import (
    DATA,
    RESULTS,
    SYSTEM_PLAIN,
    chat_prompt_ids,
    extract_numbers,
    fold_adapters,
    load_adapter_chain,
    load_model,
    norm_text,
    read_jsonl,
)

TARGET_TYPES = ["cuantia", "reduccion", "puntos", "gravedad", "precepto", "comparacion", "cuantia_para", "gravedad_para", "puntos_para"]
MAX_TOKENS = {"gsm8k": 420, "mmlu_es": 8}
DEFAULT_MAX_TOKENS = 48


# ----------------------------------------------------------------------------
# Jueces deterministas
# ----------------------------------------------------------------------------
def grade(item: dict, answer: str) -> bool:
    t, gold = item["type"], item["gold"]
    a = answer.strip()
    n = norm_text(a)
    if t in ("cuantia", "reduccion", "cuantia_para"):
        nums = extract_numbers(a)
        return bool(nums) and abs(nums[0] - float(gold)) < 0.01
    if t in ("puntos", "puntos_para"):
        nums = [x for x in extract_numbers(a) if x == int(x) and 0 <= x <= 12]
        if not nums:
            return int(gold) == 0 and bool(re.search(r"\b(no|ningun|ninguno|cero|sin)\b", n))
        return int(nums[0]) == int(gold)
    if t in ("gravedad", "gravedad_para"):
        if "muy grave" in n:
            pred = "Muy Grave"
        elif "grave" in n:
            pred = "Grave"
        elif "leve" in n:
            pred = "Leve"
        else:
            return False
        return pred == gold
    if t == "precepto":
        g = re.sub(r"\s+", "", str(gold)).lower()
        return g in re.sub(r"\s+", "", a).lower()
    if t == "comparacion":
        m = re.search(r"(?<![a-z])([ab])(?![a-z])", n)
        return bool(m) and m.group(1) == gold
    if t == "mmlu_es":
        m = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", a)
        return bool(m) and m.group(1) == gold
    if t == "gsm8k":
        nums = extract_numbers(a.replace("$", ""))
        return bool(nums) and abs(nums[-1] - float(gold)) < 1e-6
    raise ValueError(t)


def perplexity(model, tok, paragraphs: list[dict]) -> float:
    """Perplejidad media (token a token) sobre texto crudo en español."""
    tot_nll, tot_tok = 0.0, 0
    for p in paragraphs:
        ids = mx.array([tok.encode(p["text"])[:400]])
        logits = model(ids)[0, :-1].astype(mx.float32)
        tgt = ids[0, 1:]
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        nll = -mx.take_along_axis(lp, tgt[:, None], axis=-1).sum()
        mx.eval(nll)
        tot_nll += float(nll)
        tot_tok += tgt.shape[0]
    return float(mx.exp(mx.array(tot_nll / tot_tok)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--adapter", default="", help="dir(es) de adaptadores separados por coma (se aplican y pliegan en orden)")
    ap.add_argument("--context", action="store_true", help="incluir el contexto correcto en el prompt (oráculo)")
    ap.add_argument("--skip-retain", action="store_true")
    ap.add_argument("--skip-target", action="store_true")
    ap.add_argument("--only-ctx", default="", help="evaluar solo preguntas de estos ctx_id (coma)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    model, tok = load_model()
    adapter_info = None
    if args.adapter:
        dirs = [Path(d) for d in args.adapter.split(",")]
        model, adapter_info = load_adapter_chain(model, dirs)
        fold_adapters(model)
        mx.eval(model.parameters())
        print("adaptadores plegados:", [d.name for d in dirs])
    model.eval()

    contexts = {c["ctx_id"]: c for c in json.loads((DATA / "contexts.json").read_text())}
    greedy = make_sampler(temp=0.0)
    results = dict(name=args.name, adapter=args.adapter, context=args.context, items=[], metrics={})

    # ---------------- Q+ ----------------
    if not args.skip_target:
        target = read_jsonl(DATA / "eval_target.jsonl")
        para_path = DATA / "eval_target_para.jsonl"
        if para_path.exists():
            target += read_jsonl(para_path)
        if args.only_ctx:
            keep = set(args.only_ctx.split(","))
            target = [i for i in target if i["ctx_id"] in keep or i.get("ctx_id_b") in keep]
        if args.limit:
            target = target[: args.limit]
        prompts, lens = [], []
        for it in target:
            user = it["question"]
            if args.context:
                ctx = contexts[it["ctx_id"]]["text"]
                if it.get("ctx_id_b") and it["ctx_id_b"] != it["ctx_id"]:
                    ctx += "\n\n" + contexts[it["ctx_id_b"]]["text"]
                user = f"{ctx}\n\nPregunta: {user}"
            prompts.append(chat_prompt_ids(tok, user, system=SYSTEM_PLAIN))
            lens.append(MAX_TOKENS.get(it["type"], DEFAULT_MAX_TOKENS))
        print(f"Q+: {len(prompts)} preguntas")
        resp = batch_generate(model, tok, prompts, max_tokens=lens, sampler=greedy, verbose=True, prefill_batch_size=4 if args.context else 8)
        for it, ans in zip(target, resp.texts):
            ok = grade(it, ans)
            results["items"].append(dict(qid=it["qid"], type=it["type"], uid=it.get("uid"), ctx_id=it.get("ctx_id"), gold=it["gold"], answer=ans.strip(), correct=ok))

    # ---------------- Q- ----------------
    if not args.skip_retain:
        retain = read_jsonl(DATA / "eval_retain.jsonl")
        if args.limit:
            retain = retain[: args.limit]
        prompts = [chat_prompt_ids(tok, it["question"], system=SYSTEM_PLAIN) for it in retain]
        lens = [MAX_TOKENS.get(it["type"], DEFAULT_MAX_TOKENS) for it in retain]
        print(f"Q-: {len(prompts)} preguntas")
        resp = batch_generate(model, tok, prompts, max_tokens=lens, sampler=greedy, verbose=True)
        for it, ans in zip(retain, resp.texts):
            results["items"].append(dict(qid=it["qid"], type=it["type"], gold=it["gold"], answer=ans.strip(), correct=grade(it, ans)))
        ppl = perplexity(model, tok, read_jsonl(DATA / "eval_ppl.jsonl"))
        results["metrics"]["ppl_wiki_es"] = ppl
        print(f"perplejidad Wikipedia-es: {ppl:.3f}")

    # ---------------- agregados ----------------
    by_type = {}
    for it in results["items"]:
        by_type.setdefault(it["type"], []).append(it["correct"])
    for t, v in by_type.items():
        results["metrics"][t] = dict(acc=sum(v) / len(v), n=len(v))
    tgt = [results["metrics"][t]["acc"] for t in TARGET_TYPES if t in results["metrics"]]
    if tgt:
        results["metrics"]["target_score"] = sum(tgt) / len(tgt)
    ret = [results["metrics"][t]["acc"] for t in ("mmlu_es", "gsm8k") if t in results["metrics"]]
    if ret:
        results["metrics"]["retain_score"] = sum(ret) / len(ret)
    results["metrics"]["seconds"] = time.time() - t0
    out = RESULTS / f"eval_{args.name}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print(json.dumps({k: (round(v["acc"], 3) if isinstance(v, dict) else round(v, 3)) for k, v in results["metrics"].items()}, indent=1))
    print("guardado en", out)


if __name__ == "__main__":
    main()
