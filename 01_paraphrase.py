"""
Paso 1 (auxiliar): paráfrasis independientes de las descripciones de las infracciones.

Para medir *generalización* (el paper la llama "rephrasing") necesitamos preguntas
cuya redacción no haya visto el estudiante. Las genera un modelo distinto y mucho
mayor (Qwen 3.x 27B servido por LM Studio), en lotes de 10 por llamada para
amortizar el razonamiento. Salida: data/eval_target_para.jsonl
"""
from __future__ import annotations

import json
import os
import re
import urllib.request

from common import DATA, read_jsonl, write_jsonl

LMS_URL = "http://127.0.0.1:1234/v1/chat/completions"
LMS_MODEL = os.environ.get("LMS_MODEL", "qwen3.6-27b")  # identificador del modelo cargado en LM Studio
BATCH = 10

SYSTEM = (
    "Eres un redactor que reformula textos legales en lenguaje natural. Devuelves "
    "únicamente lo que se te pide, sin comentarios."
)
PROMPT = (
    "A continuación tienes {n} descripciones oficiales de infracciones de tráfico, numeradas. "
    "Reformula cada una en una sola frase, en lenguaje coloquial y en primera persona, como la "
    "contaría un conductor al que han multado (por ejemplo: «me han multado por aparcar encima de "
    "un paso de cebra»). Conserva todos los datos que distinguen la infracción (velocidades, "
    "lugares, tipo de vehículo, circunstancias); no añadas importes ni puntos; no uses mayúsculas "
    "innecesarias. Devuelve exactamente {n} líneas con el formato «<número>. <frase>» y nada más.\n\n{lista}"
)

TEMPLATES_PARA = {
    "cuantia_para": "{p}. ¿Cuál es el importe de la multa que me corresponde? Responde solo con la cantidad en euros.",
    "gravedad_para": "{p}. ¿Esa infracción se califica como leve, grave o muy grave? Responde solo con la calificación.",
    "puntos_para": "{p}. ¿Cuántos puntos del permiso de conducir me retiran por ello? Responde solo con el número.",
}


def call_lms(user: str, max_tokens: int = 3000) -> str:
    body = dict(
        model=LMS_MODEL,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        temperature=0.5,
        max_tokens=max_tokens,
    )
    req = urllib.request.Request(LMS_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return d["choices"][0]["message"].get("content") or ""


def parse_lines(txt: str, n: int) -> dict[int, str]:
    out = {}
    for line in txt.splitlines():
        m = re.match(r"\s*(\d+)[.)]\s*(.+)", line.strip())
        if m:
            i = int(m.group(1))
            if 1 <= i <= n:
                out[i] = m.group(2).strip().strip("«»\"").rstrip(".")
    return out


def main():
    sanciones = read_jsonl(DATA / "sanciones.jsonl")
    out_path = DATA / "paraphrases.jsonl"
    done = {r["uid"]: r for r in read_jsonl(out_path)} if out_path.exists() else {}
    pending = [s for s in sanciones if s["uid"] not in done]
    print(f"{len(done)} paráfrasis ya hechas, {len(pending)} pendientes")
    for i in range(0, len(pending), BATCH):
        batch = pending[i : i + BATCH]
        lista = "\n".join(f"{j+1}. {s['descripcion']}" for j, s in enumerate(batch))
        got = {}
        for attempt in range(3):
            txt = call_lms(PROMPT.format(n=len(batch), lista=lista))
            got = parse_lines(txt, len(batch))
            if len(got) == len(batch):
                break
            print(f"  lote {i//BATCH}: {len(got)}/{len(batch)} líneas, reintento {attempt+1}")
        for j, s in enumerate(batch):
            if j + 1 in got:
                done[s["uid"]] = dict(uid=s["uid"], parafrasis=got[j + 1])
        write_jsonl(out_path, done.values())
        print(f"lote {i//BATCH + 1}/{-(-len(pending)//BATCH)} listo; ejemplo: {got.get(1, '')[:100]}")

    # construir preguntas de evaluación con paráfrasis
    by_uid = {s["uid"]: s for s in sanciones}
    items = []
    for uid, r in done.items():
        s = by_uid[uid]
        p = r["parafrasis"][0].upper() + r["parafrasis"][1:]
        for t, tpl in TEMPLATES_PARA.items():
            field = t.replace("_para", "")
            if field == "puntos" and s["puntos"] == 0:
                continue  # solo las que retiran puntos (evita que "0" domine)
            items.append(dict(uid=uid, ctx_id=s["ctx_id"], type=t, question=tpl.format(p=p), gold=s[field]))
    for k, it in enumerate(items):
        it["qid"] = f"P{k:04d}"
    write_jsonl(DATA / "eval_target_para.jsonl", items)
    print(f"preguntas con paráfrasis: {len(items)}")


if __name__ == "__main__":
    main()
