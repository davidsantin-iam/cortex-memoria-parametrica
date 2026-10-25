"""
Paso 8: profesor externo para el ensayo (variante "Outside" del paper, Appx. B.3.7).

Las preguntas siguen siendo las que generó el propio Qwen3.5-2B (ensayo por facetas,
data/rehearsal_v2.jsonl), pero las RESPUESTAS las escribe un modelo mucho mayor
(Qwen 27B en LM Studio) viendo el mismo bloque del catálogo. Después cada respuesta se
filtra contra el XML: si contradice algún dato de la infracción (importe, reducción,
puntos, gravedad o precepto) se descarta. Las respuestas que sobreviven se "fuerzan"
luego a través del 2B con contexto (04_cache_teacher.py) para obtener sus logits y
estados, como en el paper.

Salida: data/rehearsal_v3.jsonl  (reanudable: va guardando por infracción)
"""
from __future__ import annotations

import collections
import json
import os
import re
import time
import urllib.request

from common import DATA, extract_numbers, norm_text, read_jsonl, write_jsonl

LMS_URL = "http://127.0.0.1:1234/v1/chat/completions"
LMS_MODEL = os.environ.get("LMS_MODEL", "qwen3.6-27b")  # identificador del modelo cargado en LM Studio

SYSTEM = (
    "Eres un experto en la normativa de tráfico y en el catálogo de infracciones y sanciones del "
    "Ayuntamiento de Madrid. Respondes en español, de forma directa, en dos o tres frases, con los "
    "datos exactos del catálogo (importes en euros, puntos, calificación y precepto tal y como figuran). "
    "Nunca mencionas que dispones de un texto, catálogo o contexto: respondes como si lo supieras de memoria."
)
PROMPT = (
    "{ctx}\n\n"
    "La infracción sobre la que preguntan es: «{desc}».\n\n"
    "Responde a cada una de estas preguntas por separado. Devuelve exactamente {n} respuestas con el "
    "formato «<número>. <respuesta>», una por línea, en el mismo orden, y nada más.\n\n{qs}"
)
BAD_WORDS = re.compile(r"catálogo|catalogo|contexto|texto (anterior|proporcionado|facilitado)|según el (texto|documento|extracto)|extracto", re.I)
# preguntas vagas que no describen la infracción (el alumno, sin contexto, no podría saber de cuál se habla)
VAGUE_RE = re.compile(r"\b(esta|esa|dicha|la presente|la mencionada|la citada) (infracci[oó]n|sanci[oó]n|multa|conducta)\b|\bese caso\b|\beste caso\b", re.I)
CODE_RE = re.compile(r"\b\d{2,3}\.[0-9A-Z.\- ]{0,10}\b(LSV|VEH|OMS|CON|SOA|RGC|CIR)\b")


def call_lms(user: str, max_tokens: int = 4000) -> str:
    body = dict(model=LMS_MODEL, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], temperature=0.3, max_tokens=max_tokens)
    req = urllib.request.Request(LMS_URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.load(r)["choices"][0]["message"].get("content") or ""


def parse_answers(txt: str, n: int) -> dict[int, str]:
    out = {}
    for line in txt.splitlines():
        m = re.match(r"\s*(\d+)[.)]\s*(.+)", line.strip())
        if m and 1 <= int(m.group(1)) <= n:
            out[int(m.group(1))] = m.group(2).strip()
    return out


def consistent(ans: str, s: dict) -> tuple[bool, str]:
    """Filtro contra el XML: rechaza respuestas que contradicen los datos de la infracción."""
    if BAD_WORDS.search(ans):
        return False, "alude al contexto"
    n = norm_text(ans)
    # importes en euros
    eur = [v for v in extract_numbers(ans) if re.search(rf"{re.escape(str(int(v)) if v.is_integer() else str(v))}[,.]?\d*\s*(€|euros?)", ans)]
    ok_eur = {s["cuantia"], s["reduccion"]}
    if any(v not in ok_eur for v in eur):
        return False, f"importe {eur} ≠ {ok_eur}"
    # puntos
    m = re.search(r"(\d+)\s*puntos?", n)
    if m and int(m.group(1)) != s["puntos"]:
        return False, f"puntos {m.group(1)} ≠ {s['puntos']}"
    if s["puntos"] > 0 and re.search(r"\b(no (se )?retir|sin (perdida|retirada) de puntos|cero puntos|0 puntos|ningun punto)", n):
        return False, "dice que no retira puntos"
    # gravedad
    g = "muy grave" if "muy grave" in n else ("grave" if re.search(r"\bgrave\b", n) else ("leve" if re.search(r"\bleve\b", n) else None))
    if g and g != s["gravedad"].lower():
        return False, f"gravedad {g} ≠ {s['gravedad']}"
    # precepto
    codes = {re.sub(r"\s+", "", c.group(0)).upper() for c in CODE_RE.finditer(ans)}
    gold = re.sub(r"\s+", "", s["precepto"]).upper()
    if codes and gold not in codes:
        return False, f"precepto {codes} ≠ {gold}"
    return True, ""


def main():
    sanciones = {s["uid"]: s for s in read_jsonl(DATA / "sanciones.jsonl")}
    contexts = {c["ctx_id"]: c["text"] for c in json.loads((DATA / "contexts.json").read_text())}
    by_uid = collections.defaultdict(list)
    for r in read_jsonl(DATA / "rehearsal_v2.jsonl"):
        by_uid[r["uid"]].append(r)
    out_path = DATA / "rehearsal_v3.jsonl"
    done = read_jsonl(out_path) if out_path.exists() else []
    done_uids = {r["uid"] for r in done}
    stats = collections.Counter()
    t0 = time.time()
    pending = [u for u in by_uid if u not in done_uids]
    print(f"{len(done_uids)} infracciones ya respondidas, {len(pending)} pendientes")
    for i, uid in enumerate(pending):
        s = sanciones[uid]
        qs = []
        for q in by_uid[uid]:
            if VAGUE_RE.search(q["q"]) or len(q["q"].split()) < 8:
                stats["pregunta vaga descartada"] += 1
            else:
                qs.append(q)
        if not qs:
            continue
        user = PROMPT.format(ctx=contexts[qs[0]["ctx_id"]], desc=s["descripcion"], n=len(qs), qs="\n".join(f"{j+1}. {q['q']}" for j, q in enumerate(qs)))
        got = {}
        for attempt in range(2):
            try:
                got = parse_answers(call_lms(user), len(qs))
            except Exception as e:  # noqa: BLE001
                print("  error LM Studio:", str(e)[:100])
                time.sleep(5)
            if len(got) >= len(qs) - 1:
                break
        for j, q in enumerate(qs):
            if j + 1 not in got:
                stats["sin respuesta"] += 1
                continue
            ok, why = consistent(got[j + 1], s)
            if ok:
                done.append(dict(rid=f"X{len(done):05d}", ctx_id=q["ctx_id"], uid=uid, q=q["q"], y=got[j + 1]))
                stats["ok"] += 1
            else:
                stats["descartada: " + why.split(" ")[0]] += 1
        write_jsonl(out_path, done)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"{i+1}/{len(pending)} infracciones, {len(done)} respuestas válidas, {el/60:.1f} min ({el/(i+1):.1f} s/infracción)", dict(stats), flush=True)
    print("FIN:", len(done), "ejemplos", dict(stats))


if __name__ == "__main__":
    main()
