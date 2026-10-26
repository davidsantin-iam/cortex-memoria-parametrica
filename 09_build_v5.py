"""
Paso 9: ensayo v5 = mucho volumen de preguntas propias + filtro contra la fuente de verdad
+ articulado explícito.

  python 09_build_v5.py contexts   -> data/contexts_v5.json
      Mismos 27 bloques, pero cada precepto se acompaña de su lectura en lenguaje natural
      ("077.C) LSV" -> "artículo 77, letra c), de la Ley de Seguridad Vial"), escrita a mano
      a partir del formato de los códigos. Solo se usa para el ensayo y el profesor; la
      evaluación sigue usando los contextos originales.

  python 09_build_v5.py merge      -> data/rehearsal_v5.jsonl, data/rehearsal_v5_self.jsonl
      Une las rondas de ensayo por facetas (data/rehearsal_v5_r*.jsonl), elimina duplicados y
      preguntas vagas, descarta las respuestas que contradicen el XML y añade, para cada
      infracción, preguntas de articulado con respuesta de plantilla (escrita a mano).
"""
from __future__ import annotations

import collections
import json
import random
import re
import sys

from common import CONTEXT_HEADER, DATA, NORM_NAMES, fmt_eur, read_jsonl, write_jsonl

random.seed(5)

NORM_FULL = {
    "LSV": "de la Ley sobre Tráfico, Circulación de Vehículos a Motor y Seguridad Vial (LSV)",
    "VEH": "del Reglamento General de Vehículos (VEH)",
    "OMS": "de la Ordenanza de Movilidad Sostenible de Madrid (OMS)",
    "CON": "del Reglamento General de Conductores (CON)",
    "SOA": "de la Ley sobre responsabilidad civil y seguro en la circulación de vehículos a motor (SOA)",
    "RGC": "del Reglamento General de Circulación (RGC)",
    "CIR": "del Reglamento General de Circulación (CIR)",
}


def articulado(precepto: str) -> str:
    """Lectura en lenguaje natural de un código de precepto del catálogo."""
    p = precepto.strip()
    m = re.match(r"^(.*?)\s+([A-Z]{3})$", p)
    if not m:
        return p
    code, norma = m.group(1).strip(), m.group(2)
    ley = NORM_FULL.get(norma, NORM_NAMES.get(norma, norma))
    # 77.C) / 076.A / 76
    m2 = re.match(r"^0*(\d+)(?:[\.\s]+([A-ZÑa-zñ]\d?)\)?)?$", code)
    if m2:
        art = f"artículo {int(m2.group(1))}"
        if m2.group(2):
            art += f", letra {m2.group(2).lower()})"
        return f"{art}, {ley}"
    m5 = re.match(r"^0*(\d+)\.(\d+)\.([A-ZÑ]{1,2})\)$", code)
    if m5:
        return f"artículo {int(m5.group(1))}, apartado {m5.group(2)}, letra {m5.group(3).lower()}), {ley}"
    m4 = re.match(r"^0*(\d+)\.(\d+)$", code)
    if m4:
        return f"artículo {int(m4.group(1))}, apartado {m4.group(2)}, {ley}"
    # 001.1.5A / 012.8-4.5A / 010. .5A / 999.9.9B
    m3 = re.match(r"^0*(\d+)\.\s*([\d\-]*)\s*\.?\s*(\d+[A-Z])$", code)
    if m3:
        art = f"artículo {int(m3.group(1))}"
        if m3.group(2):
            art += f", apartado {m3.group(2)}"
        art += f", opción {m3.group(3)}"
        return f"{art}, {ley}"
    return f"{code}, {ley}"


def render_entry_v5(s: dict) -> str:
    puntos = int(s["puntos"])
    pts = "0 (no se retiran puntos)" if puntos == 0 else str(puntos)
    return (
        f"- Infracción: {s['descripcion']}\n"
        f"  Importe: {fmt_eur(s['cuantia'])}. Con reducción por pronto pago: {fmt_eur(s['reduccion'])}. "
        f"Puntos: {pts}. Gravedad: {s['gravedad']}. Precepto: {s['precepto']} ({articulado(s['precepto'])})."
    )


def cmd_contexts():
    sanc = {s["uid"]: s for s in read_jsonl(DATA / "sanciones.jsonl")}
    ctxs = json.loads((DATA / "contexts.json").read_text())
    for c in ctxs:
        c["text"] = CONTEXT_HEADER + "\n".join(render_entry_v5(sanc[u]) for u in c["uids"])
    (DATA / "contexts_v5.json").write_text(json.dumps(ctxs, ensure_ascii=False, indent=1))
    ejemplos = random.sample(list(sanc.values()), 8)
    for s in ejemplos:
        print(f"{s['precepto']:16s} -> {articulado(s['precepto'])}")
    print("contexts_v5.json escrito")


ART_QUESTIONS = [
    "¿Qué precepto tipifica la infracción de {d}?",
    "¿En qué artículo y norma se recoge la sanción por {d}?",
    "Si me multan por {d}, ¿qué artículo de qué norma me están aplicando?",
]
ART_ANSWERS = [
    "La tipifica el precepto {p}, es decir, el {a}.",
    "Se recoge en el precepto {p}: {a}.",
    "Te aplican el precepto {p}, que corresponde al {a}.",
]


def lower_desc(d: str) -> str:
    d = d.strip().rstrip(".")
    if d.isupper():
        d = d.lower()
    return d[0].lower() + d[1:]


def cmd_merge():
    import importlib

    ot = importlib.import_module("08_outside_teacher")
    sanc = {s["uid"]: s for s in read_jsonl(DATA / "sanciones.jsonl")}
    rounds = sorted(DATA.glob("rehearsal_v5_r*.jsonl"))
    stats = collections.Counter()
    seen, self_rows = set(), []
    for f in rounds:
        for r in read_jsonl(f):
            stats["generadas"] += 1
            key = re.sub(r"\W+", " ", r["q"].lower()).strip()
            if key in seen:
                stats["duplicadas"] += 1
                continue
            seen.add(key)
            if ot.VAGUE_RE.search(r["q"]) or len(r["q"].split()) < 8:
                stats["vagas"] += 1
                continue
            ok, why = ot.consistent(r["y"], sanc[r["uid"]])
            if not ok:
                stats["descartada: " + why.split(" ")[0]] += 1
                continue
            self_rows.append(dict(ctx_id=r["ctx_id"], uid=r["uid"], q=r["q"], y=r["y"], src="propia"))
    for i, r in enumerate(self_rows):
        r["rid"] = f"V{i:05d}"
    write_jsonl(DATA / "rehearsal_v5_self.jsonl", self_rows)

    art_rows = []
    for s in sanc.values():
        d = lower_desc(s["descripcion"])
        for qt, at in zip(ART_QUESTIONS, ART_ANSWERS):
            art_rows.append(dict(ctx_id=s["ctx_id"], uid=s["uid"], q=qt.format(d=d), y=at.format(p=s["precepto"], a=articulado(s["precepto"])), src="articulado"))
    for i, r in enumerate(art_rows):
        r["rid"] = f"W{i:05d}"
    write_jsonl(DATA / "rehearsal_v5.jsonl", self_rows + art_rows)
    per_uid = collections.Counter(r["uid"] for r in self_rows)
    print(dict(stats))
    print(f"propias válidas: {len(self_rows)} ({len(per_uid)} infracciones, media {len(self_rows)/max(1,len(per_uid)):.1f} por infracción)")
    print(f"articulado (plantilla): {len(art_rows)}  -> total v5: {len(self_rows) + len(art_rows)}")
    for r in random.sample(art_rows, 2) + random.sample(self_rows, 3):
        print(" Q:", r["q"][:120], "\n A:", r["y"][:160])


if __name__ == "__main__":
    {"contexts": cmd_contexts, "merge": cmd_merge}[sys.argv[1]]()
