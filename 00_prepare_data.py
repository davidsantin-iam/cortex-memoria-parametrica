"""
Paso 0: preparación de datos.

  1. Parsea el catálogo XML de sanciones (normativa/getSanciones.txt) -> data/sanciones.jsonl
  2. Agrupa las sanciones en contextos C_i (bloques de ~16 entradas por norma) -> data/contexts.json
  3. Construye el conjunto de evaluación objetivo Q+ (preguntas con respuesta de oro
     derivada determinísticamente del catálogo) -> data/eval_target.jsonl
  4. Descarga los conjuntos de retención Q- (MMMLU-ES, GSM8K, Wikipedia-es para perplejidad)
     -> data/eval_retain.jsonl, data/eval_ppl.jsonl
  5. Escribe el banco de consultas genéricas usado durante el entrenamiento para
     preservar capacidades -> data/generic_queries.jsonl

Las preguntas de evaluación NUNCA se usan en el entrenamiento (las consultas de
ensayo las genera el propio modelo en 02_rehearsal.py).
"""
from __future__ import annotations

import collections
import json
import random
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from common import DATA, NORMATIVA_XML, render_context, write_jsonl

random.seed(0)
NS = {"n": "http://www.tempuri.org/SancionesDS.xsd"}
CHUNK_SIZE = 16


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


# ----------------------------------------------------------------------------
# 1. Parseo
# ----------------------------------------------------------------------------
def parse_sanciones() -> list[dict]:
    root = ET.parse(NORMATIVA_XML).getroot()
    rows = []
    for e in root.findall("n:sanciones", NS):
        g = lambda k: clean(e.findtext("n:" + k, default="", namespaces=NS))
        precepto = g("precepto")
        norma = precepto.split()[-1] if precepto.split() else ""
        rows.append(
            dict(
                uid=int(g("id")),
                clave=g("clave"),
                descripcion=clean(g("descripcion")),
                cuantia=float(g("cuantia")),
                reduccion=float(g("reduccion")),
                puntos=int(float(g("puntos") or 0)),
                gravedad=g("gravedad"),
                precepto=precepto,
                norma=norma,
                radar=g("radar") == "S",
                alcohol=g("alco") == "S",
                mensaje=g("mensaje"),
            )
        )
    # descripciones duplicadas: se conservan solo si todos sus datos coinciden
    by_desc = collections.defaultdict(list)
    for r in rows:
        by_desc[r["descripcion"].lower()].append(r)
    kept, dropped = [], 0
    for desc, group in by_desc.items():
        key = lambda r: (r["cuantia"], r["reduccion"], r["puntos"], r["gravedad"], r["precepto"])
        if len({key(r) for r in group}) == 1:
            kept.append(group[0])
        else:
            dropped += len(group)
    kept = [r for r in kept if r["cuantia"] > 0 and r["norma"]]
    print(f"sanciones: {len(rows)} en el XML, {len(kept)} únicas y válidas, {dropped} descartadas por conflicto")
    return kept


# ----------------------------------------------------------------------------
# 2. Contextos
# ----------------------------------------------------------------------------
def build_contexts(sanciones: list[dict]) -> list[dict]:
    def sort_key(r):
        m = re.match(r"(\d+)", r["precepto"])
        return (r["norma"], int(m.group(1)) if m else 0, r["precepto"], r["uid"])

    sanciones.sort(key=sort_key)
    by_norma = collections.defaultdict(list)
    for r in sanciones:
        by_norma[r["norma"]].append(r)
    contexts = []
    for norma in sorted(by_norma):
        items = by_norma[norma]
        n_chunks = max(1, round(len(items) / CHUNK_SIZE))
        size = -(-len(items) // n_chunks)
        for i in range(0, len(items), size):
            chunk = items[i : i + size]
            cid = f"C{len(contexts):02d}_{norma}"
            for r in chunk:
                r["ctx_id"] = cid
            contexts.append(dict(ctx_id=cid, norma=norma, uids=[r["uid"] for r in chunk], text=render_context(chunk)))
    print(f"contextos: {len(contexts)} (tamaños: {[len(c['uids']) for c in contexts]})")
    return contexts


# ----------------------------------------------------------------------------
# 3. Evaluación objetivo Q+
# ----------------------------------------------------------------------------
TEMPLATES = {
    "cuantia": "¿Cuál es el importe de la multa por la siguiente infracción: «{d}»? Responde solo con la cantidad en euros.",
    "reduccion": "¿Cuánto se paga, aplicando la reducción por pronto pago, por la infracción «{d}»? Responde solo con la cantidad en euros.",
    "puntos": "¿Cuántos puntos del permiso de conducir se retiran por la infracción «{d}»? Responde solo con el número.",
    "gravedad": "¿Cómo se califica la infracción «{d}»: leve, grave o muy grave? Responde solo con la calificación.",
    "precepto": "¿Qué precepto (artículo y norma) tipifica la infracción «{d}»? Responde solo con el precepto, tal y como figura en el catálogo.",
}


def build_target(sanciones: list[dict]) -> list[dict]:
    items = []
    for s in sanciones:
        d = s["descripcion"]
        for t in ("cuantia", "reduccion", "gravedad", "precepto"):
            items.append(dict(uid=s["uid"], ctx_id=s["ctx_id"], type=t, question=TEMPLATES[t].format(d=d), gold=s[t]))
    # puntos: conjunto equilibrado (todas las que retiran puntos + igual número con 0)
    con = [s for s in sanciones if s["puntos"] > 0]
    sin = random.sample([s for s in sanciones if s["puntos"] == 0], len(con))
    for s in con + sin:
        items.append(dict(uid=s["uid"], ctx_id=s["ctx_id"], type="puntos", question=TEMPLATES["puntos"].format(d=s["descripcion"]), gold=s["puntos"]))
    # comparación (portabilidad): inferir cuál de dos infracciones tiene multa mayor
    pool = sanciones[:]
    n_comp, tries = 0, 0
    while n_comp < 150 and tries < 5000:
        tries += 1
        a, b = random.sample(pool, 2)
        if a["cuantia"] == b["cuantia"] or a["ctx_id"] == b["ctx_id"]:
            continue
        gold = "a" if a["cuantia"] > b["cuantia"] else "b"
        q = (
            "¿Cuál de estas dos infracciones conlleva una multa de mayor importe? "
            f"(a) «{a['descripcion']}» o (b) «{b['descripcion']}». Responde solo con la letra a o b."
        )
        items.append(dict(uid=a["uid"], uid_b=b["uid"], ctx_id=a["ctx_id"], ctx_id_b=b["ctx_id"], type="comparacion", question=q, gold=gold))
        n_comp += 1
    for i, it in enumerate(items):
        it["qid"] = f"T{i:04d}"
    print("evaluación objetivo:", collections.Counter(i["type"] for i in items))
    return items


# ----------------------------------------------------------------------------
# 4. Retención Q-
# ----------------------------------------------------------------------------
def ds_rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict]:
    url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
        dict(dataset=dataset, config=config, split=split, offset=offset, length=length)
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return [x["row"] for x in json.load(r)["rows"]]
        except Exception as e:  # noqa: BLE001
            print("  reintento", dataset, offset, e)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"no se pudo descargar {dataset}")


def build_retain() -> tuple[list[dict], list[dict]]:
    items = []
    # MMMLU en español (openai/MMMLU, ES_LA): 300 preguntas de opción múltiple
    offsets = random.sample(range(0, 13900, 100), 3)
    for off in offsets:
        for r in ds_rows("openai/MMMLU", "ES_LA", "test", off, 100):
            q = (
                f"{r['Question'].strip()}\n(A) {r['A']}\n(B) {r['B']}\n(C) {r['C']}\n(D) {r['D']}\n"
                "Responde solo con la letra de la opción correcta."
            )
            items.append(dict(type="mmlu_es", subject=r["Subject"], question=q, gold=r["Answer"].strip()))
    # GSM8K (test, 100 problemas)
    for r in ds_rows("openai/gsm8k", "main", "test", 0, 100):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        items.append(dict(type="gsm8k", question=r["question"].strip() + "\nResuelve paso a paso y termina con la respuesta numérica final.", gold=gold))
    for i, it in enumerate(items):
        it["qid"] = f"R{i:04d}"
    # Wikipedia en español para perplejidad (40 párrafos)
    ppl = []
    for off in (120000, 480000):
        for r in ds_rows("wikimedia/wikipedia", "20231101.es", "train", off, 60):
            txt = r["text"].strip()
            if len(txt) > 900:
                ppl.append(dict(title=r["title"], text=txt[:1200]))
            if len(ppl) >= 40:
                break
        if len(ppl) >= 40:
            break
    print("retención:", collections.Counter(i["type"] for i in items), "| ppl párrafos:", len(ppl))
    return items, ppl


# ----------------------------------------------------------------------------
# 5. Banco de consultas genéricas (retención durante el entrenamiento)
# ----------------------------------------------------------------------------
GENERIC = [
    # conocimiento factual
    "¿Cuál es la capital de Australia y por qué no es Sídney?",
    "Explica brevemente qué es la fotosíntesis.",
    "¿Quién escribió Cien años de soledad y en qué año se publicó?",
    "¿Qué diferencia hay entre un virus y una bacteria?",
    "Resume en tres frases la Revolución Francesa.",
    "¿Cuántos planetas tiene el sistema solar? Nómbralos en orden.",
    "¿Qué es el ADN y para qué sirve?",
    "¿Qué es la inflación y cómo afecta al poder adquisitivo?",
    "Describe el ciclo del agua.",
    "¿Qué es un agujero negro?",
    "¿Quién fue Marie Curie y por qué es importante?",
    "¿Cuáles son los tres poderes del Estado en una democracia?",
    "¿Qué es la Unión Europea y cuántos países la forman aproximadamente?",
    "Explica qué es la teoría de la evolución de Darwin.",
    "¿Qué idiomas se hablan en Suiza?",
    "¿Qué es el efecto invernadero?",
    "¿En qué consiste la tabla periódica de los elementos?",
    "¿Qué fue el Imperio Romano y cuándo cayó su parte occidental?",
    "¿Qué es una vacuna y cómo funciona?",
    "¿Qué es el PIB de un país?",
    "What is the difference between weather and climate?",
    "Who painted the Mona Lisa and where is it displayed?",
    "Explain what a prime number is with three examples.",
    "What causes the seasons on Earth?",
    "What is the role of the mitochondria in a cell?",
    # razonamiento / matemáticas
    "Si un tren sale a las 9:15 y tarda 2 horas y 50 minutos, ¿a qué hora llega?",
    "Tengo 3 cajas con 12 manzanas cada una y regalo 7. ¿Cuántas me quedan?",
    "¿Cuánto es el 15 % de 240?",
    "Un rectángulo mide 7 cm por 4 cm. ¿Cuál es su área y su perímetro?",
    "Si compro 4 libros a 12,50 € cada uno y pago con 60 €, ¿cuánto me devuelven?",
    "Resuelve: 2x + 6 = 20.",
    "¿Cuál es el siguiente número de la serie 2, 6, 18, 54, ...? Explica por qué.",
    "Un coche recorre 180 km en 2 horas. ¿Cuál es su velocidad media en km/h?",
    "¿Cuántos segundos hay en un día?",
    "Convierte 3,5 kilómetros a metros.",
    "Si hoy es miércoles, ¿qué día será dentro de 10 días?",
    "¿Es 91 un número primo? Razónalo.",
    "María es mayor que Juan y Juan es mayor que Pedro. ¿Quién es el menor?",
    "A farmer has 17 sheep and all but 9 run away. How many are left?",
    "What is 25 squared minus 24 squared?",
    "Calcula la media de 4, 8, 15, 16, 23 y 42.",
    "Si el 20 % de un número es 50, ¿cuál es el número?",
    "¿Cuántas diagonales tiene un hexágono?",
    "Simplifica la fracción 36/48.",
    "Un producto costaba 80 € y ahora cuesta 60 €. ¿Qué porcentaje ha bajado?",
    # lenguaje / escritura
    "Corrige los errores de esta frase: 'Haber si bienes mañana a la reunion.'",
    "Escribe un sinónimo y un antónimo de 'generoso'.",
    "Traduce al inglés: 'El museo cierra los lunes por la tarde.'",
    "Traduce al español: 'The meeting has been postponed until next week.'",
    "Escribe un haiku sobre el otoño.",
    "Resume en una frase: 'La biblioteca municipal amplía su horario durante el periodo de exámenes para facilitar el estudio a los universitarios.'",
    "¿Qué diferencia hay entre 'por qué', 'porque' y 'porqué'?",
    "Redacta un correo breve y formal para pedir cita en el dentista.",
    "Explica el significado del refrán 'A quien madruga, Dios le ayuda'.",
    "Escribe tres títulos posibles para un artículo sobre el reciclaje en las ciudades.",
    "Conjuga el verbo 'ir' en pretérito perfecto simple.",
    "Convierte esta frase a voz pasiva: 'El ayuntamiento inauguró el nuevo parque.'",
    "Write a short, friendly reminder message for a team meeting tomorrow at 10am.",
    "Give me a one-sentence definition of 'irony'.",
    "¿Cuál es el plural de 'régimen' y de 'carácter'?",
    "Escribe una descripción de dos frases de una ciudad costera al amanecer.",
    "Explica a un niño de ocho años qué es Internet.",
    "Enumera cinco palabras que rimen con 'canción'.",
    "¿Qué es una metáfora? Pon un ejemplo.",
    "Reescribe en tono informal: 'Le rogamos disculpe las molestias ocasionadas.'",
    # código
    "Escribe una función en Python que devuelva si una cadena es un palíndromo.",
    "¿Qué hace este código? `[x*x for x in range(5) if x % 2 == 0]`",
    "Explica la diferencia entre una lista y una tupla en Python.",
    "Escribe una consulta SQL que cuente los empleados por departamento.",
    "¿Qué es una API REST?",
    "Write a JavaScript function that returns the largest number in an array.",
    "Explain what a hash map is and when to use it.",
    "¿Qué diferencia hay entre Git y GitHub?",
    "Escribe en Python un bucle que imprima los números del 1 al 10 que sean múltiplos de 3.",
    "¿Para qué sirve el comando `grep` en Linux?",
    "Explica qué es la recursividad con un ejemplo sencillo.",
    "What does HTTP status code 404 mean?",
    "Escribe una expresión regular que detecte un código postal español de cinco dígitos.",
    "¿Qué es un entorno virtual en Python y por qué se usa?",
    "Convierte este JSON a una tabla de dos columnas: {\"nombre\": \"Ana\", \"edad\": 31}",
    # tareas prácticas / consejo
    "Dame tres consejos para dormir mejor.",
    "¿Cómo se hace una tortilla de patatas? Resume los pasos.",
    "Propón un plan de tres días para visitar Lisboa.",
    "¿Qué debo llevar en una mochila para una excursión de un día a la montaña?",
    "Explica cómo cambiar una rueda pinchada.",
    "Dame ideas para un regalo de cumpleaños para alguien a quien le gusta la jardinería.",
    "¿Cómo puedo ahorrar energía en casa?",
    "Sugiere cinco nombres para una cafetería de barrio.",
    "¿Cómo se calcula el índice de masa corporal?",
    "What are three tips for giving a good presentation?",
    "¿Qué documentos suelen hacer falta para renovar el DNI en España?",
    "Explica qué es un presupuesto familiar y cómo empezar uno.",
    "Recomiéndame tres novelas clásicas de la literatura española.",
    "¿Qué hacer si se va la luz en casa?",
    "Describe los pasos para plantar un tomate en una maceta.",
    # tráfico GENERAL (no incluido en el catálogo) para no sobreajustar el dominio
    "¿Qué significa una señal triangular con el borde rojo?",
    "¿Cuál es la distancia de seguridad recomendada en autopista?",
    "¿Qué es el carné por puntos y cuántos puntos tiene un conductor novel en España?",
    "¿Qué hay que hacer si te encuentras un accidente de tráfico?",
    "Explica qué es una rotonda y quién tiene preferencia.",
    "¿Qué es la ITV y cada cuánto hay que pasarla en un turismo?",
    "¿Qué documentación debe llevar un conductor en el coche?",
    "¿Cuál es la tasa máxima de alcohol permitida para conductores noveles en España?",
    # formato / instrucciones
    "Haz una lista numerada con los días de la semana.",
    "Responde solo con 'sí' o 'no': ¿el agua hierve a 100 °C a nivel del mar?",
    "Devuelve un JSON con las claves 'ciudad' y 'pais' para París.",
    "Escribe la palabra 'hola' en cinco idiomas.",
    "Ordena alfabéticamente: pera, manzana, uva, kiwi, naranja.",
    "Cuenta del 10 al 1 hacia atrás.",
    "Dame una tabla en Markdown con tres frutas y su color.",
    "Explica en exactamente dos frases qué es un ordenador.",
    "Give me a bullet list of the primary colors.",
    "Responde en inglés: ¿cuál es tu color favorito y por qué?",
    # multiturno breve / razonamiento sentido común
    "Si dejo un helado al sol en agosto durante una hora, ¿qué pasará?",
    "¿Por qué el cielo es azul?",
    "¿Por qué flotan los barcos de acero si el acero se hunde?",
    "¿Qué pesa más, un kilo de plumas o un kilo de hierro?",
    "Si una habitación está a oscuras y tienes una cerilla, una vela y una lámpara de aceite, ¿qué enciendes primero?",
    "¿Por qué hay que lavarse las manos antes de comer?",
    "¿Qué es más rápido, la luz o el sonido? Pon un ejemplo cotidiano.",
    "¿Qué ocurre si mezclas bicarbonato con vinagre?",
    "Explica qué es el pensamiento crítico.",
    "¿Cuál es la diferencia entre tiempo y clima?",
    "Describe brevemente cómo funciona un motor de combustión.",
    "¿Qué es la empatía? Da un ejemplo.",
    "¿Por qué es importante dormir bien para estudiar?",
    "Explica qué es una hipoteca en términos sencillos.",
    "¿Qué es un ecosistema? Pon dos ejemplos.",
    "¿Qué diferencia hay entre una democracia y una dictadura?",
    "¿Qué es el método científico? Enumera sus pasos.",
]


def main():
    sanciones = parse_sanciones()
    contexts = build_contexts(sanciones)
    write_jsonl(DATA / "sanciones.jsonl", sanciones)
    (DATA / "contexts.json").write_text(json.dumps(contexts, ensure_ascii=False, indent=1))
    target = build_target(sanciones)
    write_jsonl(DATA / "eval_target.jsonl", target)
    retain, ppl = build_retain()
    write_jsonl(DATA / "eval_retain.jsonl", retain)
    write_jsonl(DATA / "eval_ppl.jsonl", ppl)
    write_jsonl(DATA / "generic_queries.jsonl", [dict(gid=f"G{i:03d}", question=q) for i, q in enumerate(GENERIC)])
    print(f"consultas genéricas: {len(GENERIC)}")
    # estadísticas de tokens de los contextos
    from mlx_lm import load

    _, tok = load("mlx-community/Qwen3.5-2B-bf16")
    lens = [len(tok.encode(c["text"])) for c in contexts]
    print(f"tokens por contexto: min {min(lens)}, mediana {sorted(lens)[len(lens)//2]}, max {max(lens)}, total {sum(lens)}")


if __name__ == "__main__":
    main()
