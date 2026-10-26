# Réplica local de *Cortex: Consolidating Context into Parametric Memory*

Código para replicar, a escala de portátil (Apple Silicon + MLX), el procedimiento de
consolidación de memoria de Zhang et al. (2026, arXiv:2610.20399): hacer que un modelo
de lenguaje pequeño (**Qwen3.5-2B**) incorpore en sus pesos el contenido de unos
documentos (en nuestro caso, un catálogo de infracciones y sanciones de tráfico en XML)
sin perder sus capacidades generales.

Este repositorio contiene **solo el código**. Los datos (el catálogo, los conjuntos de
evaluación y los ensayos generados por el modelo), los adaptadores entrenados y los
resultados no se incluyen; `00_prepare_data.py` reconstruye los conjuntos de evaluación a
partir del XML y de conjuntos públicos (MMMLU, GSM8K, Wikipedia).

## Qué implementa

- **Ensayo prospectivo**: el propio modelo genera preguntas sobre cada bloque de contexto y
  las responde con el bloque delante (`02_rehearsal.py`); variante guiada por facetas y
  variante con profesor externo servido por LM Studio y filtro contra la fuente de verdad
  (`08_outside_teacher.py`).
- **Destilación de contexto a memoria**: profesor = el mismo modelo con contexto; alumno =
  sin contexto. Pérdida KL sobre los `top-K` logits del profesor (alumno renormalizado sobre
  el mismo conjunto) + MSE sobre los estados ocultos tras cada FFN (`04_cache_teacher.py`,
  `05_train.py`).
- **Dos plasticidades** restringidas al FFN: LoRA ("sináptica") y canales nuevos
  ("estructural"), ambas plegables en los pesos (`common.py`). Líneas base: LoRA-All con
  entropía cruzada y fine-tuning sobre el texto crudo.
- **Evaluación** con jueces deterministas y métricas CR / RR del paper (`03_eval.py`,
  `06_metrics.py`); **consolidación secuencial** con *query replay* (`07_sequential.py`).
- `fast_gdn.py`: regla delta con compuerta (Gated DeltaNet de Qwen3.5) en forma *chunked* y
  diferenciable para MLX, que sustituye al bucle token a token de `mlx_lm` en entrenamiento.

## Requisitos

- macOS con chip Apple (probado en M5 Pro, 64 GB). Python 3.12 con `mlx`, `mlx_lm` (>= 0.31),
  `numpy`, `matplotlib`.
- Opcional: LM Studio sirviendo un modelo grande en `http://127.0.0.1:1234` (paráfrasis de
  evaluación y profesor externo). El identificador del modelo se toma de la variable `LMS_MODEL`.
- El XML del catálogo se espera en `../normativa/getSanciones.txt` (ajustable en `common.py`).

## Flujo

| Paso | Script | Qué hace |
|---|---|---|
| 0 | `00_prepare_data.py` | Parsea el XML, construye los contextos, el examen objetivo y descarga los conjuntos de retención. |
| 1 | `01_paraphrase.py` | Paráfrasis de las descripciones con un modelo externo (prueba de generalización). |
| 2 | `02_rehearsal.py` | Ensayo prospectivo (`--facets` para una pregunta por faceta). |
| 3 | `03_eval.py` | Evalúa una condición (base, base+contexto, adaptadores). |
| 4 | `04_cache_teacher.py` | Guarda logits top-K y estados post-FFN del profesor. |
| 5 | `05_train.py` | Entrena la variante elegida (plasticidad, objetivo, datos). |
| 6 | `06_metrics.py` | Calcula CR y RR y dibuja las figuras. |
| 7 | `07_sequential.py` | Consolidación secuencial con *query replay*. |
| 8 | `08_outside_teacher.py` | Respuestas de un profesor externo filtradas contra el XML. |
| 9 | `09_build_v5.py` | Ensayo v5: contextos con el articulado en lenguaje natural (`contexts`) y unión de varias rondas de ensayo propio filtradas contra el XML más preguntas de articulado por plantilla (`merge`). |

`run_experiments.sh` encadena el experimento principal; `run_v2.sh`…`run_v5.sh`, las variantes
del ensayo; `null_test.sh`, la prueba de estabilidad del optimizador (entrenar solo con ejemplos
en los que profesor y alumno coinciden). Los scripts usan el intérprete de la variable `PY`
(por defecto `python`).

## Métricas

- **CR** (tasa de consolidación) = (Perf(M′) − Perf(M)) / (Perf(M, C) − Perf(M)) sobre las
  preguntas objetivo.
- **RR** (tasa de retención) = Perf(M′, Q−) / Perf(M, Q−), con Q− = MMMLU-ES + GSM8K; la
  perplejidad sobre texto general se reporta aparte.

## Licencia

MIT. Si usas el código, cita el paper original (Zhang et al., 2026, arXiv:2610.20399).
