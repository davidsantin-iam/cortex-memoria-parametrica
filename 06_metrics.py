"""
Paso 6: métricas de consolidación (CR) y retención (RR) y figuras del artículo.

  CR = (Perf(M', Q+) - Perf(M, Q+)) / (Perf(M, Q+, C) - Perf(M, Q+))
  RR = Perf(M', Q-) / Perf(M, Q-)

Lee results/eval_*.json. Uso:
  python 06_metrics.py                 # tabla + figuras principales
  python 06_metrics.py --seq seq_      # además, trayectorias del experimento secuencial con ese prefijo
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FIGURES, LOGS, RESULTS  # noqa: E402

# paleta (referencia validada del skill dataviz): azul, naranja, aqua, amarillo, magenta, verde, violeta, rojo
C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100", magenta="#e87ba4", green="#008300", violet="#6250d6", red="#e34948", gray="#8a8984", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e1")

TYPE_LABEL = {
    "cuantia": "Importe", "reduccion": "Pronto pago", "puntos": "Puntos", "gravedad": "Gravedad", "precepto": "Precepto",
    "comparacion": "Comparación", "cuantia_para": "Importe (paráfrasis)", "gravedad_para": "Gravedad (paráfrasis)", "puntos_para": "Puntos (paráfrasis)",
    "mmlu_es": "MMLU-es", "gsm8k": "GSM8K",
}
TARGET_TYPES = ["cuantia", "reduccion", "puntos", "gravedad", "precepto", "comparacion", "cuantia_para", "gravedad_para", "puntos_para"]

# nombre de run -> (etiqueta, color, marcador)
RUNS = {
    "synaptic_r16": ("Cortex-Sináptico (LoRA FFN, KL+MSE)", C["blue"], "o"),
    "structural_k64": ("Cortex-Estructural (canales FFN, KL+MSE)", C["orange"], "s"),
    "synaptic_kl": ("Sináptico solo KL (sin MSE)", C["aqua"], "^"),
    "lora_all_ce": ("SFT clásico (LoRA-All, CE)", C["violet"], "D"),
    "docs_ce": ("Fine-tune sobre documentos (CE)", C["magenta"], "v"),
    "synaptic_ce": ("LoRA FFN con CE (sin profesor)", C["yellow"], "P"),
    "synaptic_fullvocab": ("Primer intento: hiperparámetros literales del paper", C["red"], "X"),
    "synaptic_fullvocab_lr1e-4": ("KL literal con la tasa de aprendizaje corregida", C["violet"], "x"),
    "synaptic_v4": ("Sináptico, respuestas propias filtradas", C["magenta"], "o"),
    "structural_v4": ("Estructural, respuestas propias filtradas", C["magenta"], "s"),
    "synaptic_v3": ("Sináptico, profesor externo 27B", C["green"], "o"),
    "structural_v3": ("Estructural, profesor externo 27B", C["green"], "s"),
    "synaptic_v2": ("Sináptico, ensayo por facetas", C["aqua"], "o"),
    "structural_v2": ("Estructural, ensayo por facetas", C["yellow"], "s"),
}


def load_runs() -> dict[str, dict]:
    runs = {}
    for p in sorted(RESULTS.glob("eval_*.json")):
        d = json.loads(p.read_text())
        runs[d["name"]] = d["metrics"]
    return runs


def acc(m: dict, t: str) -> float | None:
    return m[t]["acc"] if t in m else None


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(C["grid"])
    ax.tick_params(colors=C["ink2"], labelsize=9)
    ax.yaxis.grid(True, color=C["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def cr_rr(runs: dict, name: str, base="vanilla", ctx="context") -> tuple[float, float]:
    v, c, m = runs[base], runs[ctx], runs[name]
    gap = c["target_score"] - v["target_score"]
    cr = (m["target_score"] - v["target_score"]) / gap if gap else float("nan")
    rr = m["retain_score"] / v["retain_score"] if v.get("retain_score") else float("nan")
    return cr, rr


def table(runs: dict):
    rows = []
    names = ["vanilla", "context"] + [n for n in RUNS if n in runs]
    hdr = ["run", "objetivo", "MMLU-es", "GSM8K", "PPL", "CR", "RR"]
    print(" | ".join(hdr))
    for n in names:
        m = runs[n]
        cr, rr = (float("nan"), float("nan")) if n in ("vanilla", "context") else cr_rr(runs, n)
        row = [n, f"{m.get('target_score', float('nan')):.3f}", f"{acc(m, 'mmlu_es') or float('nan'):.3f}", f"{acc(m, 'gsm8k') or float('nan'):.3f}", f"{m.get('ppl_wiki_es', float('nan')):.2f}", f"{100*cr:.1f}%" if cr == cr else "-", f"{100*rr:.1f}%" if rr == rr else "-"]
        rows.append(row)
        print(" | ".join(row))
    (RESULTS / "summary_table.md").write_text("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n")
    return rows


def fig_context_gap(runs: dict):
    v, c = runs["vanilla"], runs["context"]
    types = [t for t in TARGET_TYPES if t in v and t in c]
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
    x = range(len(types))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [100 * acc(v, t) for t in types], w, color=C["gray"], label="Modelo base, sin contexto")
    ax.bar([i + w / 2 for i in x], [100 * acc(c, t) for t in types], w, color=C["blue"], label="Modelo base + contexto correcto")
    for i, t in enumerate(types):
        ax.text(i - w / 2, 100 * acc(v, t) + 1, f"{100*acc(v,t):.0f}", ha="center", fontsize=8, color=C["ink2"])
        ax.text(i + w / 2, 100 * acc(c, t) + 1, f"{100*acc(c,t):.0f}", ha="center", fontsize=8, color=C["ink2"])
    ax.set_xticks(list(x), [TYPE_LABEL[t] for t in types], rotation=20, ha="right")
    ax.set_ylabel("Acierto (%)", color=C["ink2"])
    ax.set_ylim(0, 122)
    ax.set_title("La brecha de contexto: lo que el modelo sabe hacer solo si le damos el catálogo", fontsize=11, color=C["ink"], loc="left")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_context_gap.png")
    plt.close(fig)


def fig_pareto(runs: dict):
    """Dos paneles: CR frente a RR (benchmarks) y CR frente a perplejidad (deformación del lenguaje)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=160, gridspec_kw=dict(width_ratios=[1, 1]))
    ax, ax2 = axes
    present = [n for n in RUNS if n in runs]
    for a in axes:
        a.axvline(0, color=C["grid"], linewidth=1)
        a.axvline(100, color=C["grid"], linewidth=1)
    ax.axhline(100, color=C["grid"], linewidth=1)
    ax.scatter([0], [100], color=C["gray"], s=80, zorder=3, label="Modelo base (sin contexto)")
    ax.scatter([100], [100], color=C["ink"], marker="*", s=160, zorder=3, label="Modelo base + contexto (referencia)")
    ax2.axhline(runs["vanilla"]["ppl_wiki_es"], color=C["grid"], linewidth=1)
    ax2.scatter([0], [runs["vanilla"]["ppl_wiki_es"]], color=C["gray"], s=80, zorder=3)
    for n in present:
        label, col, mk = RUNS[n]
        cr, rr = cr_rr(runs, n)
        ax.scatter([100 * cr], [100 * rr], color=col, marker=mk, s=95, zorder=4, edgecolor="white", linewidth=1.2, label=label)
        ppl = runs[n].get("ppl_wiki_es", float("nan"))
        ax2.scatter([100 * cr], [ppl], color=col, marker=mk, s=95, zorder=4, edgecolor="white", linewidth=1.2)
        ax2.annotate(f"{ppl:.1f}", (100 * cr, ppl), textcoords="offset points", xytext=(7, 4), fontsize=7.5, color=C["ink2"])
    ax.set_xlabel("Tasa de consolidación CR (%)", color=C["ink2"], fontsize=9)
    ax.set_ylabel("Tasa de retención RR (%)  [MMLU-es + GSM8K]", color=C["ink2"], fontsize=9)
    ax.set_title("Consolidación frente a retención", fontsize=11, color=C["ink"], loc="left")
    ax.set_xlim(-8, 112)
    ax.set_ylim(35, 115)
    ax2.set_xlabel("Tasa de consolidación CR (%)", color=C["ink2"], fontsize=9)
    ax2.set_ylabel("Perplejidad en Wikipedia-es (menor es mejor)", color=C["ink2"], fontsize=9)
    ax2.set_title("Consolidación frente a deformación del lenguaje", fontsize=11, color=C["ink"], loc="left")
    ax2.set_xlim(-8, 112)
    ax2.text(2, runs["vanilla"]["ppl_wiki_es"] + 0.4, "perplejidad del modelo base", fontsize=7.5, color=C["ink2"])
    for a in axes:
        style(a)
        a.xaxis.grid(True, color=C["grid"], linewidth=0.8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(FIGURES / "fig2_pareto_cr_rr.png")
    plt.close(fig)


def fig_per_type(runs: dict):
    main = ["synaptic_r16", "synaptic_v2", "synaptic_v3", "synaptic_v4", "lora_all_ce"]
    names = ["vanilla"] + [n for n in main if n in runs] + ["context"]
    labels = {"vanilla": "Modelo base", "context": "Base + contexto (referencia)", **{k: v[0].split(" (")[0] for k, v in RUNS.items()}}
    colors = {"vanilla": C["gray"], "context": C["ink"], **{k: v[1] for k, v in RUNS.items()}}
    types = [t for t in TARGET_TYPES if all(t in runs[n] for n in names)]
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=160)
    k = len(names)
    w = 0.8 / k
    for j, n in enumerate(names):
        xs = [i - 0.4 + w * (j + 0.5) for i in range(len(types))]
        ax.bar(xs, [100 * acc(runs[n], t) for t in types], w * 0.92, color=colors[n], label=labels[n])
    ax.set_xticks(range(len(types)), [TYPE_LABEL[t] for t in types], rotation=20, ha="right")
    ax.set_ylabel("Acierto (%)", color=C["ink2"])
    ax.set_ylim(0, 105)
    ax.set_title("Acierto por tipo de pregunta sobre el catálogo (sin contexto en el prompt, salvo la referencia)", fontsize=11, color=C["ink"], loc="left", pad=46)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_per_type.png")
    plt.close(fig)


def fig_retention(runs: dict):
    names = ["vanilla"] + [n for n in RUNS if n in runs]
    labels = {"vanilla": "Base", **{k: v[0].split(" (")[0] for k, v in RUNS.items()}}
    colors = {"vanilla": C["gray"], **{k: v[1] for k, v in RUNS.items()}}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.9), dpi=160, gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
    for k, (ax, (metric, title, fmt)) in enumerate(zip(axes, [("mmlu_es", "MMLU en español (acierto %)", 100), ("gsm8k", "GSM8K (acierto %)", 100), ("ppl_wiki_es", "Perplejidad Wikipedia-es (↓)", 1)])):
        vals = []
        for n in names:
            m = runs[n]
            vals.append(fmt * (acc(m, metric) if metric != "ppl_wiki_es" else m.get(metric, float("nan"))))
        ax.barh(range(len(names)), vals, color=[colors[n] for n in names])
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.1f}" if fmt == 100 else f" {v:.2f}", va="center", fontsize=8, color=C["ink2"])
        ax.set_yticks(range(len(names)), [labels[n] for n in names] if k == 0 else [""] * len(names), fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10, color=C["ink"], loc="left")
        style(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=C["grid"], linewidth=0.8)
        ax.set_xlim(0, max(vals) * 1.25)
    fig.suptitle("¿Qué pierde el modelo en todo lo demás?", fontsize=11, color=C["ink"], x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_retention.png")
    plt.close(fig)


def fig_training_curves():
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=160)
    for n, (label, col, _) in RUNS.items():
        p = LOGS / f"train_{n}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        key = "ce" if rows and rows[-1]["kl"] == 0 else "kl"
        ax.plot([r["update"] for r in rows], [r[key] for r in rows], color=col, linewidth=2, label=f"{label.split(' (')[0]} ({'CE' if key=='ce' else 'KL'})")
    ax.set_xlabel("Actualizaciones del optimizador", color=C["ink2"])
    ax.set_ylabel("Pérdida", color=C["ink2"])
    ax.set_title("Curvas de entrenamiento", fontsize=11, color=C["ink"], loc="left")
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_training.png")
    plt.close(fig)


def _seq_series(prefix: str, seq_cfg: dict, base: dict):
    files = sorted(RESULTS.glob(f"eval_{prefix}*.json"), key=lambda p: int(p.stem.split("_")[-1]))
    files = [f for f in files if f.stem[len("eval_" + prefix):].isdigit()]
    steps, by_group, retain, ppl, target = [], {}, [], [], []
    for p in files:
        d = json.loads(p.read_text())
        steps.append(int(p.stem.split("_")[-1]))
        for gname, ctxs in seq_cfg.items():
            its = [i for i in d["items"] if i.get("ctx_id") in ctxs and i["type"] in TARGET_TYPES]
            by_group.setdefault(gname, []).append(100 * sum(i["correct"] for i in its) / len(its) if its else float("nan"))
        retain.append(100 * d["metrics"]["retain_score"] / base["retain_score"])
        ppl.append(d["metrics"]["ppl_wiki_es"])
        # acierto medio sobre todo lo consolidado hasta t
        its = [i for i in d["items"] if i["type"] in TARGET_TYPES]
        target.append(100 * sum(i["correct"] for i in its) / len(its))
    return dict(steps=steps, by_group=by_group, rr=retain, ppl=ppl, target=target)


def fig_sequential(prefix: str, runs: dict):
    """Experimento secuencial: brazo con query replay (prefix) y sin él (prefix + 'noreplay_')."""
    base = json.loads((RESULTS / "eval_vanilla.json").read_text())["metrics"]
    seq_cfg = json.loads((RESULTS / "sequential_groups.json").read_text())
    arms = {"con query replay": _seq_series(prefix, seq_cfg, base)}
    nr = prefix.rstrip("_") + "_noreplay_"
    if list(RESULTS.glob(f"eval_{nr}*.json")):
        arms["sin replay"] = _seq_series(nr, seq_cfg, base)
    if not arms["con query replay"]["steps"]:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=160)
    cols = [C["blue"], C["orange"], C["aqua"], C["yellow"], C["magenta"], C["violet"], C["red"], C["green"]]
    s = arms["con query replay"]
    for (g, ys), col in zip(s["by_group"].items(), cols):
        t0 = list(seq_cfg).index(g) + 1
        xs = [t for t in s["steps"] if t >= t0]
        axes[0].plot(xs, [y for t, y in zip(s["steps"], ys) if t >= t0], marker="o", color=col, linewidth=1.8, label=g)
    axes[0].set_title("Acierto por bloque tras cada actualización (con replay)", fontsize=10, color=C["ink"], loc="left")
    axes[0].set_xlabel("Actualización secuencial t", color=C["ink2"]); axes[0].set_ylabel("Acierto (%)", color=C["ink2"])
    axes[0].legend(frameon=False, fontsize=7, ncol=2); style(axes[0])
    for (name, a), col, mk in zip(arms.items(), [C["blue"], C["gray"]], ["o", "s"]):
        axes[1].plot(a["steps"], a["target"], marker=mk, color=col, linewidth=2, label=name)
        axes[2].plot(a["steps"], a["rr"], marker=mk, color=col, linewidth=2, label=name)
    bitems = [i for i in json.loads((RESULTS / "eval_structural_k64.json").read_text())["items"] if i["type"] in TARGET_TYPES]
    bref = 100 * sum(i["correct"] for i in bitems) / len(bitems)
    axes[1].axhline(bref, color=C["orange"], linestyle="--", linewidth=1.2)
    axes[1].text(1, bref + 0.5, "consolidación en lote, mismo modelo (referencia)", fontsize=7.5, color=C["orange"])
    axes[1].set_title("Acierto medio sobre todo lo consolidado hasta t", fontsize=10, color=C["ink"], loc="left")
    axes[1].set_xlabel("Actualización secuencial t", color=C["ink2"]); axes[1].legend(frameon=False, fontsize=8); style(axes[1])
    axes[2].axhline(100, color=C["grid"]); axes[2].set_ylim(85, 112)
    axes[2].set_title("Retención RR (%) a lo largo de la secuencia", fontsize=10, color=C["ink"], loc="left")
    axes[2].set_xlabel("Actualización secuencial t", color=C["ink2"]); axes[2].legend(frameon=False, fontsize=8); style(axes[2])
    fig.tight_layout(); fig.savefig(FIGURES / "fig6_sequential.png"); plt.close(fig)
    (RESULTS / "sequential_summary.json").write_text(json.dumps(arms, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="")
    args = ap.parse_args()
    runs = load_runs()
    if "vanilla" not in runs or "context" not in runs:
        print("faltan eval_vanilla / eval_context")
        return
    table(runs)
    fig_context_gap(runs)
    if any(n in runs for n in RUNS):
        fig_pareto(runs)
        fig_per_type(runs)
        fig_retention(runs)
        fig_training_curves()
    if args.seq:
        fig_sequential(args.seq, runs)
    print("figuras en", FIGURES)


if __name__ == "__main__":
    main()
