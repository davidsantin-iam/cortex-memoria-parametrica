"""
Paso 7: consolidación SECUENCIAL (continual), §5.2 del paper, a escala de portátil.

Los 27 contextos se agrupan en bloques (por norma). Cada bloque llega como una
actualización t = 1..T: se parte del modelo M_{t-1} (adaptadores anteriores plegados y
congelados), se entrena un adaptador nuevo con los ejemplos de ensayo de ese bloque,
el banco genérico y *query replay* (consultas antiguas re-respondidas por M_{t-1}, sin
los contextos antiguos). Tras cada actualización se evalúa todo lo consolidado hasta
el momento y la retención. Un segundo brazo (--no-replay) repite la secuencia sin replay.

Uso:  python 07_sequential.py --variant structural --rank 32 [--no-replay] [--epochs 6]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from common import CKPT, DATA, RESULTS, read_jsonl, write_jsonl

PY = sys.executable

GROUPS = {  # bloques por norma (orden de llegada)
    "CON+SOA": ["CON", "SOA"],
    "CIR+RGC": ["CIR", "RGC"],
    "LSV-1": ["LSV", 0],
    "LSV-2": ["LSV", 1],
    "OMS": ["OMS"],
    "VEH-1": ["VEH", 0],
    "VEH-2": ["VEH", 1],
}


def group_contexts(contexts):
    by_norma = {}
    for c in contexts:
        by_norma.setdefault(c["norma"], []).append(c["ctx_id"])
    out = {}
    for g, spec in GROUPS.items():
        if len(spec) == 2 and isinstance(spec[1], int):
            ids = by_norma.get(spec[0], [])
            half = (len(ids) + 1) // 2
            out[g] = ids[:half] if spec[1] == 0 else ids[half:]
        else:
            out[g] = [i for n in spec for i in by_norma.get(n, [])]
    return {g: v for g, v in out.items() if v}


def sh(cmd):
    print("\n$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = [l for l in r.stdout.splitlines() if "Fetching" not in l and "Finished processing" not in l]
    print("\n".join(lines[-12:]))
    if r.returncode:
        raise SystemExit(f"fallo: {cmd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="structural")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()
    prefix = args.prefix or (f"seq_{args.variant}" + ("_noreplay" if args.no_replay else "") + "_")

    contexts = json.loads((DATA / "contexts.json").read_text())
    groups = group_contexts(contexts)
    (RESULTS / "sequential_groups.json").write_text(json.dumps(groups, indent=1, ensure_ascii=False))
    print("bloques:", {g: len(v) for g, v in groups.items()})
    rehearsal = read_jsonl(DATA / "rehearsal.jsonl")

    chain, seen_ctx = [], []
    for t, (g, ctxs) in enumerate(groups.items(), start=1):
        name = f"{prefix}{t}"
        ckpt = CKPT / name
        if not ckpt.exists():
            cmd = [PY, "05_train.py", "--name", name, "--variant", args.variant, "--rank", str(args.rank), "--objective", "kl_mse", "--lr", "1e-4", "--epochs", str(args.epochs), "--ctx-filter", ",".join(ctxs), "--generic", "40"]
            if chain:
                cmd += ["--init-from", ",".join(str(c) for c in chain)]
            if seen_ctx and not args.no_replay:
                old = [r for r in rehearsal if r["ctx_id"] in seen_ctx]
                rp = DATA / f"replay_{name}.jsonl"
                write_jsonl(rp, [dict(q=r["q"]) for r in old])
                cmd += ["--replay-from", str(rp), "--replay-n", "60"]
            sh(cmd)
        chain.append(ckpt)
        seen_ctx += ctxs
        if not (RESULTS / f"eval_{name}.json").exists():
            sh([PY, "03_eval.py", "--name", name, "--adapter", ",".join(str(c) for c in chain), "--only-ctx", ",".join(seen_ctx)])
    sh([PY, "06_metrics.py", "--seq", prefix])
    print("FIN secuencial", prefix)


if __name__ == "__main__":
    main()
