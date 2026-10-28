"""Gráfico comparativo: el modelo final (rerank bge-small) con 3 bases de historial —
B0, B0-v2 y B0-v2 optimizado (pesos+tau por vertical) — en dev. Δ MRR vs B0, barras agrupadas.
Guarda figuras/comparacion_b0.png.

    PYTHONPATH="$PWD" uv run --no-project --python 3.12 --with numpy --with pandas \
        --with pyarrow --with scikit-learn --with matplotlib python experimentos/grafico_b0.py
"""
from __future__ import annotations

import pathlib
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import b0
import evaluar
import b0_v2
import b0_rocco

DEV, TOP, ALPHA, K, SUF = "2023S2", 200, 0.5, 5, "bges"
VERTS = ["rd", "d", "j", "r", "it"]
AZUL, GRIS, TEAL = "#2e6f95", "#b0b0b0", "#1d9e75"


def ranks(s):
    o = np.argsort(-s, kind="stable")
    r = np.empty(len(s), dtype=np.float64); r[o] = np.arange(1, len(s) + 1)
    return r


def main():
    from sklearn.preprocessing import normalize
    d = pathlib.Path("data")
    off = pd.read_parquet(d / "office.parquet")
    res = {"B0": [], "B0-v2": [], "B0-v2 opt.": []}
    for vert in VERTS:
        lic = pd.read_parquet(d / vert / "licitaciones.parquet")
        emp = pd.read_parquet(d / vert / "empresas.parquet")
        hist = pd.read_parquet(d / vert / "historial.parquet")
        gan = pd.read_parquet(d / vert / "ganadores.parquet")
        lic["sem"] = lic.t.map(lambda t: f"{t.year}S{1 if t.month <= 6 else 2}")
        val = lic[lic["sem"] == DEV].copy()
        cand = emp[emp.semestre == DEV].reset_index(drop=True)
        uei = cand.parent_uei.values
        gan_dev = gan[gan.id.isin(set(val.id))]
        C = normalize(np.load(f".emb_cache/{vert}_{DEV}_{SUF}_emp.npy"))
        Q = normalize(np.load(f".emb_cache/{vert}_{DEV}_{SUF}_lic.npy"))
        print(f"[{vert}] {len(val)} licitaciones", flush=True)

        prior_b0 = b0.B0(hist)
        rk_b0 = pd.concat(
            [pd.DataFrame({"id": r.id, "parent_uei":
                           uei[np.argsort(-prior_b0.p_b0(r, emp).reindex(uei, fill_value=0.0).values, kind="stable")[:100]],
                           "puesto": np.arange(1, 101)}) for r in val.itertuples()],
            ignore_index=True)

        bases = {"B0": prior_b0, "B0-v2": b0_v2.B0V2(hist, off),
                 "B0-v2 opt.": b0_rocco.B0V2(hist, off, vert=vert)}
        for name, prior in bases.items():
            fr = []
            for i, r in enumerate(val.itertuples()):
                p = prior.p_b0(r, emp).reindex(uei, fill_value=0.0).values
                sl = np.argsort(-p, kind="stable")[:TOP]
                r_txt = ranks(Q[i] @ C[sl].T)
                r_cod = np.arange(1, len(sl) + 1, dtype=np.float64)
                s = (1 - ALPHA) / (K + r_cod) + ALPHA / (K + r_txt)
                ordr = sl[np.argsort(-s, kind="stable")[:100]]
                fr.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ordr], "puesto": np.arange(1, 101)}))
            rr = evaluar.evaluar(pd.concat(fr, ignore_index=True), lic, emp, gan_dev, rk_b0)
            dm = rr[rr.conjunto == "todas"].iloc[0]["Δ MRR"]
            res[name].append(dm)
            print(f"    {name:12s} ΔMRR={dm:+.3f}", flush=True)

    # --- barras agrupadas ---
    x = np.arange(len(VERTS)); w = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    for off_, (name, col) in zip((-w, 0, w), (("B0", GRIS), ("B0-v2", AZUL), ("B0-v2 opt.", TEAL))):
        ax.bar(x + off_, res[name], w, label=name, color=col)
    ax.set_xticks(x); ax.set_xticklabels(VERTS)
    ax.set_ylabel("Δ MRR sobre el B0 (dev)")
    ax.set_title("Modelo final (rerank bge-small) según la base de historial")
    ax.axhline(0, color="k", lw=0.6); ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = "figuras/comparacion_b0.png"
    fig.savefig(out, dpi=140)
    print(f"\nfigura -> {out}")
    print(pd.DataFrame(res, index=VERTS).to_string())


if __name__ == "__main__":
    main()
