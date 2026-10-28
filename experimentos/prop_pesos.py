"""¿Los pesos tuneados del B0-v2 se propagan al MODELO FINAL (rerank) en dev?

Compara el Δ MRR del rerank (α=0.5, K=5) con los pesos DADOS vs los TUNEADOS. Usa embeddings
cacheados. El B0 baseline es el mismo en los dos casos (no depende de los pesos del v2).

    PYTHONPATH="$PWD" uv run --no-project --python 3.12 --with numpy --with pandas \
        --with pyarrow --with scikit-learn python prop_pesos.py
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

import b0
import b0_v2
import evaluar

VERT, DEV, TOP, ALPHA, K = "j", "2023S2", 200, 0.5, 5
PESOS_DADOS = [0.40, 0.24, 0.12, 0.04, 0.20, 0.10, 0.06, 0.15]
PESOS_TUNEADOS = [0.458, 0.0, 0.0, 0.0, 0.275, 0.148, 0.0, 0.119]


def ranks(s):
    o = np.argsort(-s, kind="stable")
    r = np.empty(len(s), dtype=np.float64); r[o] = np.arange(1, len(s) + 1)
    return r


def main():
    from sklearn.preprocessing import normalize
    d = pathlib.Path("data")
    lic = pd.read_parquet(d / VERT / "licitaciones.parquet")
    emp = pd.read_parquet(d / VERT / "empresas.parquet")
    hist = pd.read_parquet(d / VERT / "historial.parquet")
    gan = pd.read_parquet(d / VERT / "ganadores.parquet")
    off = pd.read_parquet(d / "office.parquet")
    lic["sem"] = lic.t.map(lambda t: f"{t.year}S{1 if t.month <= 6 else 2}")
    val = lic[lic["sem"] == DEV].copy()
    cand = emp[emp.semestre == DEV].reset_index(drop=True)
    uei = cand.parent_uei.values
    gan_dev = gan[gan.id.isin(set(val.id))]

    C = normalize(np.load(f".emb_cache/{VERT}_{DEV}_emp.npy"))
    Q = normalize(np.load(f".emb_cache/{VERT}_{DEV}_lic.npy"))
    prior = b0_v2.B0V2(hist, off)       # tablas (no dependen de los pesos)
    prior_b0 = b0.B0(hist)

    # ranking B0 (baseline, fijo)
    f_b0 = []
    for i, r in enumerate(val.itertuples()):
        ob = np.argsort(-prior_b0.p_b0(r, emp).reindex(uei, fill_value=0.0).values, kind="stable")[:100]
        f_b0.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ob], "puesto": np.arange(1, 101)}))
    rk_b0 = pd.concat(f_b0, ignore_index=True)

    def rerank_delta(nombre, pesos):
        b0_v2.PESOS = pesos              # _p lee este global en cada llamada
        filas = []
        for i, r in enumerate(val.itertuples()):
            p_v2 = prior.p_b0(r, emp).reindex(uei, fill_value=0.0).values
            sl = np.argsort(-p_v2, kind="stable")[:TOP]
            r_txt = ranks(Q[i] @ C[sl].T)
            r_cod = np.arange(1, len(sl) + 1, dtype=np.float64)
            s = (1 - ALPHA) / (K + r_cod) + ALPHA / (K + r_txt)
            ordr = sl[np.argsort(-s, kind="stable")[:100]]
            filas.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ordr], "puesto": np.arange(1, 101)}))
        res = evaluar.evaluar(pd.concat(filas, ignore_index=True), lic, emp, gan_dev, rk_b0)
        t = res[res.conjunto == "todas"].iloc[0]
        print(f"  {nombre:18s} ΔMRR={t['Δ MRR']:+.3f}  {t['Δ MRR IC95']}  "
              f"MRR={t.MRR:.3f}  Hit@5={t['Hit@5']:.3f}", flush=True)

    print(f"[{VERT}] dev {DEV}: {len(val)} licitaciones — modelo final (rerank α=0.5, K=5)\n")
    rerank_delta("pesos DADOS", PESOS_DADOS)
    rerank_delta("pesos TUNEADOS", PESOS_TUNEADOS)


if __name__ == "__main__":
    main()
