"""¿Los pesos+taus de Rocco (b0_rocco.B0V2, por vertical) propagan al MODELO FINAL?

Compara el Δ MRR del rerank (α=0.5, K=5, embeddings bge-small cacheados) con el B0-v2 DEFAULT
vs el B0-v2 TUNEADO de Rocco (pesos + tau por vertical), en dev, en las 5 verticales.

    PYTHONPATH="$PWD" uv run --no-project --python 3.12 --with numpy --with pandas \
        --with pyarrow --with scikit-learn python prop_rocco.py
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
import evaluar
import b0_v2
import b0_rocco

DEV, TOP, ALPHA, K, SUF = "2023S2", 200, 0.5, 5, "bges"
VERTS = ["rd", "d", "j", "r", "it"]


def ranks(s):
    o = np.argsort(-s, kind="stable")
    r = np.empty(len(s), dtype=np.float64); r[o] = np.arange(1, len(s) + 1)
    return r


def main():
    from sklearn.preprocessing import normalize
    d = pathlib.Path("data")
    off = pd.read_parquet(d / "office.parquet")
    filas = []
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
        print(f"[{vert}] dev {DEV}: {len(val)} licitaciones", flush=True)

        prior_b0 = b0.B0(hist)
        f_b0 = []
        for i, r in enumerate(val.itertuples()):
            ob = np.argsort(-prior_b0.p_b0(r, emp).reindex(uei, fill_value=0.0).values, kind="stable")[:100]
            f_b0.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ob], "puesto": np.arange(1, 101)}))
        rk_b0 = pd.concat(f_b0, ignore_index=True)

        def rerank_delta(prior):
            fr = []
            for i, r in enumerate(val.itertuples()):
                p_v2 = prior.p_b0(r, emp).reindex(uei, fill_value=0.0).values
                sl = np.argsort(-p_v2, kind="stable")[:TOP]
                r_txt = ranks(Q[i] @ C[sl].T)
                r_cod = np.arange(1, len(sl) + 1, dtype=np.float64)
                s = (1 - ALPHA) / (K + r_cod) + ALPHA / (K + r_txt)
                ordr = sl[np.argsort(-s, kind="stable")[:100]]
                fr.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ordr], "puesto": np.arange(1, 101)}))
            res = evaluar.evaluar(pd.concat(fr, ignore_index=True), lic, emp, gan_dev, rk_b0)
            t = res[res.conjunto == "todas"].iloc[0]
            return t["Δ MRR"], t["Δ MRR IC95"]

        dd, dic = rerank_delta(b0_v2.B0V2(hist, off))
        rd_, ric = rerank_delta(b0_rocco.B0V2(hist, off, vert=vert))
        filas.append({"vert": vert, "default ΔMRR": dd, "Rocco ΔMRR": rd_, "dif": rd_ - dd})
        print(f"  default={dd:+.3f} {dic}  |  Rocco={rd_:+.3f} {ric}  |  dif={rd_-dd:+.3f}", flush=True)

    print("\n===== modelo final (rerank bge-small) — B0-v2 default vs Rocco (dev) =====")
    with pd.option_context("display.float_format", "{:+.3f}".format):
        print(pd.DataFrame(filas).to_string(index=False))


if __name__ == "__main__":
    main()
