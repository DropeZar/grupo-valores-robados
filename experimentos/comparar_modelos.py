"""Compara los 4 embedders (MiniLM, bge-small, bge-base, Gemma) en el modelo final (rerank),
todos con el B0-v2 de Rocco (pesos+tau por vertical), en dev. Δ MRR vs B0, por vertical.

    PYTHONPATH="$PWD" uv run --no-project --python 3.12 --with numpy --with pandas \
        --with pyarrow --with scikit-learn python comparar_modelos.py
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
import b0_rocco

DEV, TOP, ALPHA, K = "2023S2", 200, 0.5, 5
VERTS = ["rd", "d", "j", "r", "it"]
MODELOS = [("MiniLM", ""), ("bge-small", "bges"), ("bge-base", "bgebase"), ("gemma", "gemma")]


def ranks(s):
    o = np.argsort(-s, kind="stable")
    r = np.empty(len(s), dtype=np.float64); r[o] = np.arange(1, len(s) + 1)
    return r


def main():
    from sklearn.preprocessing import normalize
    d = pathlib.Path("data")
    off = pd.read_parquet(d / "office.parquet")
    tabla = {}
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
        print(f"[{vert}] dev {DEV}: {len(val)} licitaciones", flush=True)

        prior_b0 = b0.B0(hist)
        prior = b0_rocco.B0V2(hist, off, vert=vert)
        # rankings B0 y p_v2 (independientes del embedder) una sola vez
        rk_b0_rows, p_v2_list, sl_list = [], [], []
        for i, r in enumerate(val.itertuples()):
            ob = np.argsort(-prior_b0.p_b0(r, emp).reindex(uei, fill_value=0.0).values, kind="stable")[:100]
            rk_b0_rows.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ob], "puesto": np.arange(1, 101)}))
            p_v2 = prior.p_b0(r, emp).reindex(uei, fill_value=0.0).values
            sl = np.argsort(-p_v2, kind="stable")[:TOP]
            p_v2_list.append(p_v2); sl_list.append(sl)
        rk_b0 = pd.concat(rk_b0_rows, ignore_index=True)

        for name, suf in MODELOS:
            fe = pathlib.Path(f".emb_cache/{vert}_{DEV}{('_'+suf) if suf else ''}_emp.npy")
            fl = pathlib.Path(f".emb_cache/{vert}_{DEV}{('_'+suf) if suf else ''}_lic.npy")
            if not (fe.exists() and fl.exists()):
                tabla[(vert, name)] = None; continue
            C, Q = normalize(np.load(fe)), normalize(np.load(fl))
            fr = []
            for i, r in enumerate(val.itertuples()):
                sl = sl_list[i]
                r_txt = ranks(Q[i] @ C[sl].T)
                r_cod = np.arange(1, len(sl) + 1, dtype=np.float64)
                s = (1 - ALPHA) / (K + r_cod) + ALPHA / (K + r_txt)
                ordr = sl[np.argsort(-s, kind="stable")[:100]]
                fr.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ordr], "puesto": np.arange(1, 101)}))
            res = evaluar.evaluar(pd.concat(fr, ignore_index=True), lic, emp, gan_dev, rk_b0)
            dm = res[res.conjunto == "todas"].iloc[0]["Δ MRR"]
            tabla[(vert, name)] = dm
            print(f"    {name:10s} ΔMRR={dm:+.3f}", flush=True)

    print("\n===== Δ MRR vs B0 (dev, con B0-v2 de Rocco) =====")
    df = pd.DataFrame(index=[n for n, _ in MODELOS], columns=VERTS, dtype=float)
    for (vert, name), v in tabla.items():
        df.loc[name, vert] = v
    df["PROM"] = df.mean(axis=1)
    with pd.option_context("display.float_format", "{:+.3f}".format):
        print(df.to_string())


if __name__ == "__main__":
    main()
