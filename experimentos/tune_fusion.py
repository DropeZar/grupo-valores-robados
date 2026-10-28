"""Paso 1: tunear la fusión del rerank — barrido de α (peso del texto) y K (RRF) en dev.

Puntaje fusionado, reordenando el top-200 del B0-v2:

    s(q,e) = (1 - α)/(K + r_cod(q,e))  +  α/(K + r_txt(q,e))

donde r_cod = puesto por incumbencia (v2) y r_txt = puesto por similitud de embeddings, ambos
dentro del shortlist. r_cod y r_txt NO dependen de (α, K): se calculan una vez y el barrido es
solo re-ordenar. Usa los embeddings cacheados por latente_emb.py.

    uv run --no-project --python 3.12 --with numpy --with pandas --with pyarrow \
        --with scikit-learn python tune_fusion.py --vert j --dev 2023S2
"""
from __future__ import annotations

import argparse
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
from b0_v2 import B0V2

TOP = 200
ALPHAS = [0.40, 0.45, 0.50, 0.55, 0.60]
KS = [3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8]


def ranks(scores):
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.float64)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def mrr_de(best_ranks):
    return float(np.mean([1.0 / b if b <= 100 else 0.0 for b in best_ranks]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vert", default="j")
    ap.add_argument("--dev", default="2023S2")
    ap.add_argument("--datos", default="data")
    a = ap.parse_args()

    d = pathlib.Path(a.datos)
    lic = pd.read_parquet(d / a.vert / "licitaciones.parquet")
    emp = pd.read_parquet(d / a.vert / "empresas.parquet")
    hist = pd.read_parquet(d / a.vert / "historial.parquet")
    gan = pd.read_parquet(d / a.vert / "ganadores.parquet")
    off = pd.read_parquet(d / "office.parquet")

    lic["sem"] = lic.t.map(lambda t: f"{t.year}S{1 if t.month <= 6 else 2}")
    val = lic[lic["sem"] == a.dev].copy()
    cand = emp[emp.semestre == a.dev].reset_index(drop=True)
    uei = cand.parent_uei.values
    uei_pos = {u: i for i, u in enumerate(uei)}
    gan_dev = gan[gan.id.isin(set(val.id))]
    win = gan_dev.groupby("id").parent_uei.apply(list).to_dict()

    cache = pathlib.Path(".emb_cache")
    f_emp, f_lic = cache / f"{a.vert}_{a.dev}_emp.npy", cache / f"{a.vert}_{a.dev}_lic.npy"
    if not (f_emp.exists() and f_lic.exists()):
        sys.exit("faltan embeddings en .emb_cache/: corré primero latente_emb.py")
    from sklearn.preprocessing import normalize
    C = normalize(np.load(f_emp)); Q = normalize(np.load(f_lic))
    print(f"vertical {a.vert} | dev {a.dev}: {len(val)} licitaciones, {len(cand):,} candidatas")

    prior_b0 = b0.B0(hist)
    prior_v2 = B0V2(hist, off)

    # precálculo por licitación: r_cod, r_txt y posiciones de ganadores en el shortlist
    print("precalculando shortlists…")
    pre, best_b0 = [], []
    for i, r in enumerate(val.itertuples()):
        p_v2 = prior_v2.p_b0(r, emp).reindex(uei, fill_value=0.0).values
        sl = np.argsort(-p_v2, kind="stable")[:TOP]
        tsim = Q[i] @ C[sl].T
        r_cod = np.arange(1, len(sl) + 1, dtype=np.float64)
        r_txt = ranks(tsim)
        ws = [uei_pos[w] for w in win.get(r.id, []) if w in uei_pos]
        win_in_sl = [np.where(sl == w)[0][0] for w in ws if w in sl]
        pre.append((r_cod, r_txt, np.array(win_in_sl, dtype=np.int64)))
        # B0 (referencia)
        p_b0 = prior_b0.p_b0(r, emp).reindex(uei, fill_value=0.0).values
        rb = ranks(p_b0)
        bb = min((rb[w] for w in ws), default=np.inf)
        best_b0.append(bb)
    mrr_b0 = mrr_de(best_b0)

    def mrr_config(alpha, K):
        best = []
        for r_cod, r_txt, win_sl in pre:
            if len(win_sl) == 0:
                best.append(np.inf); continue
            s = (1 - alpha) / (K + r_cod) + alpha / (K + r_txt)
            rank_in_order = ranks(s)          # puesto final dentro del shortlist
            best.append(float(rank_in_order[win_sl].min()))
        return mrr_de(best)

    # --- barrido ---
    print(f"\nMRR del B0 (dev) = {mrr_b0:.3f}\n")
    print("Δ MRR vs B0 por (α filas, K columnas):")
    grid = pd.DataFrame(index=[f"α={al}" for al in ALPHAS],
                        columns=[f"K={k}" for k in KS], dtype=float)
    best = (-1, None, None)
    for al in ALPHAS:
        for K in KS:
            m = mrr_config(al, K)
            grid.loc[f"α={al}", f"K={K}"] = m - mrr_b0
            if m > best[0]:
                best = (m, al, K)
    with pd.option_context("display.float_format", "{:+.3f}".format, "display.width", 200):
        print(grid.to_string())
    print(f"\nmejor: α={best[1]}, K={best[2]}  →  MRR={best[0]:.3f}  (Δ={best[0]-mrr_b0:+.3f})")

    # --- evaluación oficial del mejor (α,K) con IC ---
    bal, bK = best[1], best[2]
    filas_b0, filas_rr = [], []
    for i, r in enumerate(val.itertuples()):
        r_cod, r_txt, _ = pre[i]
        p_v2 = prior_v2.p_b0(r, emp).reindex(uei, fill_value=0.0).values
        sl = np.argsort(-p_v2, kind="stable")[:TOP]
        s = (1 - bal) / (bK + r_cod) + bal / (bK + r_txt)
        ord_rr = sl[np.argsort(-s, kind="stable")[:100]]
        p_b0 = prior_b0.p_b0(r, emp).reindex(uei, fill_value=0.0).values
        ord_b0 = np.argsort(-p_b0, kind="stable")[:100]
        filas_rr.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ord_rr],
                                      "puesto": np.arange(1, len(ord_rr) + 1)}))
        filas_b0.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ord_b0],
                                      "puesto": np.arange(1, len(ord_b0) + 1)}))
    res = evaluar.evaluar(pd.concat(filas_rr, ignore_index=True), lic, emp, gan_dev,
                          pd.concat(filas_b0, ignore_index=True))
    t = res[res.conjunto == "todas"].iloc[0]
    print(f"\noficial (α={bal}, K={bK}): MRR={t.MRR:.3f}  Hit@5={t['Hit@5']:.3f}  "
          f"ΔMRR={t['Δ MRR']:+.3f} IC95={t['Δ MRR IC95']}  Hit@100={t['Hit@100']:.3f}")


if __name__ == "__main__":
    main()
