"""Paso 2: el dial de dificultad. El peso del texto α varía por licitación según cuán
predecible es (capacidad vs. dificultad).

    dificultad(q) = entropía normalizada del reparto de ganadores en el contexto de q
                    (oficina+PSC, con fecha < t; backoff a oficina; 0.5 si no hay datos)
    α(q)          = α_lo + (α_hi − α_lo) · dificultad(q)
    s(q,e)        = (1 − α(q))/(K + r_cod)  +  α(q)/(K + r_txt)     (rerank del top-200 del v2)

Barre (α_lo, α_hi) en dev y compara contra el α fijo (que es la celda α_lo=α_hi). Usa los
embeddings cacheados por latente_emb.py.

    uv run --no-project --python 3.12 --with numpy --with pandas --with pyarrow \
        --with scikit-learn python dial_dificultad.py --vert j --dev 2023S2
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
K = 5                       # el mejor del Paso 1
MIN_CTX = 8                 # mínimo de contratos para confiar en la entropía de un contexto
ALO = [0.2, 0.3, 0.4, 0.5]
AHI = [0.4, 0.5, 0.6, 0.7, 0.8]


def ranks(scores):
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.float64)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def mrr_de(best):
    return float(np.mean([1.0 / b if b <= 100 else 0.0 for b in best]))


def entropia_norm(ueis):
    """Entropía de Shannon normalizada (0..1) del reparto de ganadores. 0 = una domina, 1 = parejo."""
    _, c = np.unique(ueis, return_counts=True)
    if len(c) <= 1:
        return 0.0
    p = c / c.sum()
    H = -np.sum(p * np.log(p))
    return float(H / np.log(len(c)))


def dificultad_v2(p_v2, k=20):
    """Entropía de Shannon normalizada del TOP-k de p_v2. Mide cuán confiado está el v2:
    0 = un favorito claro (fácil), 1 = top chato sin favorito (difícil)."""
    top = np.sort(p_v2)[::-1][:k]
    top = top[top > 0]
    if len(top) <= 1:
        return 0.0
    q = top / top.sum()
    H = -np.sum(q * np.log(q))
    return float(H / np.log(len(top)))


def dificultad_renyi(p_v2, q=2.0):
    """Entropía de Rényi de orden q, normalizada, sobre TODA la distribución p_v2.
    H_q = 1/(1-q) · log(Σ p^q). q=1 → Shannon; q=2 → colisión; q grande → sólo el top.
    Pesa por p^q, así que a mayor q más se enfoca en los dominantes. 0 = uno domina, 1 = parejo."""
    p = p_v2[p_v2 > 0].astype(np.float64)
    if len(p) <= 1:
        return 0.0
    p = p / p.sum()
    if abs(q - 1.0) < 1e-9:
        H = -np.sum(p * np.log(p))
    else:
        H = np.log(np.sum(p ** q)) / (1.0 - q)
    return float(H / np.log(len(p)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vert", default="j")
    ap.add_argument("--dev", default="2023S2")
    ap.add_argument("--datos", default="data")
    ap.add_argument("--fuente", default="v2", choices=["v2", "hist", "renyi"],
                    help="dificultad: v2 (Shannon top-k de p_v2), renyi (Rényi orden q de p_v2), hist (historial)")
    ap.add_argument("--topk", type=int, default=20, help="top-k de p_v2 para la entropía (fuente v2)")
    ap.add_argument("--orden", type=float, default=2.0, help="orden q de Rényi (fuente renyi)")
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
        sys.exit("faltan embeddings: corré primero latente_emb.py")
    from sklearn.preprocessing import normalize
    C = normalize(np.load(f_emp)); Q = normalize(np.load(f_lic))
    print(f"vertical {a.vert} | dev {a.dev}: {len(val)} licitaciones, {len(cand):,} candidatas")

    prior_b0 = b0.B0(hist)
    prior_v2 = B0V2(hist, off)

    # historial agrupado por contexto, ordenado por fecha, para la entropía
    h = hist.copy()
    h["office"] = h.office.astype(str).str.upper()
    h["psc"] = h.psc.astype(str).str.upper()
    h["fecha"] = pd.to_datetime(h.fecha)
    h = h.sort_values("fecha")
    by_op = {k: (g.fecha.values, g.parent_uei.values)
             for k, g in h.groupby(["office", "psc"], sort=False)}
    by_o = {k: (g.fecha.values, g.parent_uei.values)
            for k, g in h.groupby("office", sort=False)}

    def dificultad(office, psc, t):
        tt = np.datetime64(pd.Timestamp(t))
        for tabla, key in ((by_op, (office, psc)), (by_o, office)):
            e = tabla.get(key)
            if e is None:
                continue
            n = int(np.searchsorted(e[0], tt, side="left"))
            if n >= MIN_CTX:
                return entropia_norm(e[1][:n]), ("op" if tabla is by_op else "of")
        return 0.5, "sininfo"

    # precálculo por licitación: shortlist, r_cod, r_txt, ganadores, dificultad
    print("precalculando…")
    pre, best_b0, difs, fuentes = [], [], [], []
    for i, r in enumerate(val.itertuples()):
        p_v2 = prior_v2.p_b0(r, emp).reindex(uei, fill_value=0.0).values
        sl = np.argsort(-p_v2, kind="stable")[:TOP]
        r_txt = ranks(Q[i] @ C[sl].T)
        r_cod = np.arange(1, len(sl) + 1, dtype=np.float64)
        ws = [uei_pos[w] for w in win.get(r.id, []) if w in uei_pos]
        win_sl = np.array([np.where(sl == w)[0][0] for w in ws if w in sl], dtype=np.int64)
        if a.fuente == "v2":
            dif, fte = dificultad_v2(p_v2, a.topk), "v2"
        elif a.fuente == "renyi":
            dif, fte = dificultad_renyi(p_v2, a.orden), f"renyi{a.orden:g}"
        else:
            dif, fte = dificultad(str(r.office).upper(), str(r.psc).upper(), r.t)
        pre.append((r_cod, r_txt, win_sl, dif))
        difs.append(dif); fuentes.append(fte)
        p_b0 = prior_b0.p_b0(r, emp).reindex(uei, fill_value=0.0).values
        rb = ranks(p_b0)
        best_b0.append(min((rb[w] for w in ws), default=np.inf))
    mrr_b0 = mrr_de(best_b0)

    difs = np.array(difs)
    print(f"\ndificultad: min={difs.min():.2f} mediana={np.median(difs):.2f} max={difs.max():.2f}")
    print("fuente del contexto:", pd.Series(fuentes).value_counts().to_dict())
    print(f"MRR del B0 (dev) = {mrr_b0:.3f}")

    def mrr_config(alo, ahi):
        best = []
        for r_cod, r_txt, win_sl, dif in pre:
            if len(win_sl) == 0:
                best.append(np.inf); continue
            al = alo + (ahi - alo) * dif
            s = (1 - al) / (K + r_cod) + al / (K + r_txt)
            best.append(float(ranks(s)[win_sl].min()))
        return mrr_de(best)

    # --- barrido ---
    print("\nΔ MRR vs B0 por (α_lo filas, α_hi columnas):")
    grid = pd.DataFrame(index=[f"lo={x}" for x in ALO],
                        columns=[f"hi={x}" for x in AHI], dtype=float)
    best = (-1, None, None)
    for alo in ALO:
        for ahi in AHI:
            if ahi < alo:
                continue
            m = mrr_config(alo, ahi)
            grid.loc[f"lo={alo}", f"hi={ahi}"] = m - mrr_b0
            if m > best[0]:
                best = (m, alo, ahi)
    with pd.option_context("display.float_format", lambda v: "  .  " if v != v else f"{v:+.3f}"):
        print(grid.to_string())
    fijo = mrr_config(0.5, 0.5) - mrr_b0
    print(f"\nreferencia α fijo (lo=hi=0.5): Δ={fijo:+.3f}")
    print(f"mejor dial: α_lo={best[1]}, α_hi={best[2]}  →  Δ={best[0]-mrr_b0:+.3f}")

    # --- evaluación oficial del mejor, con split S_inc/S_new ---
    alo, ahi = best[1], best[2]
    f_rr, f_b0 = [], []
    for i, r in enumerate(val.itertuples()):
        r_cod, r_txt, _, dif = pre[i]
        p_v2 = prior_v2.p_b0(r, emp).reindex(uei, fill_value=0.0).values
        sl = np.argsort(-p_v2, kind="stable")[:TOP]
        al = alo + (ahi - alo) * dif
        s = (1 - al) / (K + r_cod) + al / (K + r_txt)
        ord_rr = sl[np.argsort(-s, kind="stable")[:100]]
        ord_b0 = np.argsort(-prior_b0.p_b0(r, emp).reindex(uei, fill_value=0.0).values,
                            kind="stable")[:100]
        f_rr.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ord_rr],
                                  "puesto": np.arange(1, 101)}))
        f_b0.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ord_b0],
                                  "puesto": np.arange(1, 101)}))
    res = evaluar.evaluar(pd.concat(f_rr, ignore_index=True), lic, emp, gan_dev,
                          pd.concat(f_b0, ignore_index=True))
    print(f"\noficial (dial α_lo={alo}, α_hi={ahi}, K={K}):")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(res[["conjunto", "N", "MRR", "Hit@5", "Δ MRR", "Δ MRR IC95"]].to_string(index=False))


if __name__ == "__main__":
    main()
