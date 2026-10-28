"""Visualiza cómo el B0 entierra ganadores.

Sobre una validación temporal (el último semestre, 2024S1), para cada licitación
calcula en qué puesto deja el B0 al ganador y arma cuatro paneles:

  (a) distribución del puesto del ganador (cuántos quedan en 1, 2–5, …, >100)
  (b) curva Hit@k acumulada (en qué fracción el ganador está en el top-k)
  (c) el mecanismo: ¿el ganador ya había ganado antes en ESA oficina / ESE PSC?
  (d) contratos previos del ganador vs. el puesto que le dio el B0

Uso:
    uv run python viz_b0.py                 # vertical j, guarda PNG y muestra
    uv run python viz_b0.py --vert rd       # otra vertical
    uv run python viz_b0.py --no-show       # solo guarda el PNG (sin ventana)
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import b0

ACENTO = "#d1495b"     # enterrados / lo que duele
AZUL = "#2e6f95"       # encontrados / el B0 bien
GRIS = "#b0b0b0"


def cargar(vert: str, datos: pathlib.Path):
    d = datos / vert
    lic = pd.read_parquet(d / "licitaciones.parquet")
    emp = pd.read_parquet(d / "empresas.parquet")
    hist = pd.read_parquet(d / "historial.parquet")
    gan = pd.read_parquet(d / "ganadores.parquet")
    return lic, emp, hist, gan


def puesto_del_ganador(lic, emp, hist, gan):
    """Para cada licitación de validación: puesto del mejor ganador bajo el B0,
    sus contratos previos, y si ya había ganado en esa oficina / ese PSC."""
    # validación temporal: el último semestre presente
    sem = f"{lic.t.max().year}S{1 if lic.t.max().month <= 6 else 2}"
    val = lic[lic.t >= pd.Timestamp(f"{lic.t.max().year}-01-01" if lic.t.max().month <= 6
                                    else f"{lic.t.max().year}-07-01")].copy()
    prior = b0.B0(hist)
    emp_sem = emp[emp.semestre == sem].set_index("parent_uei")
    win = gan[gan.id.isin(set(val.id))].groupby("id").parent_uei.apply(list).to_dict()

    h = hist.copy()
    h["psc"] = h.psc.astype(str).str.upper()
    h["office"] = h.office.astype(str).str.upper()
    h["fecha"] = pd.to_datetime(h.fecha)
    por_emp = {u: g for u, g in h.groupby("parent_uei")}

    def historia(uei, t, psc, office):
        g = por_emp.get(uei)
        if g is None:
            return 0, 0, 0
        g = g[g.fecha < t]
        return len(g), int((g.psc == str(psc).upper()).sum()), \
            int((g.office == str(office).upper()).sum())

    filas = []
    n_cand = (emp.semestre == sem).sum()
    for r in val.itertuples():
        p = prior.p_b0(r, emp).sort_values(ascending=False)
        order = list(p.index)
        ranks = [order.index(w) + 1 for w in win.get(r.id, []) if w in p.index]
        if not ranks:
            filas.append(dict(best=np.inf, prev=0, psc=0, off=0, n_aw=0))
            continue
        ws = win[r.id]
        w_best = ws[int(np.argmin([order.index(w) for w in ws]))]
        prev, psc_, off_ = historia(w_best, r.t, r.psc, r.office)
        n_aw = int(emp_sem.loc[w_best].n_awards) if w_best in emp_sem.index else 0
        filas.append(dict(best=min(ranks), prev=prev, psc=psc_, off=off_, n_aw=n_aw))
    return pd.DataFrame(filas), n_cand, len(val)


def plot(R: pd.DataFrame, n_cand: int, n_val: int, vert: str):
    best = R.best.values
    fin = np.isfinite(best)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"El B0 entierra ganadores — vertical '{vert}'  "
                 f"({n_val} licitaciones, {n_cand:,} candidatas c/u)",
                 fontsize=15, fontweight="bold")

    # (a) distribución por buckets
    labels = ["1", "2–5", "6–10", "11–50", "51–100", ">100\n/afuera"]
    rangos = [(1, 1), (2, 5), (6, 10), (11, 50), (51, 100)]
    cuentas = [int(((best >= lo) & (best <= hi)).sum()) for lo, hi in rangos]
    cuentas.append(int((~fin | (best > 100)).sum()))
    colores = [AZUL, AZUL, AZUL, GRIS, GRIS, ACENTO]
    a = ax[0, 0]
    barras = a.bar(labels, cuentas, color=colores)
    for b, c in zip(barras, cuentas):
        a.text(b.get_x() + b.get_width() / 2, c, f"{100*c/n_val:.0f}%",
               ha="center", va="bottom", fontsize=10)
    a.set_title("(a) ¿En qué puesto queda el ganador?")
    a.set_ylabel("licitaciones")
    med = int(np.median(best[fin]))
    a.annotate(f"puesto mediano: {med}", xy=(0.03, 0.95), xycoords="axes fraction",
               ha="left", fontsize=11, color=ACENTO, fontweight="bold")

    # (b) Hit@k acumulada
    ks = np.arange(1, 101)
    hit = [100 * np.mean(best <= k) for k in ks]
    b_ = ax[0, 1]
    b_.plot(ks, hit, color=AZUL, lw=2)
    b_.fill_between(ks, hit, color=AZUL, alpha=0.12)
    for k in (5, 100):
        b_.plot(k, 100 * np.mean(best <= k), "o", color=ACENTO)
        b_.annotate(f"Hit@{k} = {100*np.mean(best<=k):.0f}%",
                    xy=(k, 100 * np.mean(best <= k)),
                    xytext=(-5 if k == 100 else 5, -14 if k == 100 else 8),
                    textcoords="offset points",
                    ha="right" if k == 100 else "left", fontsize=10, color=ACENTO)
    b_.set_title("(b) Fracción con el ganador en el top-k")
    b_.set_xlabel("k (puesto)"); b_.set_ylabel("% de licitaciones")
    b_.set_ylim(0, 100); b_.grid(alpha=0.3)

    # (c) mecanismo: enterrados vs encontrados
    ent, top = R[best > 10], R[(best <= 10)]
    grupos = ["nunca ganó\nen ESA oficina", "nunca ganó\ncon ESE PSC"]
    v_ent = [100 * (ent.off == 0).mean(), 100 * (ent.psc == 0).mean()]
    v_top = [100 * (top.off == 0).mean(), 100 * (top.psc == 0).mean()]
    x = np.arange(len(grupos)); w = 0.38
    c = ax[1, 0]
    c.bar(x - w/2, v_ent, w, label=f"enterrados >10 (n={len(ent)})", color=ACENTO)
    c.bar(x + w/2, v_top, w, label=f"encontrados ≤10 (n={len(top)})", color=AZUL)
    for xi, (ve, vt) in enumerate(zip(v_ent, v_top)):
        c.text(xi - w/2, ve, f"{ve:.0f}%", ha="center", va="bottom", fontsize=10)
        c.text(xi + w/2, vt, f"{vt:.0f}%", ha="center", va="bottom", fontsize=10)
    c.set_xticks(x); c.set_xticklabels(grupos)
    c.set_title("(c) Por qué los entierra"); c.set_ylabel("% de ganadores")
    c.set_ylim(0, 100); c.legend(fontsize=9)

    # (d) contratos previos vs puesto
    d = ax[1, 1]
    m = fin
    xj = R.prev.values[m] + 1          # +1 para log
    yj = best[m]
    es_new = yj > 10
    d.scatter(xj[~es_new], yj[~es_new], s=14, alpha=0.5, color=AZUL, label="S_inc (≤10)")
    d.scatter(xj[es_new], yj[es_new], s=14, alpha=0.5, color=ACENTO, label="S_new (>10)")
    d.set_xscale("log"); d.set_yscale("log")
    d.axhline(10, color="k", ls="--", lw=0.8, alpha=0.6)
    d.set_title("(d) Contratos previos del ganador vs. su puesto")
    d.set_xlabel("contratos previos en la familia (+1, log)")
    d.set_ylabel("puesto bajo el B0 (log)")
    d.legend(fontsize=9); d.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vert", default="j", help="vertical: rd, d, j, r, it")
    ap.add_argument("--datos", default="data", help="carpeta de datos")
    ap.add_argument("--out", help="PNG de salida (por defecto b0_enterrados_<vert>.png)")
    ap.add_argument("--no-show", action="store_true", help="no abrir la ventana")
    a = ap.parse_args()

    R, n_cand, n_val = puesto_del_ganador(*cargar(a.vert, pathlib.Path(a.datos)))
    fig = plot(R, n_cand, n_val, a.vert)
    out = a.out or f"b0_enterrados_{a.vert}.png"
    fig.savefig(out, dpi=130)
    print(f"figura -> {out}")
    if not a.no_show:
        plt.show()


if __name__ == "__main__":
    main()
