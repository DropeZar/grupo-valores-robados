"""Cómo el B0 (y nuestro B0-v2) entierra ganadores, en TODAS las verticales y semestres.

Para cada (vertical × semestre × modelo) genera el mismo gráfico de 4 paneles que viz_b0.py
(distribución de puestos, curva Hit@k, mecanismo, contratos previos vs. puesto). Además:
  - overview_hitk_<modelo>.png : la curva Hit@k de todos los semestres, un panel por vertical.
  - overview_recall.png        : Hit@100 por vertical y semestre, B0 vs v2 de frente
                                 (solo cuando se piden los dos modelos).

Modelos:  b0 = baseline original (b0.py) ; v2 = con backoff de códigos (b0_v2.py).

Uso (tarda unos minutos; con 'both' es ~el doble):

    uv run python viz_b0_multi.py                       # both, todas las verticales
    uv run python viz_b0_multi.py --modelo v2           # solo nuestro v2
    uv run python viz_b0_multi.py --vert j,rd --no-paneles   # rápido: solo resúmenes
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import b0
from b0_v2 import B0V2

ACENTO, AZUL, GRIS = "#d1495b", "#2e6f95", "#b0b0b0"
VERTS = ["rd", "d", "j", "r", "it"]
COLOR_MODELO = {"b0": AZUL, "v2": ACENTO}


def semestre_col(t):
    return t.dt.year.astype(str) + "S" + (t.dt.month > 6).map({False: "1", True: "2"})


def preparar(vert, datos, off, modelos):
    """Carga una vertical y arma el/los prior/s pedidos (una sola vez por vertical)."""
    d = datos / vert
    lic = pd.read_parquet(d / "licitaciones.parquet")
    emp = pd.read_parquet(d / "empresas.parquet")
    hist = pd.read_parquet(d / "historial.parquet")
    gan = pd.read_parquet(d / "ganadores.parquet")
    lic["sem"] = semestre_col(lic.t)

    priors = {}
    if "b0" in modelos:
        priors["b0"] = b0.B0(hist)
    if "v2" in modelos:
        priors["v2"] = B0V2(hist, off)

    h = hist.copy()
    h["psc"] = h.psc.astype(str).str.upper()
    h["office"] = h.office.astype(str).str.upper()
    h["fecha"] = pd.to_datetime(h.fecha)
    por_emp = {u: g for u, g in h.groupby("parent_uei")}
    return lic, emp, gan, priors, por_emp


def analizar(lic, emp, gan, prior, por_emp, sem):
    """Para las licitaciones de un semestre: puesto del ganador bajo `prior` + mecanismo."""
    val = lic[lic["sem"] == sem]
    n_cand = int((emp.semestre == sem).sum())
    if len(val) == 0 or n_cand == 0:
        return None, 0, 0
    win = gan[gan.id.isin(set(val.id))].groupby("id").parent_uei.apply(list).to_dict()
    recs = []
    for r in val.itertuples():
        p = prior.p_b0(r, emp).sort_values(ascending=False)
        order = list(p.index)
        ws = win.get(r.id, [])
        ranks = [order.index(w) + 1 for w in ws if w in p.index]
        if not ranks:
            recs.append(dict(best=np.inf, prev=0, psc=0, off=0))
            continue
        w = ws[int(np.argmin([order.index(x) for x in ws]))]
        g = por_emp.get(w)
        g = g[g.fecha < r.t] if g is not None else None
        if g is None or len(g) == 0:
            recs.append(dict(best=min(ranks), prev=0, psc=0, off=0))
        else:
            recs.append(dict(best=min(ranks), prev=len(g),
                             psc=int((g.psc == str(r.psc).upper()).sum()),
                             off=int((g.office == str(r.office).upper()).sum())))
    return pd.DataFrame(recs), n_cand, len(val)


def plot4(R, n_cand, n_val, vert, sem, modelo):
    best = R.best.values
    fin = np.isfinite(best)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{modelo.upper()} entierra ganadores — '{vert}' · {sem}  "
                 f"({n_val} licitaciones, {n_cand:,} candidatas c/u)",
                 fontsize=15, fontweight="bold")

    labels = ["1", "2–5", "6–10", "11–50", "51–100", ">100\n/afuera"]
    rangos = [(1, 1), (2, 5), (6, 10), (11, 50), (51, 100)]
    cuentas = [int(((best >= lo) & (best <= hi)).sum()) for lo, hi in rangos]
    cuentas.append(int((~fin | (best > 100)).sum()))
    a = ax[0, 0]
    for b_, c, col in zip(a.bar(labels, cuentas), cuentas,
                          [AZUL, AZUL, AZUL, GRIS, GRIS, ACENTO]):
        b_.set_color(col)
        a.text(b_.get_x() + b_.get_width() / 2, c, f"{100*c/n_val:.0f}%",
               ha="center", va="bottom", fontsize=10)
    a.set_title("(a) ¿En qué puesto queda el ganador?"); a.set_ylabel("licitaciones")
    a.annotate(f"puesto mediano: {int(np.median(best[fin])) if fin.any() else '—'}",
               xy=(0.03, 0.95), xycoords="axes fraction", ha="left",
               fontsize=11, color=ACENTO, fontweight="bold")

    ks = np.arange(1, 101)
    hit = [100 * np.mean(best <= k) for k in ks]
    b2 = ax[0, 1]
    b2.plot(ks, hit, color=AZUL, lw=2); b2.fill_between(ks, hit, color=AZUL, alpha=0.12)
    for k in (5, 100):
        y = 100 * np.mean(best <= k)
        b2.plot(k, y, "o", color=ACENTO)
        b2.annotate(f"Hit@{k} = {y:.0f}%", xy=(k, y),
                    xytext=(-5 if k == 100 else 5, -14 if k == 100 else 8),
                    textcoords="offset points", ha="right" if k == 100 else "left",
                    fontsize=10, color=ACENTO)
    b2.set_title("(b) Fracción con el ganador en el top-k")
    b2.set_xlabel("k (puesto)"); b2.set_ylabel("% de licitaciones")
    b2.set_ylim(0, 100); b2.grid(alpha=0.3)

    ent, top = R[best > 10], R[best <= 10]
    grupos = ["nunca ganó\nen ESA oficina", "nunca ganó\ncon ESE PSC"]
    v_ent = [100 * (ent.off == 0).mean(), 100 * (ent.psc == 0).mean()] if len(ent) else [0, 0]
    v_top = [100 * (top.off == 0).mean(), 100 * (top.psc == 0).mean()] if len(top) else [0, 0]
    x = np.arange(2); w = 0.38; c = ax[1, 0]
    c.bar(x - w/2, v_ent, w, label=f"enterrados >10 (n={len(ent)})", color=ACENTO)
    c.bar(x + w/2, v_top, w, label=f"encontrados ≤10 (n={len(top)})", color=AZUL)
    for xi, (ve, vt) in enumerate(zip(v_ent, v_top)):
        c.text(xi - w/2, ve, f"{ve:.0f}%", ha="center", va="bottom", fontsize=10)
        c.text(xi + w/2, vt, f"{vt:.0f}%", ha="center", va="bottom", fontsize=10)
    c.set_xticks(x); c.set_xticklabels(grupos); c.set_ylim(0, 100)
    c.set_title("(c) Por qué los entierra"); c.set_ylabel("% de ganadores"); c.legend(fontsize=9)

    d = ax[1, 1]; m = fin
    xj, yj = R.prev.values[m] + 1, best[m]
    es_new = yj > 10
    d.scatter(xj[~es_new], yj[~es_new], s=14, alpha=0.5, color=AZUL, label="S_inc (≤10)")
    d.scatter(xj[es_new], yj[es_new], s=14, alpha=0.5, color=ACENTO, label="S_new (>10)")
    d.set_xscale("log"); d.set_yscale("log"); d.axhline(10, color="k", ls="--", lw=0.8, alpha=0.6)
    d.set_title("(d) Contratos previos del ganador vs. su puesto")
    d.set_xlabel("contratos previos (+1, log)"); d.set_ylabel(f"puesto bajo {modelo.upper()} (log)")
    d.legend(fontsize=9); d.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def overview_hitk(bests_modelo, modelo, out):
    """Curva Hit@k por semestre, un panel por vertical (un modelo)."""
    verts = [v for v in VERTS if v in bests_modelo]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ks = np.arange(1, 101)
    for ax, v in zip(axes.flat, verts):
        sems = sorted(bests_modelo[v])
        cols = plt.cm.viridis(np.linspace(0, 0.85, len(sems)))
        for sem, col in zip(sems, cols):
            best = bests_modelo[v][sem]
            ax.plot(ks, [100 * np.mean(best <= k) for k in ks], color=col, lw=1.8,
                    label=f"{sem} (Hit@100={100*np.mean(best<=100):.0f}%)")
        ax.set_title(f"vertical '{v}'", fontweight="bold")
        ax.set_xlabel("k"); ax.set_ylabel("% con ganador en top-k")
        ax.set_ylim(0, 100); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    for ax in axes.flat[len(verts):]:
        ax.axis("off")
    fig.suptitle(f"Techo de {modelo.upper()} (Hit@k) por vertical y semestre",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"resumen -> {out}")


def overview_recall(resumen, out):
    """Hit@100 por vertical y semestre, una línea por modelo (B0 vs v2)."""
    df = pd.DataFrame(resumen)
    verts = [v for v in VERTS if v in set(df.vert)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, v in zip(axes.flat, verts):
        sub = df[df.vert == v]
        for mname, g in sub.groupby("modelo"):
            g = g.sort_values("sem")
            ax.plot(g["sem"], g["hit100"], marker="o", lw=2,
                    color=COLOR_MODELO.get(mname), label=mname.upper())
        ax.set_title(f"vertical '{v}'", fontweight="bold")
        ax.set_ylabel("Hit@100 (%)"); ax.set_ylim(0, 100); ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=45); ax.legend(fontsize=9)
    for ax in axes.flat[len(verts):]:
        ax.axis("off")
    fig.suptitle("Recall (Hit@100) por vertical y semestre — B0 vs v2",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"resumen -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modelo", default="both", choices=["b0", "v2", "both"])
    ap.add_argument("--vert", default="all", help="all, o lista: j,rd,it")
    ap.add_argument("--sem", default="all", help="all, o lista: 2023S2,2024S1")
    ap.add_argument("--datos", default="data")
    ap.add_argument("--out", default="figuras")
    ap.add_argument("--no-paneles", action="store_true", help="solo las figuras-resumen")
    a = ap.parse_args()

    datos = pathlib.Path(a.datos)
    out = pathlib.Path(a.out); out.mkdir(exist_ok=True)
    off = pd.read_parquet(datos / "office.parquet")
    verts = VERTS if a.vert == "all" else [v.strip() for v in a.vert.split(",")]
    sems_ped = None if a.sem == "all" else {s.strip() for s in a.sem.split(",")}
    modelos = ["b0", "v2"] if a.modelo == "both" else [a.modelo]

    bests = {m: {} for m in modelos}
    resumen = []
    for v in verts:
        print(f"\n### vertical {v} — preparando {', '.join(m.upper() for m in modelos)}…")
        lic, emp, gan, priors, por_emp = preparar(v, datos, off, modelos)
        sems = sorted(lic["sem"].unique())
        if sems_ped:
            sems = [s for s in sems if s in sems_ped]
        for m in modelos:
            bests[m][v] = {}
        for sem in sems:
            for m in modelos:
                R, n_cand, n_val = analizar(lic, emp, gan, priors[m], por_emp, sem)
                if R is None:
                    continue
                best = R.best.values
                fin = np.isfinite(best)
                bests[m][v][sem] = best
                resumen.append(dict(vert=v, sem=sem, modelo=m, N=n_val,
                                    hit100=round(100 * np.mean(best <= 100), 1),
                                    mediana=int(np.median(best[fin])) if fin.any() else -1))
                print(f"   {sem} [{m}]: {n_val:5d} lic | "
                      f"Hit@100={100*np.mean(best<=100):4.1f}% | mediana={resumen[-1]['mediana']}")
                if not a.no_paneles:
                    fig = plot4(R, n_cand, n_val, v, sem, m)
                    fig.savefig(out / f"b0_enterrados_{v}_{sem}_{m}.png", dpi=120)
                    plt.close(fig)
        for m in modelos:
            if not bests[m][v]:
                del bests[m][v]

    for m in modelos:
        if bests[m]:
            overview_hitk(bests[m], m, out / f"overview_hitk_{m}.png")
    if len(modelos) > 1 and resumen:
        overview_recall(resumen, out / "overview_recall.png")

    print("\n== resumen ==")
    with pd.option_context("display.width", 200):
        print(pd.DataFrame(resumen).sort_values(["vert", "sem", "modelo"]).to_string(index=False))


if __name__ == "__main__":
    main()
