"""Tunea los pesos de los niveles de incumbencia del B0-v2 en dev, y compara contra los
pesos dados. Responde: ¿los pesos del B0 son mejorables?

Las distribuciones por nivel NO dependen de los pesos, así que se precalculan una vez por
licitación y el barrido de pesos es solo recombinar (rápido). Objetivo: MRR del B0-v2 en dev.

    PYTHONPATH="$PWD" uv run --no-project --python 3.12 --with numpy --with pandas \
        --with pyarrow --with scipy python tune_pesos.py --vert j --dev 2023S2
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
from b0_v2 import B0V2, NIVELES, PESOS, set_aside

NOMBRES = ["psc+office", "psc", "naics", "global", "psc+sa", "fam+office", "fam", "psc+sub"]


def niveles_de(prior, lic, cand_idx):
    """Matriz (8, n_cand): la distribución normalizada de cada nivel sobre las candidatas,
    SIN el peso. Replica b0_v2._p pero junta los niveles por separado."""
    ts = np.datetime64(pd.Timestamp(lic.t), "D").astype(np.int64)
    psc, office = str(lic.psc).upper(), str(lic.office).upper()
    ctx = {"psc": psc, "office": office, "naics": str(lic.naics),
           "sa": set_aside(lic.set_aside), "fam": psc[:2],
           "sub": prior.sub_de.get(office, "")}
    M = len(prior.empresas)
    out = np.zeros((len(NIVELES), len(cand_idx)), dtype=np.float64)
    for k, (cols, tabla) in enumerate(zip(NIVELES, prior.tablas)):
        key = tuple(ctx[c] for c in cols)
        if "" in key:
            continue
        e = tabla.get(key)
        if e is None:
            continue
        t_arr, fi, w = e
        n = np.searchsorted(t_arr, ts, side="left")
        if n == 0:
            continue
        v = np.bincount(fi[:n], weights=w[:n], minlength=M)
        s = v.sum()
        if s > 0:
            out[k] = (v / s)[cand_idx]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vert", default="j")
    ap.add_argument("--dev", default="2023S2")
    ap.add_argument("--datos", default="data")
    ap.add_argument("--iters", type=int, default=400)
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
    gan_dev = gan[gan.id.isin(set(val.id))]
    win = gan_dev.groupby("id").parent_uei.apply(list).to_dict()

    prior = B0V2(hist, off)
    cand_idx = np.array([prior.idx.get(u, 0) for u in uei])
    pos = {u: i for i, u in enumerate(uei)}

    print(f"[{a.vert}] dev {a.dev}: {len(val)} licitaciones, {len(cand):,} candidatas")
    print("precalculando niveles por licitación…")
    niveles, win_idx = [], []
    for r in val.itertuples():
        niveles.append(niveles_de(prior, r, cand_idx).astype(np.float32))
        win_idx.append(np.array([pos[w] for w in win.get(r.id, []) if w in pos], dtype=np.int64))

    def mrr(w):
        w = np.asarray(w, dtype=np.float64)
        tot = 0.0
        for L, wi in zip(niveles, win_idx):
            if len(wi) == 0:
                continue
            score = w @ L                        # (n_cand,)
            best = min(1 + int(np.sum(score > score[j])) for j in wi)
            tot += 1.0 / best if best <= 100 else 0.0
        return tot / len(niveles)

    dados = np.array(PESOS)
    mrr_dados = mrr(dados)
    print(f"\nMRR del B0-v2 con pesos DADOS {list(dados)}: {mrr_dados:.4f}")

    # búsqueda: random (Dirichlet) + refinamiento por coordenadas desde el mejor
    rng = np.random.default_rng(0)
    best_w, best_m = dados.copy(), mrr_dados
    print(f"\nbuscando ({a.iters} muestras aleatorias)…")
    for _ in range(a.iters):
        w = rng.dirichlet(np.ones(len(NIVELES)))      # pesos ≥0 que suman 1
        m = mrr(w)
        if m > best_m:
            best_m, best_w = m, w.copy()
    # refinamiento por coordenadas
    for _ in range(3):
        for k in range(len(NIVELES)):
            for mult in (0.0, 0.5, 0.8, 1.25, 1.5, 2.0):
                w = best_w.copy(); w[k] *= mult
                m = mrr(w)
                if m > best_m:
                    best_m, best_w = m, w.copy()

    bw = best_w / best_w.sum()
    print(f"\n== mejor encontrado ==")
    print(f"MRR dados : {mrr_dados:.4f}")
    print(f"MRR tuneado: {best_m:.4f}   (Δ = {best_m - mrr_dados:+.4f})")
    print("\npesos (normalizados a suma 1):")
    print(f"  {'nivel':12s} {'dado':>7s} {'tuneado':>8s}")
    dn = dados / dados.sum()
    for nom, pd_, pt in zip(NOMBRES, dn, bw):
        print(f"  {nom:12s} {pd_:7.3f} {pt:8.3f}")


if __name__ == "__main__":
    main()
