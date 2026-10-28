"""Barrido acotado (Grid Search) de tau para encontrar la 'memoria' temporal óptima 
por vertical utilizando validación multi-semestre.

Uso:
    uv run --no-project --python 3.12 --with numpy --with pandas --with pyarrow \
        python grid_search_tau.py
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
import b0_v2_tuned
from b0_v2_tuned import B0V2, NIVELES, PESOS_POR_VERTICAL, PESOS_DEFAULT

VERTICALES = ['rd', 'j', 'd', 'r', 'it']
CANDIDATOS_TAU = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def ranks(scores):
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.float64)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def mrr_de(best_ranks):
    return float(np.mean([1.0 / b if b <= 100 else 0.0 for b in best_ranks]))


def semestre_col(t):
    return t.dt.year.astype(str) + "S" + (t.dt.month > 6).map({False: "1", True: "2"})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vert", default=None, help="Vertical específica o dejar vacío para todas")
    ap.add_argument("--datos", default="data")
    a = ap.parse_args()

    d = pathlib.Path(a.datos)
    verts = [a.vert] if a.vert else VERTICALES

    mejores_taus = {}

    for vert in verts:
        print(f"\n==========================================")
        print(f"BARRIDO DE TAU PARA VERTICAL: {vert.upper()}")
        print(f"==========================================")
        
        try:
            lic = pd.read_parquet(d / vert / "licitaciones.parquet")
            emp = pd.read_parquet(d / vert / "empresas.parquet")
            hist = pd.read_parquet(d / vert / "historial.parquet")
            gan = pd.read_parquet(d / vert / "ganadores.parquet")
            off = pd.read_parquet(d / "office.parquet")
        except Exception as e:
            print(f"No se pudieron cargar los datos de {vert}: {e}")
            continue

        lic["sem"] = semestre_col(lic.t)
        sems_disponibles = sorted(lic["sem"].unique())
        sems_dev = sems_disponibles[:-1] if len(sems_disponibles) > 1 else sems_disponibles
        print(f"Semestres de validación multi-período: {sems_dev}")

        pesos_vert = PESOS_POR_VERTICAL.get(vert, PESOS_DEFAULT)

        mejor_tau = 2.5
        mejor_mrr = -1.0

        for tau in CANDIDATOS_TAU:
            # Actualizamos tau en ambos módulos para que B0V2 tome el cambio
            b0.TAU_ANIOS = tau
            b0_v2_tuned.TAU_ANIOS = tau

            prior_v2 = B0V2(hist, off, vert=vert)
            emp_to_idx = prior_v2.idx

            datos_semestres = []
            for sem in sems_dev:
                val_sem = lic[lic["sem"] == sem].copy()
                cand_sem = emp[emp.semestre == sem].reset_index(drop=True)
                if len(val_sem) == 0 or len(cand_sem) == 0:
                    continue
                
                uei = cand_sem.parent_uei.values
                uei_pos = {u: i for i, u in enumerate(uei)}
                gan_sem = gan[gan.id.isin(set(val_sem.id))]
                win_sem = gan_sem.groupby("id").parent_uei.apply(list).to_dict()

                pre_sem = []
                for r in val_sem.itertuples():
                    ts = np.datetime64(pd.Timestamp(r.t), "D").astype(np.int64)
                    psc, office = str(r.psc).upper(), str(r.office).upper()
                    ctx = {"psc": psc, "office": office, "naics": str(r.naics),
                           "sa": b0.set_aside(r.set_aside), "fam": psc[:2],
                           "sub": prior_v2.sub_de.get(office, "")}
                    
                    niveles_aportes = np.zeros((len(NIVELES), len(uei)), dtype=np.float64)
                    for idx_niv, (cols, tabla) in enumerate(zip(NIVELES, prior_v2.tablas)):
                        key = tuple(ctx[c] for c in cols)
                        if "" in key: continue
                        e = tabla.get(key)
                        if e is None: continue
                        t_arr, fi, w = e
                        n = np.searchsorted(t_arr, ts, side="left")
                        if n == 0: continue
                        v = np.bincount(fi[:n], weights=w[:n], minlength=len(prior_v2.empresas))
                        if v.sum() == 0: continue
                        v_norm = v / v.sum()
                        
                        arr_uei = np.zeros(len(uei), dtype=np.float64)
                        for j, u in enumerate(uei):
                            emp_idx = emp_to_idx.get(u)
                            if emp_idx is not None:
                                arr_uei[j] = v_norm[emp_idx]
                        niveles_aportes[idx_niv] = arr_uei

                    ws = [uei_pos[w] for w in win_sem.get(r.id, []) if w in uei_pos]
                    pre_sem.append((niveles_aportes, np.array(ws, dtype=np.int64)))
                
                datos_semestres.append((sem, pre_sem))

            mrr_por_semestre = []
            for sem, pre_sem in datos_semestres:
                best = []
                for niveles_aportes, ws in pre_sem:
                    if len(ws) == 0:
                        best.append(np.inf); continue
                    s = np.array(pesos_vert) @ niveles_aportes
                    best.append(float(ranks(s)[ws].min()))
                mrr_por_semestre.append(mrr_de(best))
            
            mrr_medio = float(np.mean(mrr_por_semestre)) if mrr_por_semestre else 0.0
            print(f"  -> Tau = {tau} | MRR Promedio Multi-Semestre = {mrr_medio:.5f}")

            if mrr_medio > mejor_mrr:
                mejor_mrr = mrr_medio
                mejor_tau = tau

        print(f"\n★ MEJOR TAU PARA {vert.upper()}: {mejor_tau} (MRR: {mejor_mrr:.5f})\n")
        mejores_taus[vert] = mejor_tau

    print("==========================================")
    print("RESUMEN DE TAUS ÓPTIMOS POR VERTICAL:")
    print("==========================================")
    print(mejores_taus)


if __name__ == "__main__":
    main()