"""Tunear los 8 pesos de b0_v2 optimizando el MRR promedio en MÚLTIPLES semestres 
para TODAS las verticales de forma automatizada.

Uso:
    uv run --no-project --python 3.12 --with numpy --with pandas --with pyarrow \
        python tune_b0_v2_multisemestre_global.py --n_iter 1000
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
from b0_v2 import B0V2, NIVELES

VERTICALES = ['rd', 'j', 'd', 'r', 'it']


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
    ap.add_argument("--datos", default="data")
    ap.add_argument("--n_iter", type=int, default=1000, help="iteraciones de random search por vertical")
    a = ap.parse_args()

    d = pathlib.Path(a.datos)
    pesos_robustos_por_vertical = {}

    for vert in VERTICALES:
        print(f"\n==========================================")
        print(f"OPTIMIZACIÓN MULTI-SEMESTRE: {vert.upper()}")
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
        print(f"Semestres históricos disponibles: {sems_disponibles}")
        
        # Usamos todos los semestres anteriores (o una ventana robusta de entrenamiento)
        sems_dev = sems_disponibles[:-1] if len(sems_disponibles) > 1 else sems_disponibles
        print(f"Semestres seleccionados para validación multi-período: {sems_dev}")

        prior_v2 = B0V2(hist, off)
        emp_to_idx = prior_v2.idx

        print("Precalculando aportes por nivel para cada semestre...")
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

        if not datos_semestres:
            print(f"No hay suficientes datos históricos para {vert}.")
            continue

        PESOS_ORIG = np.array([0.40, 0.24, 0.12, 0.04, 0.20, 0.10, 0.06, 0.15], dtype=np.float64)

        def evaluar_pesos_global(pesos):
            mrr_por_semestre = []
            for sem, pre_sem in datos_semestres:
                best = []
                for niveles_aportes, ws in pre_sem:
                    if len(ws) == 0:
                        best.append(np.inf); continue
                    s = pesos @ niveles_aportes
                    best.append(float(ranks(s)[ws].min()))
                mrr_por_semestre.append(mrr_de(best))
            return float(np.mean(mrr_por_semestre))

        mrr_orig = evaluar_pesos_global(PESOS_ORIG)
        print(f"MRR promedio original ({vert}) = {mrr_orig:.3f}")

        print(f"Ejecutando Random Search ({a.n_iter} iteraciones)...")
        best_mrr = mrr_orig
        best_pesos = PESOS_ORIG.copy()

        rng = np.random.default_rng(42)
        for i in range(a.n_iter):
            p = rng.uniform(0.0, 1.0, size=len(PESOS_ORIG))
            p = p / p.sum()
            m = evaluar_pesos_global(p)
            if m > best_mrr:
                best_mrr = m
                best_pesos = p.copy()

        print(f"-> Mejor MRR Promedio robusto ({vert}): {best_mrr:.3f} (Δ = {best_mrr - mrr_orig:+.3f})")
        pesos_robustos_por_vertical[vert] = np.round(best_pesos, 3).tolist()

    print("\n==========================================")
    print("DICCIONARIO FINAL DE PESOS ROBUSTOS MULTI-SEMESTRE:")
    print("==========================================")
    import pprint
    pprint.pprint(pesos_robustos_por_vertical)


if __name__ == "__main__":
    main()