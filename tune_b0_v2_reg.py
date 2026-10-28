"""Optimización de pesos con Regularización L2 (Shrinkage) hacia el baseline b0_v2.
Especialmente útil para estabilizar verticales pequeñas o volátiles como 'd' y 'rd'.

Uso:
    uv run --no-project --python 3.12 --with numpy --with pandas --with pyarrow \
        python tune_regularizado.py --vert rd --lambda_reg 0.05 --n_iter 1500
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

PESOS_ORIG = np.array([0.40, 0.24, 0.12, 0.04, 0.20, 0.10, 0.06, 0.15], dtype=np.float64)


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
    ap.add_argument("--vert", default="rd", help="Vertical a estabilizar (ej: d, rd)")
    ap.add_argument("--datos", default="data")
    ap.add_argument("--lambda_reg", type=float, default=0.08, help="Fuerza de penalización L2 hacia el baseline")
    ap.add_argument("--n_iter", type=int, default=1500, help="Iteraciones de random search")
    a = ap.parse_args()

    d = pathlib.Path(a.datos)
    vert = a.vert

    print(f"\n==========================================")
    print(f"OPTIMIZACIÓN REGULARIZADA PARA: {vert.upper()}")
    print(f"Lambda de penalización L2 (λ) = {a.lambda_reg}")
    print(f"==========================================")

    lic = pd.read_parquet(d / vert / "licitaciones.parquet")
    emp = pd.read_parquet(d / vert / "empresas.parquet")
    hist = pd.read_parquet(d / vert / "historial.parquet")
    gan = pd.read_parquet(d / vert / "ganadores.parquet")
    off = pd.read_parquet(d / "office.parquet")

    lic["sem"] = semestre_col(lic.t)
    sems_disponibles = sorted(lic["sem"].unique())
    print(f"Semestres históricos disponibles: {sems_disponibles}")
    
    # Usar todos los semestres anteriores para entrenamiento robusto multi-período
    sems_dev = sems_disponibles[:-1] if len(sems_disponibles) > 1 else sems_disponibles
    print(f"Semestres de validación cruzada temporal: {sems_dev}")

    prior_v2 = B0V2(hist, off)
    emp_to_idx = prior_v2.idx

    print("Precalculando aportes por nivel...")
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

    def evaluar_pesos_regularizados(pesos):
        mrr_por_semestre = []
        for sem, pre_sem in datos_semestres:
            best = []
            for niveles_aportes, ws in pre_sem:
                if len(ws) == 0:
                    best.append(np.inf); continue
                s = pesos @ niveles_aportes
                best.append(float(ranks(s)[ws].min()))
            mrr_por_semestre.append(mrr_de(best))
        
        mrr_medio = float(np.mean(mrr_por_semestre))
        
        # Penalización L2 (Shrinkage) respecto a los pesos originales de b0_v2
        penalizacion = a.lambda_reg * np.sum((pesos - PESOS_ORIG) ** 2)
        
        # Score optimizado = MRR penalizado
        return mrr_medio - penalizacion, mrr_medio

    mrr_orig_score, mrr_orig_real = evaluar_pesos_regularizados(PESOS_ORIG)
    print(f"\nMRR promedio baseline original ({vert}) = {mrr_orig_real:.3f}")

    print(f"Ejecutando Random Search Regularizado ({a.n_iter} iteraciones)...")
    best_score = mrr_orig_score
    best_mrr = mrr_orig_real
    best_pesos = PESOS_ORIG.copy()

    rng = np.random.default_rng(42)
    for i in range(a.n_iter):
        p = rng.uniform(0.0, 1.0, size=len(PESOS_ORIG))
        p = p / p.sum() # Normalizar simplex
        
        score, mrr_real = evaluar_pesos_regularizados(p)
        if score > best_score:
            best_score = score
            best_mrr = mrr_real
            best_pesos = p.copy()

    print("\n==========================================")
    print(f"RESULTADO ROBUSTO ESTABILIZADO ({vert.upper()})")
    print("==========================================")
    print(f"MRR Promedio Real: {best_mrr:.3f} (Baseline: {mrr_orig_real:.3f}, Δ: {best_mrr - mrr_orig_real:+.3f})")
    print("Nuevos PESOS óptimos regularizados:")
    print(np.round(best_pesos, 3).tolist())


if __name__ == "__main__":
    main()