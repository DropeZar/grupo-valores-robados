"""Pipeline optimizado de pesos y taus.
- Etapa 1: Random search de pesos con tensor contiguo (8, L, E) y ranking O(E).
- Etapa 2: Barrido de tau al vuelo sin acumular tensores masivos.
  Tarda en correr, aviso: 1-2 horas en CPU de 8 hilos, dependiendo del tamaño de los datos y la cantidad de semestres.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import pprint

# Configuración obligatoria para evitar colisiones de hilos en macOS (M4)
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import b0
import b0_v2
from b0_v2 import B0V2, NIVELES, PESOS as PESOS_ORIG_RAW

# 'j' se coloca primero por ser el camino crítico de procesamiento[cite: 3, 9]
VERTICALES = ['j', 'rd', 'd', 'r', 'it']
CANDIDATOS_TAU = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

PESOS_ORIG = np.array(PESOS_ORIG_RAW, dtype=np.float64)
VERTICALES_REGULARIZADAS = {'rd', 'd'}
LAMBDA_REG = 0.12


def mejor_rank(s, ws, idx):
    """Función de conteo en O(E) equivalente a argsort estable con desempate por índice[cite: 6]."""
    sw = s[ws][:, None]
    r = (s[None, :] > sw).sum(1) + ((s[None, :] == sw) & (idx[None, :] < ws[:, None])).sum(1) + 1
    return r.min()


def mrr_de(best_ranks):
    return float(np.mean([1.0 / b if b <= 100 else 0.0 for b in best_ranks]))


def semestre_col(t):
    return t.dt.year.astype(str) + "S" + (t.dt.month > 6).map({False: "1", True: "2"})


def precalcular_aportes_tensor_contiguo(lic, emp, gan, prior_v2, sems_dev):
    """Precalcula y empaqueta los datos de cada semestre en un tensor contiguo (8, L, E)[cite: 7, 8]."""
    datos_semestres_packed = []
    emp_to_idx = prior_v2.idx

    for sem in sems_dev:
        val_sem = lic[lic["sem"] == sem].reset_index(drop=True)
        cand_sem = emp[emp.semestre == sem].reset_index(drop=True)
        if len(val_sem) == 0 or len(cand_sem) == 0:
            continue
        
        uei = cand_sem.parent_uei.values
        uei_pos = {u: i for i, u in enumerate(uei)}
        gan_sem = gan[gan.id.isin(set(val_sem.id))]
        win_sem = gan_sem.groupby("id").parent_uei.apply(list).to_dict()

        uei_indices = np.array([emp_to_idx.get(u, -1) for u in uei], dtype=np.int64)
        valid_mask = uei_indices != -1
        valid_indices = uei_indices[valid_mask]

        L = len(val_sem)
        E = len(uei)
        
        # Prealocación estricta del tensor (8, L, E) para evitar duplicados en RAM[cite: 7, 8]
        tensor_aportes = np.zeros((len(NIVELES), L, E), dtype=np.float64)
        ws_list = []

        for i, r in enumerate(val_sem.itertuples()):
            ts = np.datetime64(pd.Timestamp(r.t), "D").astype(np.int64)
            psc, office = str(r.psc).upper(), str(r.office).upper()
            ctx = {"psc": psc, "office": office, "naics": str(r.naics),
                   "sa": b0.set_aside(r.set_aside), "fam": psc[:2],
                   "sub": prior_v2.sub_de.get(office, "")}
            
            for idx_niv, (cols, tabla) in enumerate(zip(NIVELES, prior_v2.tablas)):
                key = tuple(ctx[c] for c in cols)
                if "" in key: continue
                e_tab = tabla.get(key)
                if e_tab is None: continue
                t_arr, fi, w = e_tab
                n = np.searchsorted(t_arr, ts, side="left")
                if n == 0: continue
                v = np.bincount(fi[:n], weights=w[:n], minlength=len(prior_v2.empresas))
                if v.sum() == 0: continue
                v_norm = v / v.sum()
                
                arr_uei = np.zeros(E, dtype=np.float64)
                arr_uei[valid_mask] = v_norm[valid_indices]
                tensor_aportes[idx_niv, i] = arr_uei

            ws = np.array([uei_pos[w] for w in win_sem.get(r.id, []) if w in uei_pos], dtype=np.int64)
            ws_list.append(ws)
        
        # Transformación a 2D contiguo para optimizar el producto matricial[cite: 7]
        tensor_2d = tensor_aportes.reshape(len(NIVELES), -1)
        datos_semestres_packed.append((sem, tensor_2d, L, E, ws_list, np.arange(E, dtype=np.int64)))
        
    return datos_semestres_packed


def evaluar_pesos_vectorizado(pesos, datos_semestres_packed, con_reg):
    mrr_por_semestre = []
    for sem, tensor_2d, L, E, ws_list, idx in datos_semestres_packed:
        scores_all = (pesos @ tensor_2d).reshape(L, E)
        
        best = []
        for i, ws in enumerate(ws_list):
            if len(ws) == 0:
                best.append(np.inf)
                continue
            s = scores_all[i]
            best.append(float(mejor_rank(s, ws, idx)))
        mrr_por_semestre.append(mrr_de(best))
    
    mrr_medio = float(np.mean(mrr_por_semestre)) if mrr_por_semestre else 0.0
    if con_reg:
        penalizacion = LAMBDA_REG * np.sum((pesos - PESOS_ORIG) ** 2)
        return mrr_medio - penalizacion, mrr_medio
    else:
        return mrr_medio, mrr_medio


def optimizar_pesos(vert, d, sems_dev, n_iter):
    print(f"\n" + "="*50)
    print(f"ETAPA 1: OPTIMIZANDO PESOS PARA VERTICAL: {vert.upper()}")
    print("="*50)

    try:
        lic = pd.read_parquet(d / vert / "licitaciones.parquet")
        emp = pd.read_parquet(d / vert / "empresas.parquet")
        hist = pd.read_parquet(d / vert / "historial.parquet")
        gan = pd.read_parquet(d / vert / "ganadores.parquet")
        off = pd.read_parquet(d / "office.parquet")
    except Exception as e:
        print(f"No se pudieron cargar los datos de {vert}: {e}")
        return vert, None

    lic["sem"] = semestre_col(lic.t)
    b0.TAU_ANIOS = 2.5
    b0_v2.TAU_ANIOS = 2.5
    
    prior_v2_pesos = B0V2(hist, off)
    datos_semestres_pesos = precalcular_aportes_tensor_contiguo(lic, emp, gan, prior_v2_pesos, sems_dev)

    con_reg = vert in VERTICALES_REGULARIZADAS
    print(f"[{vert.upper()}] Fase 1: Random Search de pesos (L2: {con_reg})")

    mrr_orig_score, mrr_orig_real = evaluar_pesos_vectorizado(PESOS_ORIG, datos_semestres_pesos, con_reg)
    best_score = mrr_orig_score
    best_mrr = mrr_orig_real
    best_pesos = PESOS_ORIG.copy()

    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        p = rng.uniform(0.0, 1.0, size=len(PESOS_ORIG))
        p = p / p.sum()
        score, mrr_real = evaluar_pesos_vectorizado(p, datos_semestres_pesos, con_reg)
        if score > best_score:
            best_score = score
            best_mrr = mrr_real
            best_pesos = p.copy()

    print(f"★ [{vert.upper()}] PESOS OPTIMIZADOS (MRR: {best_mrr:.3f})")
    return vert, np.round(best_pesos, 3).tolist()


def mrr_tau(vert, tau, best_pesos, d, sems_dev):
    """Etapa 2: Barrido de tau sin retener tensores masivos en RAM[cite: 7, 8, 9]."""
    print(f"[{vert.upper()}] Etapa 2: Evaluando Tau = {tau}")
    try:
        lic = pd.read_parquet(d / vert / "licitaciones.parquet")
        emp = pd.read_parquet(d / vert / "empresas.parquet")
        hist = pd.read_parquet(d / vert / "historial.parquet")
        gan = pd.read_parquet(d / vert / "ganadores.parquet")
        off = pd.read_parquet(d / "office.parquet")
    except Exception as e:
        print(f"Error cargando datos para tau en {vert}: {e}")
        return (vert, tau), 0.0

    lic["sem"] = semestre_col(lic.t)
    b0.TAU_ANIOS = tau
    b0_v2.TAU_ANIOS = tau

    prior_v2_tau = B0V2(hist, off)
    emp_to_idx = prior_v2_tau.idx
    pesos_arr = np.array(best_pesos, dtype=np.float64)

    mrr_por_semestre = []
    for sem in sems_dev:
        val_sem = lic[lic["sem"] == sem].reset_index(drop=True)
        cand_sem = emp[emp.semestre == sem].reset_index(drop=True)
        if len(val_sem) == 0 or len(cand_sem) == 0:
            continue
        
        uei = cand_sem.parent_uei.values
        uei_pos = {u: i for i, u in enumerate(uei)}
        gan_sem = gan[gan.id.isin(set(val_sem.id))]
        win_sem = gan_sem.groupby("id").parent_uei.apply(list).to_dict()

        uei_indices = np.array([emp_to_idx.get(u, -1) for u in uei], dtype=np.int64)
        valid_mask = uei_indices != -1
        valid_indices = uei_indices[valid_mask]
        E = len(uei)
        idx_arr = np.arange(E, dtype=np.int64)

        best = []
        for r in val_sem.itertuples():
            ts = np.datetime64(pd.Timestamp(r.t), "D").astype(np.int64)
            psc, office = str(r.psc).upper(), str(r.office).upper()
            ctx = {"psc": psc, "office": office, "naics": str(r.naics),
                   "sa": b0.set_aside(r.set_aside), "fam": psc[:2],
                   "sub": prior_v2_tau.sub_de.get(office, "")}
            
            niveles_aportes = np.zeros((len(NIVELES), E), dtype=np.float64)
            for idx_niv, (cols, tabla) in enumerate(zip(NIVELES, prior_v2_tau.tablas)):
                key = tuple(ctx[c] for c in cols)
                if "" in key: continue
                e_tab = tabla.get(key)
                if e_tab is None: continue
                t_arr, fi, w = e_tab
                n = np.searchsorted(t_arr, ts, side="left")
                if n == 0: continue
                v = np.bincount(fi[:n], weights=w[:n], minlength=len(prior_v2_tau.empresas))
                if v.sum() == 0: continue
                v_norm = v / v.sum()
                
                arr_uei = np.zeros(E, dtype=np.float64)
                arr_uei[valid_mask] = v_norm[valid_indices]
                niveles_aportes[idx_niv] = arr_uei

            s = pesos_arr @ niveles_aportes
            ws = [uei_pos[w] for w in win_sem.get(r.id, []) if w in uei_pos]
            if len(ws) == 0:
                best.append(np.inf)
                continue
            ws_arr = np.array(ws, dtype=np.int64)
            best.append(float(mejor_rank(s, ws_arr, idx_arr)))
        
        mrr_por_semestre.append(mrr_de(best))
    
    mrr_medio = float(np.mean(mrr_por_semestre)) if mrr_por_semestre else 0.0
    return (vert, tau), mrr_medio


def main():
    ap = argparse.ArgumentParser(description="Pipeline optimizado para M4")
    ap.add_argument("--datos", default="data")
    ap.add_argument("--n_iter", type=int, default=1000, help="Iteraciones random search pesos")
    ap.add_argument("--n_jobs_fase1", type=int, default=4, help="Hilos para etapa 1 (recomendado 4)")
    ap.add_argument("--n_jobs_fase2", type=int, default=6, help="Hilos para etapa 2 (recomendado 6)")
    a = ap.parse_args()

    d = pathlib.Path(a.datos)
    lic_ref = pd.read_parquet(d / VERTICALES[0] / "licitaciones.parquet")
    sems_disponibles = sorted(semestre_col(lic_ref.t).unique())
    sems_dev = sems_disponibles[:-1] if len(sems_disponibles) > 1 else sems_disponibles

    print(f"Semestres de validación: {sems_dev}")
    
    # ETAPA 1: Optimización de pesos en paralelo por vertical[cite: 3, 9]
    print("\n=== INICIANDO ETAPA 1: RANDOM SEARCH DE PESOS ===")
    res1 = Parallel(n_jobs=a.n_jobs_fase1)(
        delayed(optimizar_pesos)(v, d, sems_dev, a.n_iter) for v in VERTICALES
    )
    pesos_dict = {v: pesos for v, pesos in res1 if pesos is not None}

    # ETAPA 2: Barrido de Tau en paralelo por (vertical, tau)[cite: 3, 9]
    print("\n=== INICIANDO ETAPA 2: GRID SEARCH DE TAU ===")
    tareas = [(v, t) for v in pesos_dict for t in CANDIDATOS_TAU]
    res2 = Parallel(n_jobs=a.n_jobs_fase2)(
        delayed(mrr_tau)(v, t, pesos_dict[v], d, sems_dev) for v, t in tareas
    )

    resultados_optimos = {v: {"pesos": pesos_dict[v]} for v in pesos_dict}
    tau_por_vertical = {}
    for (v, t), mrr in res2:
        if v not in tau_por_vertical:
            tau_por_vertical[v] = []
        tau_por_vertical[v].append((t, mrr))

    for v, lista in tau_por_vertical.items():
        mejor_t, mejor_mrr = max(lista, key=lambda x: x[1])
        resultados_optimos[v]["tau"] = float(mejor_t)
        resultados_optimos[v]["mrr_tau"] = float(mejor_mrr)

    print("\n" + "="*50)
    print("RESUMEN GENERAL OPTIMIZADO:")
    print("="*50)
    pprint.pprint(resultados_optimos)


if __name__ == "__main__":
    main()