"""Optimiza los 8 pesos de b0_v2 para cada vertical de forma automatizada."""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd

import b0
from b0_v2 import B0V2, NIVELES

VERTICALES = ['rd', 'j', 'd', 'r', 'it']
DEV_SEMESTRE = "2023S2"

def ranks(scores):
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.float64)
    r[order] = np.arange(1, len(scores) + 1)
    return r

def mrr_de(best_ranks):
    return float(np.mean([1.0 / b if b <= 100 else 0.0 for b in best_ranks]))

def main():
    d = pathlib.Path("data")
    pesos_optimos_por_vertical = {}

    for vert in VERTICALES:
        print(f"\n==========================================")
        print(f"OPTIMIZANDO PESOS PARA VERTICAL: {vert.upper()}")
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

        lic["sem"] = lic.t.map(lambda t: f"{t.year}S{1 if t.month <= 6 else 2}")
        val = lic[lic["sem"] == DEV_SEMESTRE].copy()
        cand = emp[emp.semestre == DEV_SEMESTRE].reset_index(drop=True)
        uei = cand.parent_uei.values
        uei_pos = {u: i for i, u in enumerate(uei)}
        gan_dev = gan[gan.id.isin(set(val.id))]
        win = gan_dev.groupby("id").parent_uei.apply(list).to_dict()

        prior_v2 = B0V2(hist, off)
        emp_to_idx = prior_v2.idx

        print("Precalculando aportes por nivel...")
        pre = []
        for r in val.itertuples():
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

            ws = [uei_pos[w] for w in win.get(r.id, []) if w in uei_pos]
            pre.append((niveles_aportes, np.array(ws, dtype=np.int64)))

        PESOS_ORIG = np.array([0.40, 0.24, 0.12, 0.04, 0.20, 0.10, 0.06, 0.15], dtype=np.float64)

        def evaluar_pesos(pesos):
            best = []
            for niveles_aportes, ws in pre:
                if len(ws) == 0:
                    best.append(np.inf); continue
                s = pesos @ niveles_aportes
                best.append(float(ranks(s)[ws].min()))
            return mrr_de(best)

        mrr_orig = evaluar_pesos(PESOS_ORIG)
        print(f"MRR original ({vert}): {mrr_orig:.3f}")

        # Random Search
        best_mrr = mrr_orig
        best_pesos = PESOS_ORIG.copy()
        rng = np.random.default_rng(42)
        
        for i in range(800):
            p = rng.uniform(0.0, 1.0, size=len(PESOS_ORIG))
            p = p / p.sum()
            m = evaluar_pesos(p)
            if m > best_mrr:
                best_mrr = m
                best_pesos = p.copy()

        print(f"Mejor MRR ({vert}): {best_mrr:.3f} (Δ = {best_mrr - mrr_orig:+.3f})")
        pesos_optimos_por_vertical[vert] = np.round(best_pesos, 3).tolist()

    print("\n==========================================")
    print("DICCIONARIO DE PESOS ÓPTIMOS POR VERTICAL:")
    print("==========================================")
    import pprint
    pprint.pprint(pesos_optimos_por_vertical)

if __name__ == "__main__":
    main()