"""Confirma el resultado del rerank (embeddings + fusion tuneada) en TODAS las verticales.

Aplica la MISMA config elegida en j (alpha=0.5, K=5) a cada vertical en dev (2023S2) — no
re-tunea por vertical (eso seria optimista), solo verifica si el +0.052 generaliza. Construye
los documentos igual que latente_emb.py y reutiliza/crea el cache de embeddings.

    PYTHONPATH="$PWD" uv run --no-project --python 3.12 --with sentence-transformers \
        --with numpy --with pandas --with pyarrow --with scikit-learn \
        python confirmar_verticales.py
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

ALPHA = 0.5
K = 5
TOP = 200
DEV = "2023S2"
VERTS = ["j", "rd", "d", "r", "it"]
MODELO = "all-MiniLM-L6-v2"


def sem_inicio(sem):
    return pd.Timestamp(int(sem[:4]), 1 if sem[5] == "1" else 7, 1)


def ranks(s):
    o = np.argsort(-s, kind="stable")
    r = np.empty(len(s), dtype=np.float64)
    r[o] = np.arange(1, len(s) + 1)
    return r


def embeddings(vert, sem, cand, val, hist, psc_desc, naics_desc, off_name, inicio):
    cache = pathlib.Path(".emb_cache"); cache.mkdir(exist_ok=True)
    f_emp, f_lic = cache / f"{vert}_{sem}_emp.npy", cache / f"{vert}_{sem}_lic.npy"
    if f_emp.exists() and f_lic.exists():
        print(f"  [{vert}] embeddings de cache", flush=True)
        return np.load(f_emp), np.load(f_lic)
    # documentos (iguales a latente_emb.py): licitacion = title + codigos; empresa = awards + psc
    doc_lic = (val.title.fillna("") + " . "
               + val.psc.map(lambda c: psc_desc.get(str(c).upper(), "")) + " . "
               + val.naics.map(lambda c: naics_desc.get(str(c), "")) + " . "
               + val.office.map(lambda c: off_name.get(str(c), ""))).values
    h = hist.copy()
    h["fecha"] = pd.to_datetime(h.fecha)
    h = h[h.fecha < inicio]
    h["psc"] = h.psc.astype(str).str.upper()
    psc_por_emp = (h.groupby("parent_uei")["psc"]
                   .apply(lambda s: " ".join(psc_desc.get(p, "") for p in pd.unique(s))))
    cod_emp = cand.parent_uei.map(psc_por_emp).fillna("")
    doc_emp = (cand.texto_awards.fillna("") + " . " + cod_emp).values
    print(f"  [{vert}] codificando {len(doc_emp):,} empresas + {len(doc_lic)} licitaciones…", flush=True)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODELO)
    C = m.encode(list(doc_emp), batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    Q = m.encode(list(doc_lic), batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    np.save(f_emp, C); np.save(f_lic, Q)
    return C, Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem", default=DEV, help="semestre a evaluar (dev 2023S2 o test 2024S1)")
    ap.add_argument("--verts", default=",".join(VERTS), help="verticales separadas por coma")
    args = ap.parse_args()
    sem = args.sem
    verts = [v.strip() for v in args.verts.split(",")]
    inicio = sem_inicio(sem)
    from sklearn.preprocessing import normalize
    d = pathlib.Path("data")
    psc = pd.read_parquet(d / "psc.parquet")
    naics = pd.read_parquet(d / "naics.parquet")
    off = pd.read_parquet(d / "office.parquet")
    psc_desc = dict(zip(psc.codigo.astype(str).str.upper(), psc.descripcion.fillna("")))
    naics_desc = dict(zip(naics.codigo.astype(str), naics.descripcion.fillna("")))
    off_name = dict(zip(off.codigo.astype(str), off.oficina.fillna("")))

    filas = []
    for vert in verts:
        lic = pd.read_parquet(d / vert / "licitaciones.parquet")
        emp = pd.read_parquet(d / vert / "empresas.parquet")
        hist = pd.read_parquet(d / vert / "historial.parquet")
        gan = pd.read_parquet(d / vert / "ganadores.parquet")
        lic["sem"] = lic.t.map(lambda t: f"{t.year}S{1 if t.month <= 6 else 2}")
        val = lic[lic["sem"] == sem].copy()
        cand = emp[emp.semestre == sem].reset_index(drop=True)
        uei = cand.parent_uei.values
        gan_dev = gan[gan.id.isin(set(val.id))]
        print(f"[{vert}] {sem}: {len(val)} licitaciones, {len(cand):,} candidatas", flush=True)

        C, Q = embeddings(vert, sem, cand, val, hist, psc_desc, naics_desc, off_name, inicio)
        C, Q = normalize(C), normalize(Q)
        prior_b0, prior_v2 = b0.B0(hist), B0V2(hist, off)

        f_rr, f_b0 = [], []
        for i, r in enumerate(val.itertuples()):
            p_v2 = prior_v2.p_b0(r, emp).reindex(uei, fill_value=0.0).values
            sl = np.argsort(-p_v2, kind="stable")[:TOP]
            r_txt = ranks(Q[i] @ C[sl].T)
            r_cod = np.arange(1, len(sl) + 1, dtype=np.float64)
            s = (1 - ALPHA) / (K + r_cod) + ALPHA / (K + r_txt)
            ord_rr = sl[np.argsort(-s, kind="stable")[:100]]
            ord_b0 = np.argsort(-prior_b0.p_b0(r, emp).reindex(uei, fill_value=0.0).values,
                                kind="stable")[:100]
            f_rr.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ord_rr], "puesto": np.arange(1, 101)}))
            f_b0.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ord_b0], "puesto": np.arange(1, 101)}))
        res = evaluar.evaluar(pd.concat(f_rr, ignore_index=True), lic, emp, gan_dev,
                              pd.concat(f_b0, ignore_index=True))
        t = res[res.conjunto == "todas"].iloc[0]
        filas.append({"vert": vert, "N": len(val), "B0 MRR": t["B0 MRR"], "MRR": t.MRR,
                      "Hit@5": t["Hit@5"], "ΔMRR": t["Δ MRR"], "ΔMRR IC95": t["Δ MRR IC95"]})
        print(f"  [{vert}] ΔMRR = {t['Δ MRR']:+.3f}  {t['Δ MRR IC95']}", flush=True)

    print(f"\n===== RESUMEN (rerank α=0.5, K=5, {sem}) =====")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(pd.DataFrame(filas).to_string(index=False))


if __name__ == "__main__":
    main()
