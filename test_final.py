"""NÚMERO FINAL: bge-small + B0-v2 de Rocco en el TEST (2024S1), las 5 verticales.

Codifica los pools de 2024S1 con bge-small (instrucción de query), cachea _bges, y corre el
rerank (α=0.5, K=5) con el B0-v2 tuneado de Rocco. Δ MRR vs B0, con IC95. Una sola pasada.

    PYTHONPATH="$PWD" uv run --no-project --python 3.12 --with sentence-transformers \
        --with numpy --with pandas --with pyarrow --with scikit-learn python test_final.py
"""
from __future__ import annotations

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
import b0_rocco

SEM, TOP, ALPHA, K = "2024S1", 200, 0.5, 5
MODELO, SUF = "BAAI/bge-small-en-v1.5", "bges"
INSTR = "Represent this sentence for searching relevant passages: "
INICIO = pd.Timestamp("2024-01-01")       # inicio de 2024S1
VERTS = ["rd", "d", "j", "r", "it"]


def ranks(s):
    o = np.argsort(-s, kind="stable")
    r = np.empty(len(s), dtype=np.float64); r[o] = np.arange(1, len(s) + 1)
    return r


def embeddings(vert, cand, val, hist, cods):
    cache = pathlib.Path(".emb_cache"); cache.mkdir(exist_ok=True)
    f_emp, f_lic = cache / f"{vert}_{SEM}_{SUF}_emp.npy", cache / f"{vert}_{SEM}_{SUF}_lic.npy"
    if f_emp.exists() and f_lic.exists():
        print(f"  [{vert}] emb de cache", flush=True)
        return np.load(f_emp), np.load(f_lic)
    psc_desc, naics_desc, off_name = cods
    doc_lic = (val.title.fillna("") + " . "
               + val.psc.map(lambda c: psc_desc.get(str(c).upper(), "")) + " . "
               + val.naics.map(lambda c: naics_desc.get(str(c), "")) + " . "
               + val.office.map(lambda c: off_name.get(str(c), ""))).values
    h = hist.copy(); h["fecha"] = pd.to_datetime(h.fecha)
    h = h[h.fecha < INICIO]; h["psc"] = h.psc.astype(str).str.upper()
    ppe = h.groupby("parent_uei")["psc"].apply(lambda s: " ".join(psc_desc.get(p, "") for p in pd.unique(s)))
    doc_emp = (cand.texto_awards.fillna("") + " . " + cand.parent_uei.map(ppe).fillna("")).values
    print(f"  [{vert}] codificando {len(doc_emp):,} empresas con {MODELO}…", flush=True)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODELO)
    C = m.encode(list(doc_emp), batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
    Q = m.encode([INSTR + t for t in doc_lic], batch_size=32,
                 normalize_embeddings=True, convert_to_numpy=True)
    np.save(f_emp, C); np.save(f_lic, Q)
    return C, Q


def main():
    from sklearn.preprocessing import normalize
    d = pathlib.Path("data")
    psc = pd.read_parquet(d / "psc.parquet"); naics = pd.read_parquet(d / "naics.parquet")
    off = pd.read_parquet(d / "office.parquet")
    cods = (dict(zip(psc.codigo.astype(str).str.upper(), psc.descripcion.fillna(""))),
            dict(zip(naics.codigo.astype(str), naics.descripcion.fillna(""))),
            dict(zip(off.codigo.astype(str), off.oficina.fillna(""))))
    filas = []
    for vert in VERTS:
        lic = pd.read_parquet(d / vert / "licitaciones.parquet")
        emp = pd.read_parquet(d / vert / "empresas.parquet")
        hist = pd.read_parquet(d / vert / "historial.parquet")
        gan = pd.read_parquet(d / vert / "ganadores.parquet")
        lic["sem"] = lic.t.map(lambda t: f"{t.year}S{1 if t.month <= 6 else 2}")
        val = lic[lic["sem"] == SEM].copy()
        cand = emp[emp.semestre == SEM].reset_index(drop=True)
        uei = cand.parent_uei.values
        gan_t = gan[gan.id.isin(set(val.id))]
        print(f"[{vert}] TEST {SEM}: {len(val)} licitaciones, {len(cand):,} candidatas", flush=True)

        C, Q = embeddings(vert, cand, val, hist, cods)
        C, Q = normalize(C), normalize(Q)
        prior_b0 = b0.B0(hist); prior = b0_rocco.B0V2(hist, off, vert=vert)
        f_rr, f_b0 = [], []
        for i, r in enumerate(val.itertuples()):
            p_v2 = prior.p_b0(r, emp).reindex(uei, fill_value=0.0).values
            sl = np.argsort(-p_v2, kind="stable")[:TOP]
            r_txt = ranks(Q[i] @ C[sl].T)
            r_cod = np.arange(1, len(sl) + 1, dtype=np.float64)
            s = (1 - ALPHA) / (K + r_cod) + ALPHA / (K + r_txt)
            ordr = sl[np.argsort(-s, kind="stable")[:100]]
            ob = np.argsort(-prior_b0.p_b0(r, emp).reindex(uei, fill_value=0.0).values, kind="stable")[:100]
            f_rr.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ordr], "puesto": np.arange(1, 101)}))
            f_b0.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ob], "puesto": np.arange(1, 101)}))
        res = evaluar.evaluar(pd.concat(f_rr, ignore_index=True), lic, emp, gan_t, pd.concat(f_b0, ignore_index=True))
        t = res[res.conjunto == "todas"].iloc[0]
        filas.append({"vert": vert, "N": len(val), "B0 MRR": t["B0 MRR"], "MRR": t.MRR,
                      "Hit@5": t["Hit@5"], "ΔMRR": t["Δ MRR"], "ΔMRR IC95": t["Δ MRR IC95"]})
        print(f"  [{vert}] ΔMRR = {t['Δ MRR']:+.3f}  {t['Δ MRR IC95']}  Hit@5={t['Hit@5']:.3f}", flush=True)

    print("\n===== TEST 2024S1 — bge-small + B0-v2 de Rocco (Δ MRR vs B0) =====")
    df = pd.DataFrame(filas)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(df.to_string(index=False))
    print(f"\npromedio ponderado por N: {np.average(df['ΔMRR'], weights=df['N']):+.4f}")


if __name__ == "__main__":
    main()
