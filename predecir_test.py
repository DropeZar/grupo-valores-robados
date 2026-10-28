"""ENTREGA: predicciones del modelo final sobre el test oficial (test-v1).

Modelo final = B0-v2 optimizado (historial con backoff + pesos/τ por vertical) + rerank por
texto con embeddings bge-small (fusión de rangos α=0.5, K=5). Lee test/<vert>/, usa el
historial.parquet de data/<vert>/ (corte 30/6/2024) y escribe un CSV por vertical en el
formato que lee evaluar.py: id, parent_uei, puesto (hasta 100 por licitación).

    PYTHONPATH="$PWD" uv run --no-project --python 3.12 --with sentence-transformers \
        --with numpy --with pandas --with pyarrow --with scikit-learn python predecir_test.py

Si ya están los embeddings en .emb_cache/test_<vert>_bges_{emp,lic}.npy (p. ej. codificados
en Colab con GPU), los usa; si no, codifica con bge-small en esta máquina y los cachea.
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
import b0_rocco

TOP, ALPHA, K = 200, 0.5, 5
MODELO, SUF = "BAAI/bge-small-en-v1.5", "bges"
INSTR = "Represent this sentence for searching relevant passages: "
INICIO = pd.Timestamp("2024-07-01")       # foto del test: historial hasta el 30/6/2024
VERTS = ["rd", "d", "j", "r", "it"]
SALIDA = pathlib.Path("predicciones")


def ranks(s):
    o = np.argsort(-s, kind="stable")
    r = np.empty(len(s), dtype=np.float64); r[o] = np.arange(1, len(s) + 1)
    return r


def embeddings(vert, cand, val, hist, cods):
    cache = pathlib.Path(".emb_cache"); cache.mkdir(exist_ok=True)
    f_emp, f_lic = cache / f"test_{vert}_{SUF}_emp.npy", cache / f"test_{vert}_{SUF}_lic.npy"
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
    print(f"  [{vert}] codificando {len(doc_emp):,} empresas + {len(doc_lic):,} licitaciones con {MODELO}…", flush=True)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODELO)
    C = m.encode(list(doc_emp), batch_size=32, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=True)
    Q = m.encode([INSTR + t for t in doc_lic], batch_size=32, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=True)
    np.save(f_emp, C); np.save(f_lic, Q)
    return C, Q


def main():
    from sklearn.preprocessing import normalize
    SALIDA.mkdir(exist_ok=True)
    d, dt = pathlib.Path("data"), pathlib.Path("test")
    psc = pd.read_parquet(d / "psc.parquet"); naics = pd.read_parquet(d / "naics.parquet")
    off = pd.read_parquet(d / "office.parquet")
    cods = (dict(zip(psc.codigo.astype(str).str.upper(), psc.descripcion.fillna(""))),
            dict(zip(naics.codigo.astype(str), naics.descripcion.fillna(""))),
            dict(zip(off.codigo.astype(str), off.oficina.fillna(""))))
    for vert in VERTS:
        val = pd.read_parquet(dt / vert / "licitaciones.parquet").reset_index(drop=True)
        cand = pd.read_parquet(dt / vert / "empresas.parquet").reset_index(drop=True)
        hist = pd.read_parquet(d / vert / "historial.parquet")
        val["t"] = pd.to_datetime(val["t"])
        uei = cand.parent_uei.values
        print(f"[{vert}] {len(val)} licitaciones, {len(cand):,} empresas", flush=True)

        C, Q = embeddings(vert, cand, val, hist, cods)
        C, Q = normalize(C), normalize(Q)
        prior = b0_rocco.B0V2(hist, off, vert=vert)
        filas = []
        for i, r in enumerate(val.itertuples()):
            p_v2 = prior.p_b0(r, cand).reindex(uei, fill_value=0.0).values
            sl = np.argsort(-p_v2, kind="stable")[:TOP]
            r_txt = ranks(Q[i] @ C[sl].T)
            r_cod = np.arange(1, len(sl) + 1, dtype=np.float64)
            s = (1 - ALPHA) / (K + r_cod) + ALPHA / (K + r_txt)
            ordr = sl[np.argsort(-s, kind="stable")[:100]]
            filas.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ordr],
                                       "puesto": np.arange(1, len(ordr) + 1)}))
        out = pd.concat(filas, ignore_index=True)
        f = SALIDA / f"{vert}.csv"
        out.to_csv(f, index=False)
        print(f"  [{vert}] -> {f}  ({len(out):,} filas, {out.id.nunique()} licitaciones)", flush=True)


if __name__ == "__main__":
    main()
