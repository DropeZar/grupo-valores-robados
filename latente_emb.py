"""Paso A: señal de texto por EMBEDDINGS (MiniLM), con documentos limpios + puente de códigos,
rerankeando el top-200 del B0-v2.

Cambia el "vector-builder" de latente.py (TF-IDF+SVD) por un modelo de embeddings
pre-entrenado (all-MiniLM-L6-v2). El resto del pipeline es el mismo: puntuar por alineación y
fusionar con el B0-v2. En vez de fusión total, **reordena solo el top-200 del v2** para no bajar
a los ganadores que la incumbencia ya tenía bien.

Documentos (limpios, con el puente de códigos; se tira la `description` cruda ruidosa):
  - licitación: title + descripción de su PSC + NAICS + nombre de oficina
  - empresa:    texto_awards + descripciones de los PSC de su historial (antes del semestre)

Mide en dev (2023S2), test (2024S1) sellado. Correr con un Python que soporte torch:

    PYTHONPATH="$PWD" uv run --no-project --python 3.12 \
        --with sentence-transformers --with numpy --with pandas --with pyarrow \
        --with scikit-learn python latente_emb.py --vert j --dev 2023S2
"""
from __future__ import annotations

import argparse
import pathlib
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")       # Windows: evita el crash al imprimir Δ
except Exception:
    pass

import numpy as np
import pandas as pd

import b0
import evaluar
from b0_v2 import B0V2

K_RRF = 60
TOP = 200          # tamaño del shortlist a rerankear


def sem_a_inicio(sem: str) -> pd.Timestamp:
    return pd.Timestamp(int(sem[:4]), 1 if sem[5] == "1" else 7, 1)


def ranks(scores):
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.int64)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def rows(lic_id, uei_arr):
    return pd.DataFrame({"id": lic_id, "parent_uei": uei_arr,
                         "puesto": np.arange(1, len(uei_arr) + 1)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vert", default="j")
    ap.add_argument("--dev", default="2023S2")
    ap.add_argument("--datos", default="data")
    ap.add_argument("--modelo", default="all-MiniLM-L6-v2")
    a = ap.parse_args()

    d = pathlib.Path(a.datos)
    lic = pd.read_parquet(d / a.vert / "licitaciones.parquet")
    emp = pd.read_parquet(d / a.vert / "empresas.parquet")
    hist = pd.read_parquet(d / a.vert / "historial.parquet")
    gan = pd.read_parquet(d / a.vert / "ganadores.parquet")
    psc = pd.read_parquet(d / "psc.parquet")
    naics = pd.read_parquet(d / "naics.parquet")
    off = pd.read_parquet(d / "office.parquet")

    psc_desc = dict(zip(psc.codigo.astype(str).str.upper(), psc.descripcion.fillna("")))
    naics_desc = dict(zip(naics.codigo.astype(str), naics.descripcion.fillna("")))
    off_name = dict(zip(off.codigo.astype(str), off.oficina.fillna("")))

    lic["sem"] = lic.t.map(lambda t: f"{t.year}S{1 if t.month <= 6 else 2}")
    val = lic[lic["sem"] == a.dev].copy()
    cand = emp[emp.semestre == a.dev].reset_index(drop=True)
    uei = cand.parent_uei.values
    gan_dev = gan[gan.id.isin(set(val.id))]
    inicio = sem_a_inicio(a.dev)
    print(f"vertical {a.vert} | dev {a.dev}: {len(val)} licitaciones, {len(cand):,} candidatas")

    # --- documentos: licitación (limpia + códigos) ---
    doc_lic = (val.title.fillna("") + " . "
               + val.psc.map(lambda c: psc_desc.get(str(c).upper(), "")) + " . "
               + val.naics.map(lambda c: naics_desc.get(str(c), "")) + " . "
               + val.office.map(lambda c: off_name.get(str(c), ""))).values

    # --- documentos: empresa (texto_awards + descripciones de sus PSC, antes del semestre) ---
    h = hist.copy()
    h["fecha"] = pd.to_datetime(h.fecha)
    h = h[h.fecha < inicio]
    h["psc"] = h.psc.astype(str).str.upper()
    psc_por_emp = (h.groupby("parent_uei")["psc"]
                   .apply(lambda s: " ".join(psc_desc.get(p, "") for p in pd.unique(s))))
    cod_emp = cand.parent_uei.map(psc_por_emp).fillna("")
    doc_emp = (cand.texto_awards.fillna("") + " . " + cod_emp).values

    # --- embeddings (con cache en .emb_cache/) ---
    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import normalize
    cache = pathlib.Path(".emb_cache"); cache.mkdir(exist_ok=True)
    f_emp = cache / f"{a.vert}_{a.dev}_emp.npy"
    f_lic = cache / f"{a.vert}_{a.dev}_lic.npy"
    if f_emp.exists() and f_lic.exists():
        print("cargando embeddings de cache…")
        C = np.load(f_emp); Q = np.load(f_lic)
    else:
        print(f"cargando modelo {a.modelo} y codificando (puede tardar)…")
        m = SentenceTransformer(a.modelo)
        C = m.encode(list(doc_emp), batch_size=64, normalize_embeddings=True,
                     show_progress_bar=True, convert_to_numpy=True)
        Q = m.encode(list(doc_lic), batch_size=64, normalize_embeddings=True,
                     show_progress_bar=True, convert_to_numpy=True)
        np.save(f_emp, C); np.save(f_lic, Q)
    C = normalize(C); Q = normalize(Q)

    # --- priors de códigos ---
    prior_b0 = b0.B0(hist)
    prior_v2 = B0V2(hist, off)

    # --- rankings ---
    print("rankeando…")
    filas = {m: [] for m in ("b0", "v2", "rerank", "rerank_txt", "fus_full")}
    for i, r in enumerate(val.itertuples()):
        p_b0 = prior_b0.p_b0(r, emp).reindex(uei, fill_value=0.0).values
        p_v2 = prior_v2.p_b0(r, emp).reindex(uei, fill_value=0.0).values
        order_v2 = np.argsort(-p_v2, kind="stable")
        sl = order_v2[:TOP]                              # shortlist del v2
        tsim_sl = Q[i] @ C[sl].T                         # texto, solo en el shortlist
        # rerank: RRF entre el orden del v2 (1..TOP) y el del texto en el shortlist
        rrf = 1.0 / (K_RRF + np.arange(1, TOP + 1)) + 1.0 / (K_RRF + ranks(tsim_sl))
        filas["b0"].append(rows(r.id, uei[np.argsort(-p_b0, kind="stable")[:100]]))
        filas["v2"].append(rows(r.id, uei[sl[:100]]))
        filas["rerank"].append(rows(r.id, uei[sl[np.argsort(-rrf, kind="stable")[:100]]]))
        filas["rerank_txt"].append(rows(r.id, uei[sl[np.argsort(-tsim_sl, kind="stable")[:100]]]))
        tsim_all = Q[i] @ C.T                            # fusión total (contraste)
        fus = 1.0 / (K_RRF + ranks(p_v2)) + 1.0 / (K_RRF + ranks(tsim_all))
        filas["fus_full"].append(rows(r.id, uei[np.argsort(-fus, kind="stable")[:100]]))
    rk = {m: pd.concat(v, ignore_index=True) for m, v in filas.items()}

    # --- evaluación vs B0 ---
    print("\n== dev (vs B0). MRR, Hit@5, dMRR (IC95), Hit@100 ==")
    out = []
    for m in ("v2", "rerank", "rerank_txt", "fus_full"):
        res = evaluar.evaluar(rk[m], lic, emp, gan_dev, rk["b0"])
        t = res[res.conjunto == "todas"].iloc[0]
        out.append({"modelo": m, "MRR": t.MRR, "Hit@5": t["Hit@5"],
                    "dMRR": t["Δ MRR"], "dMRR_IC95": t["Δ MRR IC95"],
                    "Hit@100": t["Hit@100"], "B0 MRR": t["B0 MRR"]})
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
