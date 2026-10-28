"""Señal de texto por factores latentes (LSA), fusionada con el B0-v2.

Representa cada empresa (su `texto_awards`) y cada licitación (`title`+`description`) como
vectores en un mismo espacio, y rankea por alineación (coseno). Compara, sobre dev:

    v0  = TF-IDF coseno (palabras crudas)      -> el baseline "de todos"
    v1  = + SVD/LSA (factores latentes)        -> ¿gana el espacio latente?
    fus = rank fusion (B0-v2  +  v1)           -> ¿sube el MRR fusionado con los códigos?

Disciplina temporal: los vectores de empresa son los del semestre evaluado (texto_awards ya
viene "congelado"); el TF-IDF y el SVD se ajustan sin mirar el semestre de test.

Uso:
    uv run --with scikit-learn python latente.py --vert j --dev 2023S2
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

import b0
import evaluar
from b0_v2 import B0V2

K_RRF = 60


def sem_de(ts):
    ts = pd.Timestamp(ts)
    return f"{ts.year}S{1 if ts.month <= 6 else 2}"


def ranks(scores):
    """Puesto 1-based de cada candidata según `scores` (desc), empates estables."""
    order = np.argsort(-scores, kind="stable")
    r = np.empty(len(scores), dtype=np.int64)
    r[order] = np.arange(1, len(scores) + 1)
    return r


def topk_rows(lic_id, uei, scores, k=100):
    idx = np.argsort(-scores, kind="stable")[:k]
    return pd.DataFrame({"id": lic_id, "parent_uei": uei[idx],
                         "puesto": np.arange(1, len(idx) + 1)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vert", default="j")
    ap.add_argument("--dev", default="2023S2", help="semestre de validación")
    ap.add_argument("--datos", default="data")
    ap.add_argument("--k", type=int, default=200, help="factores latentes (SVD)")
    a = ap.parse_args()

    d = pathlib.Path(a.datos)
    lic = pd.read_parquet(d / a.vert / "licitaciones.parquet")
    emp = pd.read_parquet(d / a.vert / "empresas.parquet")
    hist = pd.read_parquet(d / a.vert / "historial.parquet")
    gan = pd.read_parquet(d / a.vert / "ganadores.parquet")
    off = pd.read_parquet(d / "office.parquet")

    lic["sem"] = lic.t.map(sem_de)
    val = lic[lic["sem"] == a.dev].copy()
    dev_fin = val.t.max()                                   # fin del semestre de dev
    cand = emp[emp.semestre == a.dev].reset_index(drop=True)  # candidatas de dev, orden fijo
    uei = cand.parent_uei.values
    gan_dev = gan[gan.id.isin(set(val.id))]
    print(f"vertical {a.vert} | dev {a.dev}: {len(val)} licitaciones, {len(cand):,} candidatas")

    # --- documentos ---
    doc_emp = cand.texto_awards.fillna("").values
    doc_val = (val.title.fillna("") + " . " + val.description.fillna("")).values
    # corpus para ajustar: empresas de dev + licitaciones hasta dev (NUNCA el test posterior)
    lic_fit = lic[lic.t <= dev_fin]
    doc_fit_lic = (lic_fit.title.fillna("") + " . " + lic_fit.description.fillna("")).values
    corpus = np.concatenate([doc_emp, doc_fit_lic])

    # --- TF-IDF compartido + SVD ---
    print("ajustando TF-IDF…")
    tfidf = TfidfVectorizer(max_features=40000, min_df=3, sublinear_tf=True,
                            stop_words="english", ngram_range=(1, 2))
    tfidf.fit(corpus)
    print(f"  vocabulario: {len(tfidf.vocabulary_):,}")
    Ct = normalize(tfidf.transform(doc_emp))               # empresas, TF-IDF
    Qt = normalize(tfidf.transform(doc_val))               # licitaciones, TF-IDF
    print(f"ajustando SVD (k={a.k})…")
    svd = TruncatedSVD(n_components=a.k, random_state=0)
    svd.fit(tfidf.transform(corpus))
    C = normalize(svd.transform(tfidf.transform(doc_emp)))  # empresas, latente
    Q = normalize(svd.transform(tfidf.transform(doc_val)))  # licitaciones, latente
    print(f"  varianza explicada por {a.k} factores: {svd.explained_variance_ratio_.sum():.1%}")

    # --- priors de códigos ---
    prior_b0 = b0.B0(hist)
    prior_v2 = B0V2(hist, off)

    # --- rankings por licitación ---
    print("rankeando…")
    CtT = Ct.T.tocsr()                                       # empresas^T, una sola vez
    CT = C.T                                                 # idem, latente
    filas = {m: [] for m in ("b0", "v2", "v0_tfidf", "v1_lsa", "fus")}
    for i, r in enumerate(val.itertuples()):
        p_b0 = prior_b0.p_b0(r, emp).reindex(uei, fill_value=0.0).values
        p_v2 = prior_v2.p_b0(r, emp).reindex(uei, fill_value=0.0).values
        s_tfidf = np.asarray((Qt[i] @ CtT).todense()).ravel()  # coseno TF-IDF
        s_lsa = Q[i] @ CT                                       # coseno latente
        # fusión: B0-v2 + texto latente, por rank fusion
        s_fus = 1.0 / (K_RRF + ranks(p_v2)) + 1.0 / (K_RRF + ranks(s_lsa))
        for m, s in (("b0", p_b0), ("v2", p_v2), ("v0_tfidf", s_tfidf),
                     ("v1_lsa", s_lsa), ("fus", s_fus)):
            filas[m].append(topk_rows(r.id, uei, s))
    rk = {m: pd.concat(v, ignore_index=True) for m, v in filas.items()}

    # --- evaluación: cada modelo vs el B0 ---
    print("\n== dev (vs B0). N, MRR, Hit@5/1, Δ MRR (IC95) y Hit@100 (recall) ==")
    rows = []
    for m in ("v2", "v0_tfidf", "v1_lsa", "fus"):
        res = evaluar.evaluar(rk[m], lic, emp, gan_dev, rk["b0"])
        tot = res[res.conjunto == "todas"].iloc[0]
        rows.append({"modelo": m, "MRR": tot.MRR, "Hit@5": tot["Hit@5"],
                     "ΔMRR": tot["Δ MRR"], "ΔMRR IC95": tot["Δ MRR IC95"],
                     "Hit@100": tot["Hit@100"], "B0 MRR": tot["B0 MRR"]})
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
