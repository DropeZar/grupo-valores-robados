"""Prueba el embedder BAAI/bge-base-en-v1.5 (vs MiniLM) en j/dev, con el mismo rerank.

bge pide una INSTRUCCIÓN DE CONSULTA prepended a las queries (licitaciones); las empresas van
sin instrucción. Cachea con sufijo _bge para no pisar los embeddings de MiniLM. Compara el
Δ MRR del rerank (α=0.5, K=5) contra el B0 — referencia MiniLM: +0.052.

    PYTHONPATH="$PWD" uv run --no-project --python 3.12 --with sentence-transformers \
        --with numpy --with pandas --with pyarrow --with scikit-learn python probar_bge.py
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
from b0_v2 import B0V2

VERT, DEV, TOP, ALPHA, K = "j", "2023S2", 200, 0.5, 5
MODELO = "BAAI/bge-small-en-v1.5"
SUF = "bges"   # sufijo de cache
INSTR = "Represent this sentence for searching relevant passages: "   # instrucción de query de bge


def ranks(s):
    o = np.argsort(-s, kind="stable")
    r = np.empty(len(s), dtype=np.float64); r[o] = np.arange(1, len(s) + 1)
    return r


def main():
    from sklearn.preprocessing import normalize
    d = pathlib.Path("data")
    lic = pd.read_parquet(d / VERT / "licitaciones.parquet")
    emp = pd.read_parquet(d / VERT / "empresas.parquet")
    hist = pd.read_parquet(d / VERT / "historial.parquet")
    gan = pd.read_parquet(d / VERT / "ganadores.parquet")
    psc = pd.read_parquet(d / "psc.parquet"); naics = pd.read_parquet(d / "naics.parquet")
    off = pd.read_parquet(d / "office.parquet")
    psc_desc = dict(zip(psc.codigo.astype(str).str.upper(), psc.descripcion.fillna("")))
    naics_desc = dict(zip(naics.codigo.astype(str), naics.descripcion.fillna("")))
    off_name = dict(zip(off.codigo.astype(str), off.oficina.fillna("")))

    lic["sem"] = lic.t.map(lambda t: f"{t.year}S{1 if t.month <= 6 else 2}")
    val = lic[lic["sem"] == DEV].copy()
    cand = emp[emp.semestre == DEV].reset_index(drop=True)
    uei = cand.parent_uei.values
    gan_dev = gan[gan.id.isin(set(val.id))]

    cache = pathlib.Path(".emb_cache")
    f_emp, f_lic = cache / f"{VERT}_{DEV}_{SUF}_emp.npy", cache / f"{VERT}_{DEV}_{SUF}_lic.npy"
    if f_emp.exists() and f_lic.exists():
        print("embeddings bge de cache", flush=True)
        C, Q = np.load(f_emp), np.load(f_lic)
    else:
        # documentos (iguales a latente_emb): licitación = title + códigos; empresa = awards + psc
        doc_lic = (val.title.fillna("") + " . "
                   + val.psc.map(lambda c: psc_desc.get(str(c).upper(), "")) + " . "
                   + val.naics.map(lambda c: naics_desc.get(str(c), "")) + " . "
                   + val.office.map(lambda c: off_name.get(str(c), ""))).values
        h = hist.copy(); h["fecha"] = pd.to_datetime(h.fecha)
        h = h[h.fecha < pd.Timestamp("2023-07-01")]
        h["psc"] = h.psc.astype(str).str.upper()
        psc_por_emp = (h.groupby("parent_uei")["psc"]
                       .apply(lambda s: " ".join(psc_desc.get(p, "") for p in pd.unique(s))))
        doc_emp = (cand.texto_awards.fillna("") + " . "
                   + cand.parent_uei.map(psc_por_emp).fillna("")).values
        print(f"codificando con {MODELO}: {len(doc_emp):,} empresas + {len(doc_lic)} licitaciones…", flush=True)
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(MODELO)
        C = m.encode(list(doc_emp), batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        Q = m.encode([INSTR + t for t in doc_lic], batch_size=32,
                     normalize_embeddings=True, convert_to_numpy=True)
        np.save(f_emp, C); np.save(f_lic, Q)
    C, Q = normalize(C), normalize(Q)

    prior_v2 = B0V2(hist, off); prior_b0 = b0.B0(hist)
    f_rr, f_b0 = [], []
    for i, r in enumerate(val.itertuples()):
        p_v2 = prior_v2.p_b0(r, emp).reindex(uei, fill_value=0.0).values
        sl = np.argsort(-p_v2, kind="stable")[:TOP]
        r_txt = ranks(Q[i] @ C[sl].T)
        r_cod = np.arange(1, len(sl) + 1, dtype=np.float64)
        s = (1 - ALPHA) / (K + r_cod) + ALPHA / (K + r_txt)
        ordr = sl[np.argsort(-s, kind="stable")[:100]]
        ob = np.argsort(-prior_b0.p_b0(r, emp).reindex(uei, fill_value=0.0).values, kind="stable")[:100]
        f_rr.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ordr], "puesto": np.arange(1, 101)}))
        f_b0.append(pd.DataFrame({"id": r.id, "parent_uei": uei[ob], "puesto": np.arange(1, 101)}))
    res = evaluar.evaluar(pd.concat(f_rr, ignore_index=True), lic, emp, gan_dev,
                          pd.concat(f_b0, ignore_index=True))
    t = res[res.conjunto == "todas"].iloc[0]
    print(f"\n== {VERT} dev {DEV} — rerank con bge-base ==")
    print(f"  ΔMRR={t['Δ MRR']:+.3f}  {t['Δ MRR IC95']}  MRR={t.MRR:.3f}  Hit@5={t['Hit@5']:.3f}")
    print(f"  (referencia MiniLM: ΔMRR=+0.052, Hit@5=0.303)")


if __name__ == "__main__":
    main()
