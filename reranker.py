"""Paso D: reranker aprendido (MaxEnt + LightGBM) sobre el top-200 del B0-v2.

En vez de la fusión a mano (α fijo), un modelo aprende a combinar features mirando quién ganó.

    train = 2022S1 + 2022S2 + 2023S1   (se aprende acá)
    dev   = 2023S2                      (se compara vs el +0.052 de la fusión fija)
    test  = 2024S1                      (sellado)

Features por par (licitación, empresa del shortlist): incumbencia (rank_v2, p_v2), texto
(sim, rank_txt), concentración de p_v2 (Rényi-2/3, min-entropía), n_awards, e interacciones
texto×dificultad. Etiqueta: ganó (1/0).

    PYTHONPATH="$PWD" uv run --no-project --python 3.12 --with sentence-transformers \
        --with lightgbm --with scikit-learn --with numpy --with pandas --with pyarrow \
        python reranker.py --vert j
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

TRAIN_SEMS = ["2022S1", "2022S2", "2023S1"]
DEV = "2023S2"
TOP = 200
MODELO = "all-MiniLM-L6-v2"
FEATS = ["rank_v2", "p_v2", "sim", "rank_txt", "n_awards",
         "renyi2", "renyi3", "min_ent", "sim_x_r2", "pv2_x_r2"]


def sem_inicio(sem):
    return pd.Timestamp(int(sem[:4]), 1 if sem[5] == "1" else 7, 1)


def ranks(s):
    o = np.argsort(-s, kind="stable")
    r = np.empty(len(s), dtype=np.float64)
    r[o] = np.arange(1, len(s) + 1)
    return r


def renyi(p, q):
    p = p[p > 0].astype(np.float64)
    if len(p) <= 1:
        return 0.0
    p = p / p.sum()
    H = -np.sum(p * np.log(p)) if abs(q - 1) < 1e-9 else np.log(np.sum(p ** q)) / (1 - q)
    return float(H / np.log(len(p)))


def min_entropia(p):
    p = p[p > 0].astype(np.float64)
    if len(p) <= 1:
        return 0.0
    p = p / p.sum()
    return float(-np.log(p.max()) / np.log(len(p)))


def get_emb(vert, sem, cand, val, hist, cods):
    cache = pathlib.Path(".emb_cache"); cache.mkdir(exist_ok=True)
    f_emp, f_lic = cache / f"{vert}_{sem}_emp.npy", cache / f"{vert}_{sem}_lic.npy"
    if f_emp.exists() and f_lic.exists():
        print(f"  [{vert} {sem}] emb de cache", flush=True)
        return np.load(f_emp), np.load(f_lic)
    psc_desc, naics_desc, off_name = cods
    doc_lic = (val.title.fillna("") + " . "
               + val.psc.map(lambda c: psc_desc.get(str(c).upper(), "")) + " . "
               + val.naics.map(lambda c: naics_desc.get(str(c), "")) + " . "
               + val.office.map(lambda c: off_name.get(str(c), ""))).values
    h = hist.copy()
    h["fecha"] = pd.to_datetime(h.fecha)
    h = h[h.fecha < sem_inicio(sem)]
    h["psc"] = h.psc.astype(str).str.upper()
    psc_por_emp = (h.groupby("parent_uei")["psc"]
                   .apply(lambda s: " ".join(psc_desc.get(p, "") for p in pd.unique(s))))
    doc_emp = (cand.texto_awards.fillna("") + " . "
               + cand.parent_uei.map(psc_por_emp).fillna("")).values
    print(f"  [{vert} {sem}] codificando {len(doc_emp):,} empresas…", flush=True)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODELO)
    C = m.encode(list(doc_emp), batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    Q = m.encode(list(doc_lic), batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
    np.save(f_emp, C); np.save(f_lic, Q)
    return C, Q


def tabla_features(vert, sems, lic, emp, gan, hist, prior_v2, cods, naw):
    from sklearn.preprocessing import normalize
    filas = []
    for sem in sems:
        val = lic[lic["sem"] == sem].copy()
        cand = emp[emp.semestre == sem].reset_index(drop=True)
        if len(val) == 0 or len(cand) == 0:
            continue
        uei = cand.parent_uei.values
        C, Q = get_emb(vert, sem, cand, val, hist, cods)
        C, Q = normalize(C), normalize(Q)
        win = gan[gan.id.isin(set(val.id))].groupby("id").parent_uei.apply(set).to_dict()
        for i, r in enumerate(val.itertuples()):
            p_v2 = prior_v2.p_b0(r, emp).reindex(uei, fill_value=0.0).values
            sl = np.argsort(-p_v2, kind="stable")[:TOP]
            sim = Q[i] @ C[sl].T
            r_cod = np.arange(1, len(sl) + 1, dtype=np.float64)
            r_txt = ranks(sim)
            r2, r3, me = renyi(p_v2, 2), renyi(p_v2, 3), min_entropia(p_v2)
            ws = win.get(r.id, set())
            for j, idx in enumerate(sl):
                e = uei[idx]
                filas.append((r.id, r_cod[j], float(p_v2[idx]), float(sim[j]), r_txt[j],
                              float(naw.get((e, sem), 0)), r2, r3, me,
                              float(sim[j]) * r2, float(p_v2[idx]) * r2,
                              int(e in ws), e))
    df = pd.DataFrame(filas, columns=["qid"] + FEATS + ["label", "uei"])
    return df


def rank_dev(df, score_col, lic, emp, gan_dev):
    filas = []
    for qid, g in df.groupby("qid", sort=False):
        g = g.sort_values(score_col, ascending=False, kind="stable").head(100)
        filas.append(pd.DataFrame({"id": qid, "parent_uei": g.uei.values,
                                   "puesto": np.arange(1, len(g) + 1)}))
    return pd.concat(filas, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vert", default="j")
    a = ap.parse_args()
    d = pathlib.Path("data")
    lic = pd.read_parquet(d / a.vert / "licitaciones.parquet")
    emp = pd.read_parquet(d / a.vert / "empresas.parquet")
    hist = pd.read_parquet(d / a.vert / "historial.parquet")
    gan = pd.read_parquet(d / a.vert / "ganadores.parquet")
    psc = pd.read_parquet(d / "psc.parquet"); naics = pd.read_parquet(d / "naics.parquet")
    off = pd.read_parquet(d / "office.parquet")
    cods = (dict(zip(psc.codigo.astype(str).str.upper(), psc.descripcion.fillna(""))),
            dict(zip(naics.codigo.astype(str), naics.descripcion.fillna(""))),
            dict(zip(off.codigo.astype(str), off.oficina.fillna(""))))
    naw = dict(zip(zip(emp.parent_uei, emp.semestre), emp.n_awards))
    lic["sem"] = lic.t.map(lambda t: f"{t.year}S{1 if t.month <= 6 else 2}")

    prior_v2 = B0V2(hist, off)
    prior_b0 = b0.B0(hist)
    print(f"[{a.vert}] armando features de train {TRAIN_SEMS}…", flush=True)
    tr = tabla_features(a.vert, TRAIN_SEMS, lic, emp, gan, hist, prior_v2, cods, naw)
    print(f"[{a.vert}] armando features de dev {DEV}…", flush=True)
    dv = tabla_features(a.vert, [DEV], lic, emp, gan, hist, prior_v2, cods, naw)
    print(f"  train: {len(tr):,} filas, {tr.qid.nunique()} licitaciones | "
          f"dev: {len(dv):,} filas, {dv.qid.nunique()} licitaciones", flush=True)

    gan_dev = gan[gan.id.isin(set(dv.qid))]

    # baseline: fusión fija α=0.5, K=5 (referencia) y B0
    dv["fija"] = 0.5 / (5 + dv.rank_v2) + 0.5 / (5 + dv.rank_txt)
    rk_b0 = rank_dev(dv.assign(pv2neg=-dv.rank_v2), "pv2neg", lic, emp, gan_dev)

    def delta(score_col, nombre):
        rk = rank_dev(dv, score_col, lic, emp, gan_dev)
        res = evaluar.evaluar(rk, lic, emp, gan_dev, rk_b0)
        t = res[res.conjunto == "todas"].iloc[0]
        print(f"  {nombre:16s} ΔMRR={t['Δ MRR']:+.3f}  {t['Δ MRR IC95']}  "
              f"MRR={t.MRR:.3f}  Hit@5={t['Hit@5']:.3f}", flush=True)

    # MaxEnt (logística)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(tr[FEATS])
    lrm = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(tr[FEATS]), tr.label)
    dv["maxent"] = lrm.predict_proba(sc.transform(dv[FEATS]))[:, 1]

    # LightGBM lambdarank
    import lightgbm as lgb
    grp = tr.groupby("qid", sort=False).size().values
    rk = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.05,
                        num_leaves=31, min_child_samples=30, subsample=0.8,
                        colsample_bytree=0.8, verbose=-1)
    rk.fit(tr[FEATS], tr.label, group=grp)
    dv["lgbm"] = rk.predict(dv[FEATS])

    print(f"\n== dev {DEV} (vert {a.vert}), Δ MRR vs B0 ==")
    delta("fija", "fusión fija")
    delta("maxent", "MaxEnt (logíst)")
    delta("lgbm", "LightGBM")

    print("\ncoeficientes MaxEnt (estandarizados):")
    for f, c in sorted(zip(FEATS, lrm.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {f:12s} {c:+.3f}")
    print("\nimportancia LightGBM:")
    for f, c in sorted(zip(FEATS, rk.feature_importances_), key=lambda x: -x[1]):
        print(f"  {f:12s} {int(c)}")


if __name__ == "__main__":
    main()
