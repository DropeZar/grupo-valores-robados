"""B0 v2: el B0 con *backoff* de códigos, para meter más ganadores al top-100.

El B0 original solo premia coincidencias EXACTAS de código: mismo PSC, misma oficina.
Eso entierra a empresas que hicieron el mismo trabajo pero con un PSC hermano o en otra
oficina de la misma subagencia. v2 agrega tres niveles más blandos a la mezcla, SIN tocar
los cinco originales:

    (psc_familia, office)   la empresa ganó un PSC de la misma familia en esta oficina
    (psc_familia,)          ... en cualquier oficina        [familia = primeros 2 chars]
    (psc, subagencia)       ganó este mismo PSC en otra oficina de la misma subagencia

Todo lo demás es idéntico al B0: mismo decaimiento (τ=2,5 años), mismo corte estricto
`fecha < t`, misma mezcla por pesos, mismos empates deterministas. Los pesos de los niveles
nuevos están puestos a mano (abajo) y son conservadores; más adelante se aprenden.

Como script, compara v2 contra el B0 en la validación temporal (último semestre):

    uv run python b0_v2.py --vert j
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

import b0
from b0 import TAU_ANIOS, set_aside

# niveles = (columnas, peso). Los 5 primeros son EXACTAMENTE el B0 original.
NIVELES = [("psc", "office"), ("psc",), ("naics",), (), ("psc", "sa"),
           ("fam", "office"), ("fam",), ("psc", "sub")]
PESOS = [0.40, 0.24, 0.12, 0.04, 0.20,
         0.10, 0.06, 0.15]


class B0V2(b0.B0):
    """El B0 con los niveles de backoff. Reusa `_indexar`, `p_b0` y `ranking` del padre;
    solo cambia qué columnas se indexan (`__init__`) y qué contexto se consulta (`_p`)."""

    def __init__(self, historial: pd.DataFrame, office: pd.DataFrame):
        self.sub_de = dict(zip(office.codigo.astype(str),
                               office.subagencia.fillna("").astype(str)))
        h = historial.copy()
        h["psc"] = h.psc.astype(str).str.upper()
        h["office"] = h.office.astype(str).str.upper()
        h["naics"] = h.naics.astype(str)
        h["sa"] = h.set_aside.map(set_aside)
        h["fam"] = h.psc.str[:2]                              # familia PSC
        h["sub"] = h.office.map(lambda o: self.sub_de.get(o, ""))  # subagencia
        self.empresas = np.array(sorted(h.parent_uei.unique()))
        self.idx = {u: i for i, u in enumerate(self.empresas)}
        h["fi"] = h.parent_uei.map(self.idx).astype(np.int64)
        h["ts"] = pd.to_datetime(h.fecha).values.astype("datetime64[D]").astype(np.int64)
        self.h = h.sort_values(["ts", "fi"], kind="stable").reset_index(drop=True)
        self.w = np.exp((self.h.ts.values - self.h.ts.values.max()) / (TAU_ANIOS * 365.25))
        self.tablas = [self._indexar(cols) for cols in NIVELES]

    def _p(self, lic) -> np.ndarray:
        ts = np.datetime64(pd.Timestamp(lic.t), "D").astype(np.int64)
        psc, office = str(lic.psc).upper(), str(lic.office).upper()
        ctx = {"psc": psc, "office": office, "naics": str(lic.naics),
               "sa": set_aside(lic.set_aside), "fam": psc[:2],
               "sub": self.sub_de.get(office, "")}
        p = np.zeros(len(self.empresas))
        for cols, peso, tabla in zip(NIVELES, PESOS, self.tablas):
            key = tuple(ctx[c] for c in cols)
            if "" in key:                 # falta un campo (p. ej. subagencia desconocida): se saltea
                continue
            e = tabla.get(key)
            if e is None:
                continue
            t_arr, fi, w = e
            n = np.searchsorted(t_arr, ts, side="left")       # estricto: sólo antes de t
            if n == 0:
                continue
            v = np.bincount(fi[:n], weights=w[:n], minlength=len(self.empresas))
            p += peso * v / v.sum()
        return p


def _rankear(prior, lic_val, emp, top):
    return pd.concat([prior.ranking(f, emp, top) for f in lic_val.itertuples()],
                     ignore_index=True)


def _cutoff(lic):
    y, m = lic.t.max().year, lic.t.max().month
    return pd.Timestamp(f"{y}-01-01") if m <= 6 else pd.Timestamp(f"{y}-07-01")


def main() -> None:
    import evaluar
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vert", default="j", help="rd, d, j, r, it")
    ap.add_argument("--datos", default="data")
    ap.add_argument("--top", type=int, default=100)
    a = ap.parse_args()

    d = pathlib.Path(a.datos)
    lic = pd.read_parquet(d / a.vert / "licitaciones.parquet")
    emp = pd.read_parquet(d / a.vert / "empresas.parquet")
    hist = pd.read_parquet(d / a.vert / "historial.parquet")
    gan = pd.read_parquet(d / a.vert / "ganadores.parquet")
    off = pd.read_parquet(d / "office.parquet")

    val = lic[lic.t >= _cutoff(lic)].copy()
    gan_val = gan[gan.id.isin(set(val.id))]
    print(f"vertical {a.vert} | validación temporal desde {_cutoff(lic).date()}: "
          f"{len(val)} licitaciones")

    rk_b0 = _rankear(b0.B0(hist), val, emp, a.top)
    rk_v2 = _rankear(B0V2(hist, off), val, emp, a.top)

    res = evaluar.evaluar(rk_v2, lic, emp, gan_val, rk_b0)
    print("\n== B0-v2 (backoff) vs B0. 'B0 ...' y 'Δ ...' son el baseline y la mejora ==")
    with pd.option_context("display.width", 220, "display.max_columns", None,
                           "display.float_format", "{:.3f}".format):
        print(res.to_string(index=False))


if __name__ == "__main__":
    main()
