"""Valida el FORMATO de los 5 CSV de la entrega contra el test oficial, con el mismo
validador que usa la cátedra (evaluar.validar). Chequea, por vertical:
  - columnas id, parent_uei, puesto
  - están TODAS las licitaciones del test, y ninguna de más
  - puestos 1..K por licitación, sin saltos ni repetidos, hasta 100
  - sin (id, parent_uei) duplicados
  - cada empresa rankeada pertenece a test/<vert>/empresas.parquet

    PYTHONPATH="$PWD" uv run --no-project --python 3.12 --with pandas --with pyarrow \
        --with numpy python validar_entrega.py
"""
from __future__ import annotations

import pathlib
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

import evaluar

VERTS = ["rd", "d", "j", "r", "it"]
T, P = pathlib.Path("test"), pathlib.Path("predicciones")


def main():
    ok_todas = True
    for v in VERTS:
        lic = pd.read_parquet(T / v / "licitaciones.parquet")
        emp = pd.read_parquet(T / v / "empresas.parquet")
        f = P / f"{v}.csv"
        if not f.exists():
            print(f"[{v}] FALTA {f}"); ok_todas = False; continue
        rk = pd.read_csv(f)
        try:
            evaluar.validar(rk, lic, emp, set(lic.id))
            porlic = rk.groupby("id").size()
            print(f"[{v}] OK  {len(rk):,} filas · {rk.id.nunique()}/{len(lic)} licitaciones · "
                  f"empresas/lic min {porlic.min()} máx {porlic.max()}")
        except evaluar.RankingInvalido as e:
            print(f"[{v}] INVÁLIDO: {e}"); ok_todas = False
    print("\n" + ("TODO OK ✓ — listo para empaquetar" if ok_todas else "HAY ERRORES ✗ — revisar"))
    sys.exit(0 if ok_todas else 1)


if __name__ == "__main__":
    main()
