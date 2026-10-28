import pathlib
import pandas as pd
import b0
import b0_v2
import b0_v2_tuned
import evaluar

# Lista de todas las verticales del proyecto
verticales = ['rd', 'j', 'd', 'r', 'it']
filas_resultado = []

for v in verticales:
    print(f"\n==========================================")
    print(f"PROCESANDO VERTICAL: {v.upper()}")
    print(f"==========================================")
    
    d = pathlib.Path("data")
    lic = pd.read_parquet(d / v / "licitaciones.parquet")
    emp = pd.read_parquet(d / v / "empresas.parquet")
    hist = pd.read_parquet(d / v / "historial.parquet")
    gan = pd.read_parquet(d / v / "ganadores.parquet")
    off = pd.read_parquet(d / "office.parquet")

    # Definir el corte temporal de validación (último semestre)
    val = lic[lic.t >= b0_v2._cutoff(lic)].copy()
    gan_val = gan[gan.id.isin(set(val.id))]
    print(f"Validación temporal desde {b0_v2._cutoff(lic).date()}: {len(val)} licitaciones")

    # ----------------------------------------------------
    # 1. Evaluar Modelo B0 original
    # ----------------------------------------------------
    print("-> Evaluando modelo b0...")
    rk_b0 = b0_v2._rankear(b0.B0(hist), val, emp, 100)
    res_b0 = evaluar.evaluar(rk_b0, lic, emp, gan_val)
    fila_b0 = res_b0[res_b0['conjunto'] == 'todas'].iloc[0]
    
    filas_resultado.append({
        "modelo": "b0",
        "vertical": v,
        "nro_licitaciones": int(fila_b0['N']),
        "MRR": float(fila_b0['MRR']),
        "Hit@5": float(fila_b0['Hit@5']),
        "Hit@1": float(fila_b0['Hit@1']),
        "Hit@100": float(fila_b0['Hit@100']),
        "Recall@100": float(fila_b0['Recall@100']),
        "puesto_mediano": float(fila_b0['puesto mediano'])
    })

    # ----------------------------------------------------
    # 2. Evaluar Modelo B0_v2 (con backoff base)
    # ----------------------------------------------------
    print("-> Evaluando modelo b0_v2...")
    rk_v2 = b0_v2._rankear(b0_v2.B0V2(hist, off), val, emp, 100)
    res_v2 = evaluar.evaluar(rk_v2, lic, emp, gan_val, rk_b0)
    fila_v2 = res_v2[res_v2['conjunto'] == 'todas'].iloc[0]
    
    filas_resultado.append({
        "modelo": "b0_v2",
        "vertical": v,
        "nro_licitaciones": int(fila_v2['N']),
        "MRR": float(fila_v2['MRR']),
        "Hit@5": float(fila_v2['Hit@5']),
        "Hit@1": float(fila_v2['Hit@1']),
        "Hit@100": float(fila_v2['Hit@100']),
        "Recall@100": float(fila_v2['Recall@100']),
        "puesto_mediano": float(fila_v2['puesto mediano'])
    })

    # ----------------------------------------------------
    # 3. Evaluar Modelo B0_v2_tuned (pesos optimizados por vertical)
    # ----------------------------------------------------
    print("-> Evaluando modelo b0_v2_tuned...")
    rk_v2_tuned = b0_v2._rankear(b0_v2_tuned.B0V2(hist, off, vert=v), val, emp, 100)
    res_v2_tuned = evaluar.evaluar(rk_v2_tuned, lic, emp, gan_val, rk_b0)
    fila_v2_tuned = res_v2_tuned[res_v2_tuned['conjunto'] == 'todas'].iloc[0]
    
    filas_resultado.append({
        "modelo": "b0_v2_tuned",
        "vertical": v,
        "nro_licitaciones": int(fila_v2_tuned['N']),
        "MRR": float(fila_v2_tuned['MRR']),
        "Hit@5": float(fila_v2_tuned['Hit@5']),
        "Hit@1": float(fila_v2_tuned['Hit@1']),
        "Hit@100": float(fila_v2_tuned['Hit@100']),
        "Recall@100": float(fila_v2_tuned['Recall@100']),
        "puesto_mediano": float(fila_v2_tuned['puesto mediano'])
    })

# ----------------------------------------------------
# 4. Consolidar y Guardar Resultados
# ----------------------------------------------------
df_resumen = pd.DataFrame(filas_resultado)
df_resumen = df_resumen.sort_values(by=["vertical", "modelo"]).reset_index(drop=True)

nombre_csv_resumen = "resumen_evaluacion.csv"
df_resumen.to_csv(nombre_csv_resumen, index=False)

print("\n==========================================")
print("¡Proceso completado con éxito!")
print(f"Resultados guardados en: {nombre_csv_resumen}")
print("==========================================\n")
print("Tabla resumen comparativa de las 3 versiones:")
with pd.option_context("display.width", 220, "display.max_columns", None, "display.float_format", "{:.3f}".format):
    print(df_resumen)