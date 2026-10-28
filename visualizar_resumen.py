"""Visualización comparativa de los tres baselines: b0 vs b0_v2 vs b0_v2_tuned.

Lee el archivo 'resumen_evaluacion.csv' generado por evaluar_todas.py
y construye un dashboard gráfico con las métricas principales (MRR, Hit@100, Hit@5, Recall@100 y Puesto Mediano)
guardándolo en 'comparativa_baselines.png'.

Uso:
    uv run python visualizar_resumen.py
"""
from __future__ import annotations

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Paleta de colores unificada del proyecto
COLORES = {
    "b0": "#2e6f95",           # Azul baseline
    "b0_v2": "#b0b0b0",        # Gris neutro
    "b0_v2_tuned": "#d1495b"   # Rojo acento / destacado
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", default="resumen_evaluacion.csv", help="CSV de evaluación consolidado")
    ap.add_argument("--out", default="comparativa_baselines.png", help="Archivo PNG de salida")
    ap.add_argument("--no-show", action="store_true", help="No desplegar la ventana gráfica")
    a = ap.parse_args()

    ruta_csv = pathlib.Path(a.csv)
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el archivo '{a.csv}'. "
                                f"Asegurate de ejecutar 'evaluar_todas.py' primero.")

    df = pd.read_csv(ruta_csv)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # Orden del eje X por verticales fijas
    order_verts = ["d", "it", "j", "r", "rd"]

    # ----------------------------------------------------
    # Panel (a): MRR
    # ----------------------------------------------------
    ax_mrr = axes[0, 0]
    sns.barplot(data=df, x="vertical", y="MRR", hue="modelo", order=order_verts,
                palette=COLORES, ax=ax_mrr)
    ax_mrr.set_title("(a) MRR por Vertical", fontsize=12, fontweight="bold")
    ax_mrr.set_ylabel("MRR")
    for p in ax_mrr.patches:
        h = p.get_height()
        if h > 0:
            ax_mrr.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=8, xytext=(0, 2),
                            textcoords='offset points')

    # ----------------------------------------------------
    # Panel (b): Hit@100
    # ----------------------------------------------------
    ax_h100 = axes[0, 1]
    sns.barplot(data=df, x="vertical", y="Hit@100", hue="modelo", order=order_verts,
                palette=COLORES, ax=ax_h100)
    ax_h100.set_title("(b) Cobertura Top-100 (Hit@100) por Vertical", fontsize=12, fontweight="bold")
    ax_h100.set_ylabel("Hit@100")
    ax_h100.set_ylim(0, 0.7)
    for p in ax_h100.patches:
        h = p.get_height()
        if h > 0:
            ax_h100.annotate(f"{h*100:.1f}%", (p.get_x() + p.get_width() / 2., h),
                             ha='center', va='bottom', fontsize=8, xytext=(0, 2),
                             textcoords='offset points')

    # ----------------------------------------------------
    # Panel (c): Hit@5
    # ----------------------------------------------------
    ax_h5 = axes[1, 0]
    sns.barplot(data=df, x="vertical", y="Hit@5", hue="modelo", order=order_verts,
                palette=COLORES, ax=ax_h5)
    ax_h5.set_title("(c) Precisión Top-5 (Hit@5) por Vertical", fontsize=12, fontweight="bold")
    ax_h5.set_ylabel("Hit@5")
    for p in ax_h5.patches:
        h = p.get_height()
        if h > 0:
            ax_h5.annotate(f"{h*100:.1f}%", (p.get_x() + p.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=8, xytext=(0, 2),
                            textcoords='offset points')

    # ----------------------------------------------------
    # Panel (d): Recall@100
    # ----------------------------------------------------
    ax_r100 = axes[1, 1]
    sns.barplot(data=df, x="vertical", y="Recall@100", hue="modelo", order=order_verts,
                palette=COLORES, ax=ax_r100)
    ax_r100.set_title("(d) Recall@100 por Vertical", fontsize=12, fontweight="bold")
    ax_r100.set_ylabel("Recall@100")
    ax_r100.set_ylim(0, 0.7)
    for p in ax_r100.patches:
        h = p.get_height()
        if h > 0:
            ax_r100.annotate(f"{h*100:.1f}%", (p.get_x() + p.get_width() / 2., h),
                             ha='center', va='bottom', fontsize=8, xytext=(0, 2),
                             textcoords='offset points')

    # ----------------------------------------------------
    # Panel (e): Puesto Mediano
    # ----------------------------------------------------
    ax_med = axes[2, 0]
    sns.barplot(data=df, x="vertical", y="puesto_mediano", hue="modelo", order=order_verts,
                palette=COLORES, ax=ax_med)
    ax_med.set_title("(e) Puesto Mediano del Ganador", fontsize=12, fontweight="bold")
    ax_med.set_ylabel("Puesto Mediano")
    for p in ax_med.patches:
        h = p.get_height()
        if h > 0:
            ax_med.annotate(f"{h:.1f}", (p.get_x() + p.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=8, xytext=(0, 2),
                            textcoords='offset points')

    # Ocultar el último subplot vacío de la grilla de 3x2
    axes[2, 1].axis('off')

    fig.suptitle("Comparativa de Modelos Baseline: b0 vs b0_v2 vs b0_v2_tuned",
                 fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(a.out, dpi=300)
    print(f"Gráfico comparativo guardado con éxito en: {a.out}")

    if not a.no_show:
        plt.show()


if __name__ == "__main__":
    main()