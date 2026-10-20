"""
Módulo B0_V3 Optimizado - Investigathón 2026 (Track Física Social).
Implementa clustering dinámico adaptativo (K-Means + Embeddings) sobre códigos PSC.
Selecciona el 'k' óptimo minimizando la entropía ponderada de masa de contratos.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class OptimizationManager:
    """Manejador para la vectorización y clustering adaptativo 'caso por caso' de los PSC."""

    def __init__(self, psc_df: pd.DataFrame):
        self.psc_df = psc_df.copy()

        # Generar embeddings TF-IDF de las descripciones textuales
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        descriptions = self.psc_df["descripcion"].fillna("").astype(str)
        self.embeddings = self.vectorizer.fit_transform(descriptions).toarray()

    def find_optimal_k(self, historial_df: pd.DataFrame, max_k: int = 15) -> int:
        """Calcula de forma iterativa la cota/granularidad k óptima.

        Balancea la masa crítica acumulada contra el tamaño/variabilidad del
        grupo usando Entropía de la distribución + penalización de desbalance.
        """
        psc_counts = historial_df["psc"].value_counts().to_dict()
        n_samples = len(self.psc_df)

        if n_samples <= 2:
            return 1

        k_candidates = range(2, min(max_k, n_samples))
        best_k = 2
        min_score = float("inf")

        for k in k_candidates:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = kmeans.fit_predict(self.embeddings)

            temp_df = self.psc_df.copy()
            temp_df["cluster"] = labels
            temp_df["history_count"] = temp_df["codigo"].map(psc_counts).fillna(0)

            # Suma de contratos por cluster
            cluster_sizes = temp_df.groupby("cluster")["history_count"].sum()
            total_contracts = cluster_sizes.sum()

            if total_contracts == 0:
                continue

            # Probabilidades de masa por cluster
            probs = cluster_sizes / total_contracts
            probs = probs[probs > 0]

            # Entropía de la información (Métrica de dispersión)
            entropy = -np.sum(probs * np.log2(probs))

            # Penalización por hipertrofia / desbalance de conjuntos
            size_penalty = np.std(cluster_sizes) / (
                np.mean(cluster_sizes) + 1e-5
            )

            # Puntuación global a minimizar
            score = entropy + 0.1 * size_penalty

            if score < min_score:
                min_score = score
                best_k = k

        return best_k

    def build_psc_clusters(self, historial_df: pd.DataFrame) -> dict:
        """Construye la tabla Hash/Mapa {codigo_psc: cluster_id}."""
        optimal_k = self.find_optimal_k(historial_df)
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(self.embeddings)

        return dict(zip(self.psc_df["codigo"], clusters))


class B0V3:
    """Módulo B0_V3 integrable con evaluar.py y adaptativo por rubro."""

    def __init__(self, data_dir: str, tau_years: float = 2.5):
        self.data_dir = Path(data_dir)
        self.tau_days = tau_years * 365.25

        # Carga y normalización de fechas del historial
        self.historial = pd.read_parquet(self.data_dir / "historial.parquet")

        # Corrección de nombres de columna de fecha (soporta 'fecha' o 't')
        if "fecha" in self.historial.columns:
            self.historial["t"] = pd.to_datetime(self.historial["fecha"])
        else:
            self.historial["t"] = pd.to_datetime(self.historial["t"])

        # Cargar diccionario PSC (ubicado en el directorio raíz de data)
        psc_path = self.data_dir.parent / "psc.parquet"
        if not psc_path.exists():
            psc_path = self.data_dir / "psc.parquet"

        psc_df = pd.read_parquet(psc_path)

        # Normalización de nombres de columnas del PSC
        if "codigo" not in psc_df.columns and "psc" in psc_df.columns:
            psc_df = psc_df.rename(columns={"psc": "codigo"})

        # Optimización caso por caso de la taxonomía del PSC
        optimizer = OptimizationManager(psc_df)
        self.psc_cluster_map = optimizer.build_psc_clusters(self.historial)

        # Mapear el ID de cluster asignado al historial
        self.historial["psc_cluster"] = (
            self.historial["psc"].map(self.psc_cluster_map).fillna(-1)
        )

    def p_b0(self, licitacion: pd.Series, candidatas: pd.DataFrame) -> pd.Series:
        """Calcula el vector de probabilidades B0_V3 para un set de empresas candidatas."""
        t_q = pd.to_datetime(licitacion["t"])

        # Filtrado estricto: Solo contratos pasados (t < t_q)
        hist = self.historial[self.historial["t"] < t_q].copy()

        candidatas_list = candidatas["parent_uei"].unique()
        if hist.empty:
            return pd.Series(1.0 / len(candidatas_list), index=candidatas_list)

        # Ponderación con decaimiento exponencial
        delta_days = (t_q - hist["t"]).dt.total_seconds() / (24 * 3600)
        hist["w"] = np.exp(-delta_days / self.tau_days)

        q_psc = licitacion["psc"]
        q_cluster = self.psc_cluster_map.get(q_psc, -1)
        q_office = licitacion["office"]
        q_naics = licitacion["naics"]

        # Mascaras de los niveles de filtrado refinados
        mask_office = (hist["psc_cluster"] == q_cluster) & (
            hist["office"] == q_office
        )
        mask_cluster = hist["psc_cluster"] == q_cluster
        mask_naics = hist["naics"] == q_naics

        # Pesos por mezcla de niveles
        lambda_office = 0.40
        lambda_cluster = 0.30
        lambda_naics = 0.20
        lambda_global = 0.10

        prob_vec = pd.Series(0.0, index=candidatas_list)

        def calc_level_probs(mask):
            sub_hist = hist[mask]
            tot_w = sub_hist["w"].sum()
            if tot_w > 0:
                emp_w = sub_hist.groupby("parent_uei")["w"].sum()
                return (emp_w / tot_w).reindex(candidatas_list, fill_value=0.0)
            return pd.Series(0.0, index=candidatas_list)

        # Mezcla lineal
        prob_vec += lambda_office * calc_level_probs(mask_office)
        prob_vec += lambda_cluster * calc_level_probs(mask_cluster)
        prob_vec += lambda_naics * calc_level_probs(mask_naics)
        prob_vec += lambda_global * calc_level_probs(slice(None))

        # Desempate determinístico vía Hashing de ID de licitación
        seed = int(abs(hash(str(licitacion["id"])))) % (2**32)
        np.random.seed(seed)
        prob_vec += np.random.uniform(0, 1e-9, size=len(prob_vec))

        # Normalización
        sum_p = prob_vec.sum()
        if sum_p > 0:
            prob_vec /= sum_p

        return prob_vec


def generar_ranking_b0_v3(data_dir: str, output_csv: str):
    """Genera el CSV ajustado a las especificaciones de la entrega."""
    path = Path(data_dir)
    licitaciones = pd.read_parquet(path / "licitaciones.parquet")
    empresas = pd.read_parquet(path / "empresas.parquet")

    b0_v3 = B0V3(data_dir=data_dir)

    rows = []
    print(f"Procesando {len(licitaciones)} licitaciones para {data_dir}...")

    for _, lic in licitaciones.iterrows():
        lic_dt = pd.to_datetime(lic["t"])
        semestre = f"{lic_dt.year}S1" if lic_dt.month <= 6 else f"{lic_dt.year}S2"

        candidatas = empresas[empresas["semestre"] == semestre]
        if candidatas.empty:
            continue

        probs = b0_v3.p_b0(lic, candidatas)
        top_100 = probs.nlargest(100)

        for puesto, (uei, _) in enumerate(top_100.items(), start=1):
            rows.append({"id": lic["id"], "parent_uei": uei, "puesto": puesto})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print(f"Ranking generado exitosamente en: {output_csv}")


if __name__ == "__main__":
    import sys

    # Path absoluto/relativo parametrizado con fallback a la ruta de Windows
    default_data_path = (
        r"C:\Users\tomas\OneDrive\Desktop\Luciana\Investigathon\data\j"
    )

    rubro_path = sys.argv[1] if len(sys.argv) > 1 else default_data_path
    out_path = sys.argv[2] if len(sys.argv) > 2 else "b0_v3_j.csv"

    generar_ranking_b0_v3(rubro_path, out_path)