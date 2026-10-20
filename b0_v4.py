"""
Módulo B0_V4 - Investigathón 2026 (Track Física Social)
Modelado Híbrido: Combina la Frecuencia Histórica Temporal B0 con
Similitud Semántica de Texto (TF-IDF / Sentence Embeddings) entre Licitación y Empresa.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class B0V4:
    def __init__(self, data_dir: str, tau_years: float = 2.5, alpha: float = 0.7):
        self.data_dir = Path(data_dir)
        self.tau_days = tau_years * 365.25
        self.alpha = alpha  # Peso otorgado al B0 histórico vs el Texto (1-alpha)

        # Cargar historial de contratos
        self.historial = pd.read_parquet(self.data_dir / "historial.parquet")
        if "fecha" in self.historial.columns:
            self.historial["t"] = pd.to_datetime(self.historial["fecha"])
        else:
            self.historial["t"] = pd.to_datetime(self.historial["t"])

        # Vectorizador de Texto
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        
        # Pre-vectorizar descripciones del historial para optimizar velocidad
        hist_texts = self.historial["description"].fillna("").astype(str)
        self.hist_embeddings = self.vectorizer.fit_transform(hist_texts)

    def p_b0_v4(self, licitacion: pd.Series, candidatas: pd.DataFrame) -> pd.Series:
        t_q = pd.to_datetime(licitacion["t"])
        candidatas_list = candidatas["parent_uei"].unique()

        # Filtrar historial estricto (sin contaminación futura)
        mask_past = self.historial["t"] < t_q
        hist_past = self.historial[mask_past].copy()

        if hist_past.empty:
            return pd.Series(1.0 / len(candidatas_list), index=candidatas_list)

        # 1. COMPONENTE 1: SCORE B0 HISTÓRICO
        delta_days = (t_q - hist_past["t"]).dt.total_seconds() / (24 * 3600)
        hist_past["w"] = np.exp(-delta_days / self.tau_days)

        q_office = licitacion["office"]
        q_psc = licitacion["psc"]

        mask_office = (hist_past["office"] == q_office) & (hist_past["psc"] == q_psc)
        mask_psc = hist_past["psc"] == q_psc

        def calc_probs(mask):
            sub = hist_past[mask]
            tot = sub["w"].sum()
            if tot > 0:
                return (sub.groupby("parent_uei")["w"].sum() / tot).reindex(candidatas_list, fill_value=0.0)
            return pd.Series(0.0, index=candidatas_list)

        p_b0_score = 0.6 * calc_probs(mask_office) + 0.3 * calc_probs(mask_psc) + 0.1 * calc_probs(slice(None))

        # 2. COMPONENTE 2: SIMILITUD DE TEXTO
        lic_text = str(licitacion.get("title", "")) + " " + str(licitacion.get("description", ""))
        lic_vec = self.vectorizer.transform([lic_text])

        # Calcular similitud respecto a los contratos históricos filtrados
        hist_indices = hist_past.index.values
        sub_embeddings = self.hist_embeddings[hist_indices]
        sims = cosine_similarity(lic_vec, sub_embeddings).flatten()

        hist_past["sim"] = sims
        text_score = hist_past.groupby("parent_uei")["sim"].max().reindex(candidatas_list, fill_value=0.0)

        # 3. COMBINACIÓN HÍBRIDA
        final_score = self.alpha * p_b0_score + (1 - self.alpha) * text_score

        # Desempate determinístico
        seed = int(abs(hash(str(licitacion["id"])))) % (2**32)
        np.random.seed(seed)
        final_score += np.random.uniform(0, 1e-9, size=len(final_score))

        sum_s = final_score.sum()
        if sum_s > 0:
            final_score /= sum_s

        return final_score


def generar_ranking_b0_v4(data_dir: str, output_csv: str):
    path = Path(data_dir)
    licitaciones = pd.read_parquet(path / "licitaciones.parquet")
    empresas = pd.read_parquet(path / "empresas.parquet")

    b0_v4 = B0V4(data_dir=data_dir)
    rows = []

    for _, lic in licitaciones.iterrows():
        lic_dt = pd.to_datetime(lic["t"])
        semestre = f"{lic_dt.year}S1" if lic_dt.month <= 6 else f"{lic_dt.year}S2"

        candidatas = empresas[empresas["semestre"] == semestre]
        if candidatas.empty:
            continue

        probs = b0_v4.p_b0_v4(lic, candidatas)
        top_100 = probs.nlargest(100)

        for puesto, (uei, _) in enumerate(top_100.items(), start=1):
            rows.append({"id": lic["id"], "parent_uei": uei, "puesto": puesto})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_csv, index=False)
    print(f"Ranking B0_V4 generado exitosamente en: {output_csv}")


if __name__ == "__main__":
    import sys
    default_data_path = r"C:\Users\tomas\grupo-valores-robados\data\j"
    rubro_path = sys.argv[1] if len(sys.argv) > 1 else default_data_path
    out_path = sys.argv[2] if len(sys.argv) > 2 else "b0_v4_j.csv"

    generar_ranking_b0_v4(rubro_path, out_path)