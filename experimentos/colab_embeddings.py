# ==========================================================================================
#  COLAB — Codificar con modelos grandes (bge-base / EmbeddingGemma) en GPU
# ==========================================================================================
#  CÓMO USARLO:
#   1. Abrí https://colab.research.google.com  →  nuevo notebook.
#   2. Runtime → Change runtime type → Hardware accelerator: GPU (T4 alcanza).
#   3. Para Gemma: aceptá la licencia en huggingface.co/google/embeddinggemma-300m,
#      creá un token (hf_...), y en Colab: panel izquierdo → llave 🔑 (Secrets) →
#      add secret  nombre: HF_TOKEN  valor: tu token  →  activá "Notebook access".
#      (bge-base no necesita token.)
#   4. Pegá cada bloque "CELDA N" en una celda separada y ejecutalas en orden.
#   5. Al final se descarga embeddings.zip. Descomprimilo dentro de tu .emb_cache/ local.
#
#  Resultado: archivos {vertical}_{semestre}_{modelo}_emp.npy y _lic.npy, con el MISMO
#  nombre que usan los scripts locales (enchufan directo con SUF="bgebase" o "gemma").
# ==========================================================================================


# ===================== CELDA 1 — GPU + instalar dependencias =====================
# OJO: NO actualizamos pandas/pyarrow (Colab ya los trae y subirlos rompe dependencias).
!nvidia-smi -L
!pip -q install -U "sentence-transformers>=5.0" "transformers>=4.51"


# ===================== CELDA 2 — Autenticar HuggingFace (para Gemma) =====================
import os
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    from huggingface_hub import login
    login(os.environ["HF_TOKEN"])
    print("HF autenticado ✓")
except Exception as e:
    print("Sin HF_TOKEN (está OK para bge-base; Gemma lo necesita):", e)


# ===================== CELDA 3 — Descargar los datos =====================
import pathlib
if not pathlib.Path("data/j/historial.parquet").exists():
    !curl -sLO https://github.com/bravoivanagustin/ranking-licitaciones/releases/download/datos-v1/data.zip
    !unzip -q data.zip
print("verticales:", [p.name for p in pathlib.Path("data").iterdir() if p.is_dir()])


# ===================== CELDA 4 — Montar Drive + config =====================
import numpy as np, pandas as pd, pathlib, torch
from sentence_transformers import SentenceTransformer
from google.colab import drive
drive.mount("/content/drive")       # te pide autorizar una vez (abre un popup)

VERTS  = ["rd", "d", "j", "r", "it"]
SEMS   = ["2023S2"]                 # dev. Agregá "2024S1" para el test cuando quieras.
MODELS = {                          # comentá el que no quieras correr
    "bgebase": "BAAI/bge-base-en-v1.5",
    "gemma":   "google/embeddinggemma-300m",
}

# Los .npy se guardan DIRECTO en tu Google Drive → persisten aunque se corte Colab/la compu.
OUT = pathlib.Path("/content/drive/MyDrive/emb_out"); OUT.mkdir(parents=True, exist_ok=True)
D = pathlib.Path("data")
psc = pd.read_parquet(D/"psc.parquet"); naics = pd.read_parquet(D/"naics.parquet"); off = pd.read_parquet(D/"office.parquet")
psc_desc = dict(zip(psc.codigo.astype(str).str.upper(), psc.descripcion.fillna("")))
naics_desc = dict(zip(naics.codigo.astype(str), naics.descripcion.fillna("")))
off_name = dict(zip(off.codigo.astype(str), off.oficina.fillna("")))

def sem_inicio(sem):
    return pd.Timestamp(int(sem[:4]), 1 if sem[5] == "1" else 7, 1)

def armar_docs(vert, sem):
    lic = pd.read_parquet(D/vert/"licitaciones.parquet")
    emp = pd.read_parquet(D/vert/"empresas.parquet")
    hist = pd.read_parquet(D/vert/"historial.parquet")
    lic["sem"] = lic.t.map(lambda t: f"{t.year}S{1 if t.month <= 6 else 2}")
    val = lic[lic["sem"] == sem].copy()
    cand = emp[emp.semestre == sem].reset_index(drop=True)
    doc_lic = (val.title.fillna("") + " . "
               + val.psc.map(lambda c: psc_desc.get(str(c).upper(), "")) + " . "
               + val.naics.map(lambda c: naics_desc.get(str(c), "")) + " . "
               + val.office.map(lambda c: off_name.get(str(c), ""))).tolist()
    h = hist.copy(); h["fecha"] = pd.to_datetime(h.fecha)
    h = h[h.fecha < sem_inicio(sem)]; h["psc"] = h.psc.astype(str).str.upper()
    ppe = h.groupby("parent_uei")["psc"].apply(lambda s: " ".join(psc_desc.get(p, "") for p in pd.unique(s)))
    doc_emp = (cand.texto_awards.fillna("") + " . " + cand.parent_uei.map(ppe).fillna("")).tolist()
    return doc_emp, doc_lic

def codificar(model, name, doc_emp, doc_lic):
    if name == "gemma":   # prompts recomendados de EmbeddingGemma
        emp_in = ["title: none | text: " + t for t in doc_emp]
        lic_in = ["task: search result | query: " + t for t in doc_lic]
    else:                  # bge: instrucción de query
        instr = "Represent this sentence for searching relevant passages: "
        emp_in = doc_emp
        lic_in = [instr + t for t in doc_lic]
    C = model.encode(emp_in, normalize_embeddings=True, convert_to_numpy=True, batch_size=128, show_progress_bar=True)
    Q = model.encode(lic_in, normalize_embeddings=True, convert_to_numpy=True, batch_size=128, show_progress_bar=True)
    return C.astype(np.float32), Q.astype(np.float32)

print("config lista. Modelos:", list(MODELS), "| semestres:", SEMS)


# ===================== CELDA 5 — Codificar todo (la parte que usa la GPU) =====================
# Guarda cada archivo en Drive apenas lo genera, y SALTEA los que ya están
# (así, si se corta, lo volvés a correr y reanuda donde quedó).
for name, repo in MODELS.items():
    print(f"\n### {name}  ({repo})")
    model = None
    for vert in VERTS:
        for sem in SEMS:
            fe = OUT/f"{vert}_{sem}_{name}_emp.npy"; fl = OUT/f"{vert}_{sem}_{name}_lic.npy"
            if fe.exists() and fl.exists():
                print(f"  {vert} {sem}: ya está en Drive, salteo"); continue
            if model is None:
                model = SentenceTransformer(repo, device="cuda")   # carga solo si hace falta
            de, dl = armar_docs(vert, sem)
            C, Q = codificar(model, name, de, dl)
            np.save(fe, C); np.save(fl, Q)      # ← se guarda en Drive al toque
            print(f"  {vert} {sem}: emp {C.shape}  lic {Q.shape}  → Drive ✓")
    if model is not None:
        del model; torch.cuda.empty_cache()
print("\n¡listo! los .npy están en tu Drive: MyDrive/emb_out/")


# ===================== CELDA 6 — (OPCIONAL) bajar un zip =====================
# Ya NO hace falta: los .npy están en tu Drive (MyDrive/emb_out), aunque se corte Colab.
# Para traerlos a la compu: entrá a drive.google.com → carpeta emb_out → descargá.
# (O si querés un zip ahora, descomentá estas 3 líneas:)
# !cd /content/drive/MyDrive/emb_out && zip -q -r /content/embeddings.zip . && echo "zip creado"
# from google.colab import files
# files.download("/content/embeddings.zip")
