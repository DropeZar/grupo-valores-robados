# ==========================================================================================
#  COLAB — Embeddings bge-small del TEST OFICIAL (test-v1) para la ENTREGA
# ==========================================================================================
#  Genera test_<vert>_bges_emp.npy y test_<vert>_bges_lic.npy (los que lee predecir_test.py).
#  Construcción de texto IDÉNTICA al modelo final: foto de empresas al 1/7/2024, historial de
#  data/ cortado al 30/6/2024, instrucción de query de bge en las licitaciones.
#
#  CÓMO USARLO:
#   1. https://colab.research.google.com → nuevo notebook.
#   2. Runtime → Change runtime type → Hardware accelerator: GPU (T4 alcanza). bge-small NO
#      necesita token de HuggingFace.
#   3. Pegá cada bloque "CELDA N" en una celda separada y ejecutalas en orden.
#   4. Al final los .npy quedan en tu Drive (MyDrive/emb_out). Bajalos y copialos a tu
#      .emb_cache/ local; después corré predecir_test.py y salen los 5 CSV.
# ==========================================================================================


# ===================== CELDA 1 — GPU + dependencias =====================
# OJO: NO actualizar pandas/pyarrow (Colab ya los trae; subirlos rompe dependencias).
!nvidia-smi -L
!pip -q install -U "sentence-transformers>=5.0"


# ===================== CELDA 2 — Descargar data/ (histórico) y test/ =====================
# (si salteás esta celda, la CELDA 3 igual los baja: el chequeo está repetido a propósito)
import os, pathlib
REL = "https://github.com/bravoivanagustin/ranking-licitaciones/releases/download"
def asegurar_datos():
    if not pathlib.Path("data/rd/historial.parquet").exists():
        print("bajando data.zip…"); os.system(f"curl -sLO {REL}/datos-v1/data.zip && unzip -q -o data.zip")
    if not pathlib.Path("test/rd/licitaciones.parquet").exists():
        print("bajando test.zip…"); os.system(f"curl -sLO {REL}/test-v1/test.zip && unzip -q -o test.zip")
asegurar_datos()
print("data:", sorted(p.name for p in pathlib.Path("data").iterdir() if p.is_dir()))
print("test:", sorted(p.name for p in pathlib.Path("test").iterdir() if p.is_dir()))


# ===================== CELDA 3 — Montar Drive + config =====================
import numpy as np, pandas as pd, pathlib, os, torch
from sentence_transformers import SentenceTransformer
from google.colab import drive
drive.mount("/content/drive")       # autorizar una vez (popup)

# red de seguridad: si la CELDA 2 no corrió, bajamos acá igual
REL = "https://github.com/bravoivanagustin/ranking-licitaciones/releases/download"
if not pathlib.Path("data/rd/historial.parquet").exists():
    print("bajando data.zip…"); os.system(f"curl -sLO {REL}/datos-v1/data.zip && unzip -q -o data.zip")
if not pathlib.Path("test/rd/licitaciones.parquet").exists():
    print("bajando test.zip…"); os.system(f"curl -sLO {REL}/test-v1/test.zip && unzip -q -o test.zip")
assert pathlib.Path("data/rd/historial.parquet").exists(), "no se bajó data/ — revisá la conexión"
assert pathlib.Path("test/rd/licitaciones.parquet").exists(), "no se bajó test/ — revisá la conexión"

VERTS  = ["rd", "d", "j", "r", "it"]
MODELO = "BAAI/bge-small-en-v1.5"
SUF    = "bges"
INSTR  = "Represent this sentence for searching relevant passages: "
INICIO = pd.Timestamp("2024-07-01")   # foto del test: historial hasta el 30/6/2024

OUT = pathlib.Path("/content/drive/MyDrive/emb_out"); OUT.mkdir(parents=True, exist_ok=True)
D, T = pathlib.Path("data"), pathlib.Path("test")
psc = pd.read_parquet(D/"psc.parquet"); naics = pd.read_parquet(D/"naics.parquet"); off = pd.read_parquet(D/"office.parquet")
psc_desc = dict(zip(psc.codigo.astype(str).str.upper(), psc.descripcion.fillna("")))
naics_desc = dict(zip(naics.codigo.astype(str), naics.descripcion.fillna("")))
off_name = dict(zip(off.codigo.astype(str), off.oficina.fillna("")))

def armar_docs(vert):
    # MISMO orden que predecir_test.py (parquet + reset_index), para que las filas de los
    # .npy correspondan 1 a 1 con cand.parent_uei (emp) y con val (lic).
    val = pd.read_parquet(T/vert/"licitaciones.parquet").reset_index(drop=True)
    cand = pd.read_parquet(T/vert/"empresas.parquet").reset_index(drop=True)
    hist = pd.read_parquet(D/vert/"historial.parquet")
    doc_lic = (val.title.fillna("") + " . "
               + val.psc.map(lambda c: psc_desc.get(str(c).upper(), "")) + " . "
               + val.naics.map(lambda c: naics_desc.get(str(c), "")) + " . "
               + val.office.map(lambda c: off_name.get(str(c), ""))).tolist()
    h = hist.copy(); h["fecha"] = pd.to_datetime(h.fecha)
    h = h[h.fecha < INICIO]; h["psc"] = h.psc.astype(str).str.upper()
    ppe = h.groupby("parent_uei")["psc"].apply(lambda s: " ".join(psc_desc.get(p, "") for p in pd.unique(s)))
    doc_emp = (cand.texto_awards.fillna("") + " . " + cand.parent_uei.map(ppe).fillna("")).tolist()
    return doc_emp, doc_lic

print("config lista.")


# ===================== CELDA 4 — Codificar en GPU y guardar a Drive =====================
# Guarda cada archivo apenas lo genera y SALTEA los que ya están (reanuda si se corta).
model = None
for vert in VERTS:
    fe = OUT/f"test_{vert}_{SUF}_emp.npy"; fl = OUT/f"test_{vert}_{SUF}_lic.npy"
    if fe.exists() and fl.exists():
        print(f"  {vert}: ya está en Drive, salteo"); continue
    if model is None:
        model = SentenceTransformer(MODELO, device="cuda")
    de, dl = armar_docs(vert)
    C = model.encode(de, normalize_embeddings=True, convert_to_numpy=True,
                     batch_size=128, show_progress_bar=True).astype(np.float32)
    Q = model.encode([INSTR + t for t in dl], normalize_embeddings=True, convert_to_numpy=True,
                     batch_size=128, show_progress_bar=True).astype(np.float32)
    np.save(fe, C); np.save(fl, Q)
    print(f"  {vert}: emp {C.shape}  lic {Q.shape}  → Drive ✓")
if model is not None:
    del model; torch.cuda.empty_cache()
print("\n¡listo! los .npy están en MyDrive/emb_out/ (archivos test_<vert>_bges_*.npy)")


# ===================== CELDA 5 — (OPCIONAL) bajar un zip con solo estos archivos =====================
# !cd /content/drive/MyDrive/emb_out && zip -q /content/test_emb.zip test_*_bges_*.npy && echo ok
# from google.colab import files
# files.download("/content/test_emb.zip")
