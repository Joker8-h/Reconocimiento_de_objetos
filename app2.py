from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io, os, requests

app = FastAPI()

# =====================================================
# CORS (restringir en producción)
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# CARGA DE MODELO
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")

model = YOLO(MODEL_PATH)

print("✅ Modelo cargado correctamente")
print("📌 Clases detectables:", model.names)

# ⚠️ AJUSTA ESTOS NOMBRES EXACTAMENTE COMO APARECEN EN model.names
REQUIRED_LICENCIA = [
    "colombia",
    "escudo_colombia",
    "nombre",
    "numerolic",
    "fechaexpedicion",
    "fechanacimiento",
    "foto",
    "ministerio_transporte",
    "tarjeta_conduccion",
    "titulo_licencia"
]


# =====================================================
# FUNCIÓN PRINCIPAL
# =====================================================
def analizar_licencia(img):

    results = model.predict(img, imgsz=640, conf=0.25)

    detected_classes = set()
    detections = []

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            label = model.names.get(cls_id, str(cls_id))
            detected_classes.add(label)

            detections.append({
                "class": label,
                "confidence": round(conf, 3),
                "bbox": [x1, y1, x2, y2]
            })

    # Comparación estructural
    required_set = set(REQUIRED_LICENCIA)
    missing = list(required_set - detected_classes)

    total_required = len(required_set)
    detected_valid = len(required_set & detected_classes)

    if total_required == 0:
        confianza = 0
    else:
        confianza = detected_valid / total_required

    sospecha_fraude = detected_valid < total_required

    return {
        "detected": list(detected_classes),
        "missing": missing,
        "sospecha_fraude": sospecha_fraude,
        "confianza": round(confianza, 2),
        "boxes": detections
    }


# =====================================================
# ENDPOINT: Subir imagen
# =====================================================
@app.post("/verificar-licencia-file")
async def verificar_licencia_file(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        return analizar_licencia(img)

    except Exception as e:
        return {"error": str(e)}


# =====================================================
# ENDPOINT: Imagen desde URL
# =====================================================
@app.post("/verificar-licencia-url")
async def verificar_licencia_url(payload: dict):
    try:
        image_url = payload.get("image_url")
        if not image_url:
            return {"error": "Falta image_url"}

        response = requests.get(image_url, timeout=10)

        if response.status_code != 200:
            return {"error": "No se pudo descargar la imagen"}

        img = Image.open(io.BytesIO(response.content)).convert("RGB")

        return analizar_licencia(img)

    except Exception as e:
        return {"error": str(e)}