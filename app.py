from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io, base64, os

import easyocr
import numpy as np

app = FastAPI()

# Inicializar EasyOCR (Español e Inglés)
reader = easyocr.Reader(['es', 'en'])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "runs/detect/train/weights/best.pt")

model = YOLO(MODEL_PATH)

# Elementos requeridos en la licencia
REQUIRED_ITEMS = [
    "colombia", "escudo de colombia", "fechaexpedicion", "fechanacimiento",
    "foto", "ministerio de transporte", "nombre", "numerolic",
    "tarjeta conduccion", "titulo licencia"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_np = np.array(img_pil)
    draw = ImageDraw.Draw(img_pil)

    # Fuente para texto
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    results = model.predict(img_pil, conf=0.25)
    boxes = results[0].boxes

    detected = []
    extracted_data = {}

    # --- DIBUJAR DETECTADOS (VERDE) Y EXTRAER TEXTO ---
    for box in boxes:
        cls_id = int(box.cls)
        label = model.names[cls_id]
        detected.append(label)

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Recortar área para OCR si es un campo de interés
        if label in ["nombre", "numerolic", "fechaexpedicion"]:
            # Pequeño margen para mejor lectura
            crop = img_np[max(0, y1-5):min(img_np.shape[0], y2+5), max(0, x1-5):min(img_np.shape[1], x2+5)]
            ocr_result = reader.readtext(crop, detail=0)
            if ocr_result:
                extracted_data[label] = " ".join(ocr_result)

        # cuadro verde
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
        draw.text((x1, y1 - 20), f"{label}: {extracted_data.get(label, '')}", fill="lime", font=font)

    # Elementos faltantes
    missing = [x for x in REQUIRED_ITEMS if x not in detected]

    # --- DIBUJAR FALTANTES (ROJO) ---
    if missing:
        width, height = img_pil.size
        x_center = width // 2 - 150
        y_center = height // 2 - 150

        y_offset = 0
        for item in missing:
            draw.rectangle(
                [x_center, y_center + y_offset, x_center + 300, y_center + 50 + y_offset],
                outline="red",
                width=4
            )
            draw.text(
                (x_center + 10, y_center + 10 + y_offset),
                f"Falta: {item}",
                fill="red",
                font=font
            )
            y_offset += 70

    # Convertir a base64
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return JSONResponse({
        "detected_items": detected,
        "missing_items": missing,
        "is_fully_equipped": len(missing) == 0,
        "data": extracted_data,
        "message": "🟢 Licencia Válida" if not missing else "🔴 Licencia Sospechosa",
        "image_base64": img_base64
    })
# --- Endpoint rápido para video frame ---
@app.post("/video-frame")
async def video_frame(file: UploadFile = File(...)):
    # Leer bytes → imagen
    img_bytes = await file.read()

    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return {
            "error": "Frame inválido",
            "detected": [],
            "missing": REQUIRED_ITEMS,
            "is_complete": False,
            "boxes": []
        }

    # Correr predicción
    results = model.predict(img, imgsz=640, conf=0.25, verbose=False)

    detections = []
    missing = REQUIRED_ITEMS.copy()

    r = results[0]

    # Si YOLO no detecta nada
    if r.boxes is None or len(r.boxes) == 0:
        return {
            "detected": [],
            "missing": REQUIRED_ITEMS,
            "is_complete": False,
            "boxes": []
        }

    # Procesar detecciones
    for box in r.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(float, box.xyxy[0])

        label = model.names[cls_id]

        detections.append({
            "class": label,
            "confidence": round(conf, 2),
            "bbox": [x1, y1, x2, y2]
        })

        # Quitar de la lista de faltantes
        if label in missing:
            missing.remove(label)

    return {
        "detected": list({d["class"] for d in detections}),  # sin duplicados
        "missing": missing,
        "is_complete": len(missing) == 0,
        "boxes": detections
    }

