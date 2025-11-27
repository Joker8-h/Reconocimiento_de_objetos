from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io, base64, os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

model = YOLO(MODEL_PATH)

# Elementos requeridos
REQUIRED = ["lab_coat", "stethoscope"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Fuente para texto
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    results = model.predict(img, conf=0.25)
    boxes = results[0].boxes

    detected = []

    # --- DIBUJAR DETECTADOS (VERDE) ---
    for box in boxes:
        cls_id = int(box.cls)
        label = model.names[cls_id]
        detected.append(label)

        x1, y1, x2, y2 = box.xyxy[0]

        # cuadro verde
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
        draw.text((x1, y1 - 20), label, fill="lime", font=font)

    # Elementos faltantes
    missing = [x for x in REQUIRED if x not in detected]

    # --- DIBUJAR FALTANTES (ROJO) ---
    if missing:
        width, height = img.size
        x_center = width // 2 - 150
        y_center = height // 2 - 150

        y_offset = 0
        for item in missing:
            # Caja roja centrada (simulada)
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
    img.save(buffer, format="JPEG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode()

    return JSONResponse({
        "detected_items": detected,
        "missing_items": missing,
        "is_fully_equipped": len(missing) == 0,
        "message": "🟢 Completo" if not missing else "🔴 Faltan elementos",
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

