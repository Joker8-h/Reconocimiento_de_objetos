
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io, os

app = FastAPI()

# ✅ CORS para permitir llamadas desde React
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en producción reemplaza por la URL real
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Ruta absoluta del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Cargar modelo YOLOv11 entrenado
MODEL_PATH = os.path.join(BASE_DIR, "yolo11s.pt")  # o best.pt si ya entrenaste
model = YOLO(MODEL_PATH)

# ✅ Clases del dataset (IMPORTANTE: en INGLÉS)
REQUIRED_ITEMS = ["lab_coat", "stethoscope"]


@app.post("/video-frame")
async def video_frame(file: UploadFile = File(...)):
    import numpy as np
    from PIL import Image

    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    results = model.predict(img, imgsz=640)

    detections = []
    missing = ["lab_coat", "stethoscope"]

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0])

            label = model.names[cls_id]

            detections.append({
                "class": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

            if label in missing:
                missing.remove(label)

    return {
        "detected": [d["class"] for d in detections],
        "missing": missing,
        "is_complete": len(missing) == 0,
        "boxes": detections
    }
