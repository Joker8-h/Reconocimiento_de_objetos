from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import os

app = FastAPI()

# Cargar el modelo de YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Modelo preentrenado

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Leer imagen recibida
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Realizar inferencia con YOLO
    results = model(image)
    
    # Devolver los resultados
    result_json = results.pandas().xywh[0].to_dict(orient="records")  # Convertir resultados a diccionario
    return JSONResponse(content={"predictions": result_json})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
