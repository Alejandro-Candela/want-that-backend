from fastapi import FastAPI, UploadFile, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from typing import List
from src.main import process_image_pipeline

app = FastAPI()

# Crear directorios necesarios si no existen
os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especifica los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Almacenar conexiones WebSocket activas
active_connections: List[WebSocket] = []


async def broadcast_progress(message: str):
    """Envía mensaje a todas las conexiones activas"""
    for connection in active_connections:
        await connection.send_json({"progress": message})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        active_connections.remove(websocket)


@app.post("/api/search")
async def search_object(image: UploadFile, text_prompt: str = Form(...)):
    safe_filename = os.path.basename(image.filename)
    temp_image_path = os.path.join("input", f"temp_{safe_filename}")

    try:
        # Paso 1: Guardar imagen
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Procesar imagen usando el pipeline y obtener resultados y mensajes
        results, progress_steps, segmented_image, segmentation_score = (
            await process_image_pipeline(
                image_path=temp_image_path,
                text_prompt=text_prompt,
                progress_callback=broadcast_progress,
            )
        )

        return {
            "status": "success",
            "results": results,
            "progress_steps": progress_steps,
            "segmented_image": segmented_image,
            "segmentation_score": segmentation_score,
        }

    except Exception as e:
        await broadcast_progress(f"Error: {str(e)}")
        return {"status": "error", "message": str(e)}

    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
