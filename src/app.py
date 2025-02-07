from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import shutil
import os

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

@app.post("/api/search")
async def search_object(
    image: UploadFile,
    text_prompt: str = Form(...)
):
    # Asegurar que el nombre del archivo sea seguro
    safe_filename = os.path.basename(image.filename)
    temp_image_path = os.path.join("input", f"temp_{safe_filename}")
    
    try:
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        # Procesar imagen usando el pipeline
        results = process_image_pipeline(
            image_path=temp_image_path,
            text_prompt=text_prompt
        )
        
        return {"status": "success", "results": results}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
    finally:
        # Limpiar archivo temporal
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)