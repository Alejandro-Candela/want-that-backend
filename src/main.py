import sys
import os
import base64

from src.modules.search.load_to_supabase import load_to_supabase

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from PIL import Image
from modules.search.google_lens_search import search_similar_product_online
from modules.segmentation.grounding_dino import get_grounding_dino_boxes
from modules.segmentation.sam_segmentation import (
    save_optimized_segmented_image,
    segment_with_sam,
)
from config import BOX_THRESHOLD, TEXT_THRESHOLD, BUCKET_NAME

def process_image_pipeline(image_path: str, text_prompt: str) -> tuple[dict, list[str], str]:
    """
    Procesa una imagen a través del pipeline completo.
    
    Args:
        image_path: Ruta a la imagen a procesar
        text_prompt: Texto descriptivo del objeto a buscar
    
    Returns:
        tuple[dict, list[str], str]: Resultados del procesamiento, lista de pasos y URL de imagen segmentada
    """
    progress_steps = []
    
    # 1) Cargar imagen
    progress_steps.append("Loading image...")
    image_pil = Image.open(image_path).convert("RGB")
    progress_steps.append("Image loaded successfully")

    # 2) Obtener bounding box
    progress_steps.append("Detecting object in image...")
    best_box, best_score, used_prompt = get_grounding_dino_boxes(
        image=image_pil,
        text_prompt=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )
    progress_steps.append(f"Object detected with confidence score: {best_score:.2f}")

    # 3) Segmentar con SAM
    progress_steps.append("Segmenting object from background...")
    mask = segment_with_sam(image_pil, best_box)
    segmented_path = "output/segmented_object.webp"
    save_optimized_segmented_image(image_pil, mask, segmented_path)
    progress_steps.append("Object segmented successfully")

    # Convertir la imagen segmentada a base64
    with open(segmented_path, "rb") as image_file:
        segmented_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    # 4) Subir a Supabase y obtener URL
    progress_steps.append("Uploading segmented image...")
    imgur_url = load_to_supabase(segmented_path, BUCKET_NAME, best_score, used_prompt)
    progress_steps.append("Image uploaded successfully")

    # 5) Buscar productos similares
    progress_steps.append("Searching for similar products...")
    search_results = search_similar_product_online(imgur_url)
    progress_steps.append("Search completed successfully")

    return search_results, progress_steps, f"data:image/webp;base64,{segmented_base64}"
