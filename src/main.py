import sys
import os

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

def process_image_pipeline(image_path: str, text_prompt: str) -> dict:
    """
    Procesa una imagen a través del pipeline completo.
    
    Args:
        image_path: Ruta a la imagen a procesar
        text_prompt: Texto descriptivo del objeto a buscar
    
    Returns:
        dict: Resultados del procesamiento y búsqueda
    """
    # 1) Cargar imagen
    image_pil = Image.open(image_path).convert("RGB")

    # 2) Obtener bounding box
    best_box, best_score, used_prompt = get_grounding_dino_boxes(
        image=image_pil,
        text_prompt=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # 3) Segmentar con SAM
    mask = segment_with_sam(image_pil, best_box)
    segmented_path = "output/segmented_object.webp"
    save_optimized_segmented_image(image_pil, mask, segmented_path)

    # 4) Subir a Supabase y obtener URL
    imgur_url = load_to_supabase(segmented_path, BUCKET_NAME, best_score, used_prompt)

    # 5) Buscar productos similares
    search_results = search_similar_product_online(imgur_url)

    return search_results
