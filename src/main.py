import sys
import os

from modules.search.load_to_supabase import load_to_supabase

current_dir = os.path.dirname(__file__)
sys.path.append(current_dir)

from PIL import Image
from modules.search.google_lens_search import search_similar_product_online
from modules.segmentation.grounding_dino import get_grounding_dino_boxes
from modules.segmentation.sam_segmentation import (
    save_optimized_segmented_image,
    segment_with_sam,
)
from config import BOX_THRESHOLD, TEXT_THRESHOLD

input_image_path = "input/input.jpg"
text_prompt = "the chair"
output_path = "output/segmented_object.webp"  # Define output path for segmented image
bucket_name = "images-bucket"

# -------------------------
#  PIPELINE PRINCIPAL
# -------------------------
def main():
    # --- Configuración inicial ---
    # Ruta de la imagen original y texto del prompt

    # Bucket de Supabase (ya creado en la consola de Supabase)

    # --- Paso 1: Cargar y procesar la imagen ---
    image_pil = Image.open(input_image_path).convert("RGB")

    # --- Paso 2: Obtener la bounding box, score y prompt con Grounding DINO ---
    best_box, best_score, used_prompt = get_grounding_dino_boxes(
        image=image_pil,
        text_prompt=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # --- Paso 3: Segmentar el objeto con SAM ---
    # Se obtiene la máscara a partir de la imagen y la bounding box
    mask = segment_with_sam(image_pil, best_box)

    # Guardar la imagen segmentada (con fondo transparente) en formato WebP
    segmented_image_path = "segmented_object.webp"
    saved_path = save_optimized_segmented_image(image_pil, mask, segmented_image_path)

    # --- Paso 4: Subir la imagen segmentada a Supabase con metadata (score y prompt) ---
    imgur_url = load_to_supabase(saved_path, bucket_name, best_score, used_prompt)

    if imgur_url:
        print(f"Imagen subida exitosamente. URL: {imgur_url}")
    else:
        print("Error al subir la imagen a imgur.")
        

    # --- Paso 5: Búsqueda de productos similares usando la URL de la imagen en Supabase ---
    search_similar_product_online(imgur_url)


# Llama a la función principal
if __name__ == "__main__":
    main()
