import numpy as np
from PIL import Image
# Segment-Anything
from segment_anything import SamPredictor, sam_model_registry

import sys
import os
import torch

current_dir = os.path.dirname(__file__)              # ruta de /src/modules/segmentation
src_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(src_dir)

from config import DEVICE, SAM_CHECKPOINT_PATH, SAM_MODEL_TYPE
# -------------------------
#   1) CARGAR MODELOS
# -------------------------

sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(
    DEVICE
)
sam_predictor = SamPredictor(sam_model)


def segment_with_sam(image_pil: Image.Image, box: torch.Tensor, multimask_output=False) -> tuple[np.ndarray, float]:
    """
    Segmenta una imagen usando SAM y retorna la máscara y el porcentaje de confianza.
    
    Returns:
        tuple[np.ndarray, float]: Máscara binaria y score de confianza
    """
    image_np = np.array(image_pil)
    sam_predictor.set_image(image_np)

    input_box = box.numpy()
    
    mask_predictions, scores, _ = sam_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=multimask_output
    )

    best_mask_idx = np.argmax(scores)
    confidence_score = float(scores[best_mask_idx])  # Convertir a float para serialización JSON
    return mask_predictions[best_mask_idx], confidence_score


def save_optimized_segmented_image(image_pil: Image.Image, mask: np.ndarray, output_path: str):
    """
    Guarda la imagen segmentada en formato JPEG optimizado con fondo blanco,
    recortada al bounding box con un margen adicional del 10%.
    
    Args:
        image_pil: Imagen original
        mask: Máscara binaria
        output_path: Ruta de salida para la imagen
    
    Returns:
        str: Ruta de la imagen guardada
    """
    # Encontrar los límites de la máscara
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    # Calcular el margen (10% del ancho de la región)
    width = x2 - x1
    margin = int(width * 0.1)
    
    # Ajustar las coordenadas con el margen, asegurando que no excedan los límites de la imagen
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(image_pil.width, x2 + margin)
    y2 = min(image_pil.height, y2 + margin)
    
    # Recortar la imagen y la máscara
    image_rgb = image_pil.convert("RGB")
    image_np = np.array(image_rgb)
    cropped_image = image_np[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    
    # Expandir la máscara a 3 canales
    mask_3channel = np.repeat(cropped_mask[:, :, np.newaxis], 3, axis=2)
    
    # Crear fondo blanco del mismo tamaño
    white_background = np.full_like(cropped_image, 255)
    
    # Combinar imagen original y fondo blanco
    segmented_image = np.where(mask_3channel, cropped_image, white_background).astype(np.uint8)
    
    # Convertir y guardar
    segmented_pil = Image.fromarray(segmented_image)
    segmented_pil.save(output_path, "JPEG", quality=50, optimize=True)
    
    return output_path
