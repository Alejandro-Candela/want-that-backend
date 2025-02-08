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


def save_optimized_segmented_image(image_pil, mask, output_path):
    """
    Guarda la imagen segmentada en formato JPEG optimizado con fondo blanco.
    
    Se asume que:
      - `image_pil` es una imagen PIL.
      - `mask` es un array binario (con valores 0 o 1) del mismo tamaño que la imagen.
      
    En las zonas donde la máscara es 1 se conserva la imagen original; en las zonas donde es 0 se rellena con blanco.
    
    La imagen se guarda en formato JPEG, que suele producir archivos de menor peso.
    
    Parámetros:
      image_pil (PIL.Image): Imagen original.
      mask (np.array): Array binario (0 y 1) con la máscara.
      output_path (str): Ruta donde se guardará la imagen optimizada (por ejemplo, "output.jpg").
    
    Retorna:
      str: La ruta de salida (output_path).
    """
    # Convertir la imagen a RGB (sin canal alfa)
    image_rgb = image_pil.convert("RGB")
    image_np = np.array(image_rgb)
    
    # Expandir la máscara a 3 canales para que coincida con la imagen (alto, ancho, 3)
    mask_3channel = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    
    # Crear una imagen de fondo blanco del mismo tamaño
    white_background = np.full_like(image_np, 255)
    
    # Combinar la imagen original y el fondo blanco:
    # Se conserva el pixel original donde la máscara es 1, y se usa blanco donde es 0.
    segmented_image = np.where(mask_3channel, image_np, white_background).astype(np.uint8)
    
    # Convertir el array resultante a una imagen PIL y guardarla en formato JPEG
    segmented_pil = Image.fromarray(segmented_image)
    segmented_pil.save(output_path, "JPEG", quality=50, optimize=True)
    
    return output_path
