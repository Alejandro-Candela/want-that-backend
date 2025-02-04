
import numpy as np
from PIL import Image
# Segment-Anything
from segment_anything import SamPredictor, sam_model_registry

import sys
import os

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


def segment_with_sam(image_pil: Image.Image, boxes, multimask_output=False):
    """Dada una lista de bounding boxes (xyxy) y una imagen PIL,
       retorna una lista de máscaras (cada elemento es un np.ndarray binario).
    """
    image_np = np.array(image_pil)
    sam_predictor.set_image(image_np)

    masks = []
    for box in boxes:
        x0, y0, x1, y1 = box.tolist()
        input_box = np.array([x0, y0, x1, y1])

        mask_predictions, scores, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],  # 1 caja en batch
            multimask_output=multimask_output
        )

        best_mask_idx = np.argmax(scores)
        best_mask = mask_predictions[best_mask_idx]
        masks.append(best_mask)
        print(f"Segmentado: {best_mask.shape}")
    return masks
