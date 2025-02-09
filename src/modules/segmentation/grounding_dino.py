from PIL import Image
import torch
import sys
import os

current_dir = os.path.dirname(__file__)              # ruta de /src/modules/segmentation
src_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(src_dir)

from transformers import AutoProcessor, GroundingDinoForObjectDetection
from config import DEVICE, GROUNDING_DINO_MODEL
from utils.utils import preprocess_caption

processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
model_dino = (
    GroundingDinoForObjectDetection.from_pretrained(GROUNDING_DINO_MODEL)
    .to(DEVICE)
    .eval()
)


def get_grounding_dino_boxes(image: Image.Image, text_prompt: str, box_threshold: float, text_threshold: float):
    """
    Retorna la bounding box con el score más alto, su score y el text prompt.
    """
    inputs = processor(
        images=image, 
        text=preprocess_caption(text_prompt), 
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model_dino(**inputs)

    width, height = image.size
    postprocessed = processor.post_process_grounded_object_detection(
        outputs=outputs,
        input_ids=inputs.input_ids,
        target_sizes=[(height, width)],  # (alto, ancho)
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    results = postprocessed[0]  # Diccionario con 'scores', 'labels', 'boxes'
    max_score_index = results['scores'].argmax().item()

    best_box = results['boxes'][max_score_index]
    best_score = results['scores'][max_score_index]

    return best_box, best_score, text_prompt

