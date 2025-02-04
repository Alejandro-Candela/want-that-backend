import torch
from PIL import Image
from modules.search.google_lens_search import search_similar_product_online
from modules.segmentation.grounding_dino import get_grounding_dino_boxes
from modules.segmentation.sam_segmentation import segment_with_sam
from utils.utils import (draw_boxes_and_labels, mask_to_transparent_image)




image_path = "input/input_lamp.jpg"
text_prompt = "the lamp"

# -------------------------
#  PIPELINE PRINCIPAL
# -------------------------


def main():
    # 1) Cargar imagen
    image_pil = Image.open(image_path).convert("RGB")

    # 2) Obtener bounding boxes con GroundingDINO
    results = get_grounding_dino_boxes(
        image=image_pil, text_prompt=text_prompt, box_threshold=0.3, text_threshold=0.1
    )

    # 3) Visualizar (opcional) la imagen con bounding boxes
    #draw_boxes_and_labels(image_pil, results)

    # 4) Segmentar con SAM cada caja
    boxes = results["boxes"]  # [N, 4]
    masks = segment_with_sam(image_pil, boxes)

    # 5) Guardar objetos segmentados individualmente
    for i, (mask, label, score) in enumerate(
        zip(masks, results["labels"], results["scores"])
    ):
        segmented_image = mask_to_transparent_image(image_pil, mask)
        x0, y0, x1, y1 = boxes[i].tolist()
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        cropped_segmented = segmented_image.crop((x0, y0, x1, y1))

        filename = f"output/object_{i}_label_{label}_score_{score:.2f}.png"
        cropped_segmented.save(filename)
        print(f"Guardado: {filename}")

        # 6) Identificar el objeto con mayor score y buscarlo en Internet
    scores = results["scores"]
    if len(scores) == 0:
        print("No se detectó ningún objeto, no se realizará la búsqueda.")

    max_score_idx = torch.argmax(scores).item()
    best_label = results["labels"][max_score_idx]
    best_score = scores[max_score_idx]

    print(f"\nObjeto con mayor score: {best_label} (score={best_score:.2f})")

    # 7) Buscar producto similar en Internet (Google Shopping vía SerpAPI)
    search_similar_product_online("output/object_0_label_the lamp_score_0.76.png")


if __name__ == "__main__":
    main()
