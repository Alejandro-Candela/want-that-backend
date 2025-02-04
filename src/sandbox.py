import os
import base64
import requests
from io import BytesIO
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Librería para búsquedas en Google usando SerpAPI
from serpapi import GoogleSearch

# Segment-Anything
from segment_anything import SamPredictor, sam_model_registry

# GroundingDINO
from transformers import AutoProcessor, GroundingDinoForObjectDetection


# API key de SerpAPI (¡PON AQUÍ TU KEY!)
SERPAPI_API_KEY = "c2fc161fb6009ba04cb7129c31b621f7ce5df3507be95be93954857236b44158"

# Tu Client ID de Imgur (registro en: https://api.imgur.com/oauth2)
IMGUR_CLIENT_ID = "64e1f84dd892027"

# -------------------------
#   PARÁMETROS PRINCIPALES
# -------------------------
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-base"
SAM_CHECKPOINT_PATH  = "sam_vit_b_01ec64.pth"  # Ajusta a la ruta local del checkpoint si es necesario
SAM_MODEL_TYPE       = "vit_b"                # "vit_h", "vit_l", "vit_b", etc.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"            = "cuda" if torch.cuda.is_available() else "cpu"

# Asegúrate de que existe la carpeta de salida
os.makedirs("outputs", exist_ok=True)


# -------------------------
#   1) CARGAR MODELOS
# -------------------------
processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
model_dino = GroundingDinoForObjectDetection.from_pretrained(GROUNDING_DINO_MODEL).to(DEVICE).eval()

sam_model = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT_PATH).to(DEVICE)
sam_predictor = SamPredictor(sam_model)

# -------------------------
#   2) FUNCIONES AUXILIARES
# -------------------------
def preprocess_caption(caption: str) -> str:
    """Preprocesa el texto para GroundingDINO (minus, sin espacios extras, con punto al final)."""
    result = caption.lower().strip()
    if not result.endswith("."):
        result += "."
    return result


def get_grounding_dino_boxes(image: Image.Image, text_prompt: str, box_threshold=0.3, text_threshold=0.1):
    """Retorna las bounding boxes, scores, labels de GroundingDINO."""
    inputs = processor(images=image, text=preprocess_caption(text_prompt), return_tensors="pt").to(DEVICE)

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
    return results


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
    return masks


def mask_to_transparent_image(image_pil: Image.Image, mask: np.ndarray):
    """Convierte una máscara binaria (True/False) en una imagen RGBA con fondo transparente."""
    image_rgba = image_pil.convert("RGBA")
    image_array = np.array(image_rgba)

    alpha = (mask * 255).astype(np.uint8)  # 255 donde mask=True, 0 donde False
    image_array[:, :, 3] = alpha

    return Image.fromarray(image_array)


def draw_boxes_and_labels(image: Image.Image, results):
    """Dibuja las bounding boxes con etiquetas y scores sobre la imagen (matplotlib)."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)

    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        x0, y0, x1, y1 = box.tolist()
        w, h = x1 - x0, y1 - y0
        rect = patches.Rectangle((x0, y0), w, h, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        caption = f"{label} ({score:.2f})"
        ax.text(x0, y0 - 5, caption, color='red', fontsize=12,
                bbox=dict(facecolor='yellow', alpha=0.5))

    plt.axis('off')
    plt.show()
    
def search_similar_product_online(image_path: str) -> None:
    """
    1) Lee la imagen local desde `image_path`.
    2) La sube a Imgur (modo anónimo) y obtiene la URL pública.
    3) Hace una búsqueda inversa de la imagen en Google Lens (vía SerpAPI)
       usando la URL de Imgur.
    4) Muestra las coincidencias y resultados de shopping (si existen).
    """

    # Verificar la ruta del archivo
    if not os.path.isfile(image_path):
        print(f"No se encontró el archivo de imagen: {image_path}")
        return

    print(f"\nBuscando producto similar con Google Lens para la imagen: {image_path}\n")

    # --------------------------------------------------------------------
    # 1) Subir la imagen a Imgur
    # --------------------------------------------------------------------
    with open(image_path, "rb") as f:
        image_data = f.read()
    encoded_image = base64.b64encode(image_data).decode("utf-8")

    url_imgur = "https://api.imgur.com/3/upload"
    headers_imgur = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
    data_imgur = {
        "image": encoded_image,
        "type": "base64",
    }

    resp_imgur = requests.post(url_imgur, headers=headers_imgur, data=data_imgur)
    if resp_imgur.status_code != 200:
        print(f"Error al subir a Imgur: {resp_imgur.text}")
        return

    imgur_json = resp_imgur.json()
    if not imgur_json.get("success"):
        print(f"Error en respuesta de Imgur: {imgur_json}")
        return

    # URL pública devuelta por Imgur
    imgur_link = imgur_json["data"]["link"]
    print(f"Imagen subida con éxito a Imgur: {imgur_link}")

    # --------------------------------------------------------------------
    # 2) Búsqueda en Google Lens (vía SerpAPI)
    # --------------------------------------------------------------------
    params = {
        "engine": "google_lens",
        "api_key": SERPAPI_API_KEY,
        "hl": "en",           # idioma
        "url": imgur_link,  # usamos la URL de Imgur
    }

    search = GoogleSearch(params)

    # En caso de error, a veces SerpAPI retorna HTML (lo que genera JSONDecodeError).
    # Si quieres diagnosticar, podrías hacer:
    #   response_html = search.get_html()
    #   print("=== RAW RESPONSE ===", response_html)

    results = search.get_dict()
    print(results)

    # --------------------------------------------------------------------
    # 3) Analizar resultados - visual_matches
    # --------------------------------------------------------------------
    visual_matches = results.get("visual_matches", [])
    if visual_matches:
        print("=== Visual Matches ===")
        for i, match in enumerate(visual_matches[:5]):
            title = match.get("title", "Sin título")
            link = match.get("link") or "Link no disponible"
            thumbnail = match.get("thumbnail") or "Thumbnail no disponible"

            print(f"\nCoincidencia #{i+1}")
            print(f"Título: {title}")
            print(f"Link: {link}")
            print(f"Imagen/Thumbnail: {thumbnail}")
    else:
        print("No se encontraron coincidencias visuales en 'visual_matches'.")

    # --------------------------------------------------------------------
    # 4) Analizar resultados de 'inline_shopping_results'
    # --------------------------------------------------------------------
    inline_shopping = results.get("inline_shopping_results", [])
    if inline_shopping:
        print("\n=== Resultados de Compras (inline_shopping_results) ===")
        for i, item in enumerate(inline_shopping[:5]):
            product_title = item.get("title", "Sin título")
            product_link = item.get("link") or "Link no disponible"
            price = item.get("extracted_price", "Precio no disponible")

            print(f"\nProducto #{i+1}")
            print(f"Título: {product_title}")
            print(f"Link: {product_link}")
            print(f"Precio: {price}")
    else:
        print("\nNo se encontraron resultados de 'inline_shopping_results'.")

image_path = "input_lamp.jpg"
text_prompt = "the lamp"

# -------------------------
#  PIPELINE PRINCIPAL
# -------------------------

# 1) Cargar imagen
image_pil = Image.open(image_path).convert("RGB")

# 2) Obtener bounding boxes con GroundingDINO
results = get_grounding_dino_boxes(
    image=image_pil,
    text_prompt=text_prompt,
    box_threshold=0.3,
    text_threshold=0.1
)


# 3) Visualizar (opcional) la imagen con bounding boxes
draw_boxes_and_labels(image_pil, results)

# 4) Segmentar con SAM cada caja
boxes = results['boxes']  # [N, 4]
masks = segment_with_sam(image_pil, boxes)

# 5) Guardar objetos segmentados individualmente
for i, (mask, label, score) in enumerate(zip(masks, results['labels'], results['scores'])):
    segmented_image = mask_to_transparent_image(image_pil, mask)
    x0, y0, x1, y1 = boxes[i].tolist()
    x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
    cropped_segmented = segmented_image.crop((x0, y0, x1, y1))

    filename = f"output/object_{i}_label_{label}_score_{score:.2f}.png"
    cropped_segmented.save(filename)
    print(f"Guardado: {filename}")
    
    
    # 6) Identificar el objeto con mayor score y buscarlo en Internet
scores = results['scores']
if len(scores) == 0:
    print("No se detectó ningún objeto, no se realizará la búsqueda.")

max_score_idx = torch.argmax(scores).item()
best_label = results['labels'][max_score_idx]
best_score = scores[max_score_idx]

print(f"\nObjeto con mayor score: {best_label} (score={best_score:.2f})")

# 7) Buscar producto similar en Internet (Google Shopping vía SerpAPI)
search_similar_product_online("output/object_0_label_the lamp_score_0.76.png")