import time
import requests
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def preprocess_caption(caption: str) -> str:
    """Preprocesa el texto para GroundingDINO (minus, sin espacios extras, con punto al final)."""
    result = caption.lower().strip()
    if not result.endswith("."):
        result += "."
    return result

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
    
    import time
import requests

def wait_for_url(url: str, timeout: int = 300, interval: int = 2) -> bool:
    """
    Espera a que la URL esté disponible, realizando peticiones cada 'interval' segundos,
    hasta un máximo de 'timeout' segundos.
    
    Retorna:
        bool: True si la URL responde con un status_code 200 dentro del timeout, False en caso contrario.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return True
        except Exception:
            # Ignorar errores y volver a intentar
            pass
        time.sleep(interval)
    return False
