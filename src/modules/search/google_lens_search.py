import base64
import os
import requests
from dotenv import load_dotenv
from serpapi import GoogleSearch

load_dotenv()

# Librería para búsquedas en Google usando SerpAPI

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

    print(f"\nBuscando producto similar con Google Lens para la imagen: {
          image_path}\n")

    # --------------------------------------------------------------------
    # 1) Subir la imagen a Imgur
    # --------------------------------------------------------------------
    with open(image_path, "rb") as f:
        image_data = f.read()
    encoded_image = base64.b64encode(image_data).decode("utf-8")

    url_imgur = "https://api.imgur.com/3/upload"
    headers_imgur = {
        "Authorization": f"Client-ID {os.getenv('IMGUR_CLIENT_ID')}"}
    data_imgur = {
        "image": encoded_image,
        "type": "base64",
    }

    resp_imgur = requests.post(
        url_imgur, headers=headers_imgur, data=data_imgur, timeout=10)
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
        "api_key": os.getenv('SERPAPI_API_KEY'),
        "hl": "en",           # idioma
        "url": imgur_link,  # usamos la URL de Imgur
    }

    search = GoogleSearch(params)

    # En caso de error, a veces SerpAPI retorna HTML (lo que genera JSONDecodeError).
    # Si quieres diagnosticar, podrías hacer:
    #   response_html = search.get_html()
    #   print("=== RAW RESPONSE ===", response_html)

    results = search.get_dict()

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
