import os
from dotenv import load_dotenv
from serpapi import GoogleSearch

load_dotenv()

# Librería para búsquedas en Google usando SerpAPI

import os
from serpapi import GoogleSearch

def search_similar_product_online(image_url: str) -> None:
    """
    Hace una búsqueda inversa de la imagen en Google Lens (vía SerpAPI)
    usando la URL pública de AWS S3 y muestra las coincidencias visuales y 
    resultados de shopping (si existen).

    Parámetros:
        image_url (str): URL pública de la imagen (por ejemplo, almacenada en AWS S3).
    """

    print(f"\nBuscando producto similar con Google Lens para la imagen: {image_url}\n")

    # --------------------------------------------------------------------
    # Búsqueda inversa en Google Lens (vía SerpAPI)
    # --------------------------------------------------------------------
    params = {
        "engine": "google_lens",
        "api_key": os.getenv('SERPAPI_API_KEY'),
        "hl": "en",  # idioma
        "url": image_url,  # Se utiliza la URL de AWS S3
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    # --------------------------------------------------------------------
    # Procesar coincidencias visuales (visual_matches)
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
    # Procesar resultados de shopping (inline_shopping_results)
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
