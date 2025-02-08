import os
from dotenv import load_dotenv
from serpapi import GoogleSearch

load_dotenv()

def search_similar_product_online(image_url: str) -> list[dict]:
    """
    Hace una búsqueda inversa de la imagen en Google Lens (vía SerpAPI)
    y retorna las 5 primeras coincidencias visuales más relevantes.

    Args:
        image_url (str): URL pública de la imagen (por ejemplo, almacenada en AWS S3).

    Returns:
        list[dict]: Lista con las 5 primeras coincidencias visuales, cada una conteniendo
                   título, link y thumbnail. Lista vacía si no hay coincidencias.
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
    top_matches = []

    for match in visual_matches[:3]:
        top_matches.append({
            "title": match.get("title", "Sin título"),
            "link": match.get("link", "Link no disponible"),
            "thumbnail": match.get("thumbnail", "Thumbnail no disponible")
        })

    return top_matches
