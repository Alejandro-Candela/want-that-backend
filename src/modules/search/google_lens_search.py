import os
from dotenv import load_dotenv
from serpapi import GoogleSearch
from typing import List, Dict
from google.cloud import vision
from google.api_core import retry
from google.oauth2 import service_account
import logging

load_dotenv()

logger = logging.getLogger(__name__)

def search_similar_product_online(image_url: str) -> list[dict]:
    """
    Hace una búsqueda inversa de la imagen en Google Lens (vía SerpAPI)
    y retorna las 5 primeras coincidencias visuales más relevantes.

    Args:
        image_url (str): URL pública de la imagen (por ejemplo, almacenada en AWS S3).

    Returns:
        list[dict]: Lista con las 5 primeras coincidencias visuales, cada una conteniendo:
                   title, link, thumbnail y price. Lista vacía si no hay coincidencias.
    """

    print(f"\nBuscando producto similar con Google Lens para la imagen: {image_url}\n")

    # --------------------------------------------------------------------
    # Búsqueda inversa en Google Lens (vía SerpAPI)
    # --------------------------------------------------------------------
    params = {
        "engine": "google_lens",
        "api_key": os.getenv("SERPAPI_API_KEY"),
        "hl": "es",  # idioma
        "url": image_url,  # Se utiliza la URL de imgur
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    # --------------------------------------------------------------------
    # Procesar coincidencias visuales (visual_matches)
    # --------------------------------------------------------------------
    visual_matches = results.get("visual_matches", [])
    top_matches = []

    for match in visual_matches[:5]:
        top_matches.append(
            {
                "title": match.get("title", "Sin título"),
                "link": match.get("link", "Link no disponible"),
                "thumbnail": match.get("thumbnail", "Thumbnail no disponible"),
                "price": match.get("price", "Precio no disponible"),
            }
        )

    # Imprimir en consola los resultados formateados
    print("Resultados de productos similares:")
    for product in top_matches:
        print(f"Title: {product['title']}")
        print(f"Link: {product['link']}")
        print(f"Thumbnail: {product['thumbnail']}")
        print(f"Price: {product['price']}")
        print("-------------")
    
    return top_matches

def get_vision_client():
    """Initialize Vision API client with credentials"""
    try:
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not credentials_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
            
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        return vision.ImageAnnotatorClient(credentials=credentials)
    except Exception as e:
        logger.error(f"Error initializing Vision API client: {str(e)}")
        raise

def get_product_search_client():
    """Initialize Product Search client with credentials"""
    try:
        credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        if not credentials_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
            
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path
        )
        return vision.ProductSearchClient(credentials=credentials)
    except Exception as e:
        logger.error(f"Error initializing Product Search client: {str(e)}")
        raise

@retry.Retry(predicate=retry.if_transient_error)
def product_search(image_url: str, 
                  project_id: str = "item-finder-alejandro",  # From your credentials
                  location: str = "us-west1",
                  product_set_id: str = "your_product_set_id",
                  product_category: str = "general-goods",
                  max_results: int = 5) -> List[Dict]:
    """
    Searches for visually similar products using Vision API Product Search.

    Args:
        image_url (str): Public URL of the image to search
        project_id (str): Google Cloud project ID
        location (str): Google Cloud region (e.g., "us-west1")
        product_set_id (str): ID of the product set to search in
        product_category (str): Category of products to search
        max_results (int): Maximum number of results to return

    Returns:
        List[Dict]: List of matching products, each containing:
            - name: Product name
            - display_name: Display name
            - description: Product description
            - score: Confidence score
            - image_url: Product image URL
            
    Raises:
        vision.ImageAnnotatorError: If the API request fails
        ValueError: If the input parameters are invalid
    """
    try:
        # Initialize clients with credentials
        image_annotator_client = get_vision_client()
        product_search_client = get_product_search_client()
        

        # Validate inputs
        if not image_url or not image_url.startswith(('http://', 'https://')):
            raise ValueError("Invalid image URL provided")

        # Construct product set path
        product_set_path = product_search_client.product_set_path(
            project=project_id,
            location=location,
            product_set=product_set_id
        )

        # Create image source and context
        image_source = vision.ImageSource(image_uri=image_url)
        image = vision.Image(source=image_source)
        
        # Configure product search parameters
        product_search_params = vision.ProductSearchParams(
            product_set=product_set_path,
            product_categories=[product_category],
            filter="",  # Add filters if needed
        )
        
        image_context = vision.ImageContext(
            product_search_params=product_search_params
        )
        

        # Perform the product search including image_context
        response = image_annotator_client.product_search(
            image=image,
            image_context=image_context
        )
        
        # Se elimina la impresión cruda y se prepara la salida formateada

        # Extract and format results
        results = response.product_search_results.results
        top_matches = []
        


        for result in results[:max_results]:
            product = result.product
            match = {
                "name": product.name.split('/')[-1],  # Extract ID from full path
                "display_name": product.display_name,
                "description": product.description,
                "score": float(result.score),  # Convert to native Python float
                "image_url": result.image
            }
            top_matches.append(match)

        # Imprime en consola cada uno de los resultados formateados:
        print("Top product matches:")
        for match in top_matches:
            print(f"Name: {match['name']}, "
                  f"Display Name: {match['display_name']}, "
                  f"Description: {match['description']}, "
                  f"Score: {match['score']}, "
                  f"Image URL: {match['image_url']}")

        logger.info(f"Found {len(top_matches)} product matches")
        return top_matches

    except vision.ImageAnnotatorError as e:
        logger.error(f"Vision API error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in product search: {str(e)}")
        raise

@retry.Retry(predicate=retry.if_transient_error)
def related_search(image_url: str, max_results: int = 5) -> List[Dict]:
    """
    Obtiene resultados relacionados a partir de una imagen usando la detección web (web detection)
    de Google Cloud Vision, sin requerir un product set ni etiquetas específicas.

    Args:
        image_url (str): URL pública de la imagen a analizar.
        max_results (int): Número máximo de resultados a retornar.

    Returns:
        List[Dict]: Lista de entidades web relacionadas, donde cada diccionario contiene:
            - entity_id: ID de la entidad (puede ser None)
            - description: Descripción o etiqueta de la entidad
            - score: Nivel de confianza

    Raises:
        ValueError: Si la URL es inválida.
        Exception: Si ocurre un error al llamar a la API.
    """
    # Validar la URL de la imagen.
    if not image_url or not image_url.startswith(('http://', 'https://')):
        raise ValueError("Invalid image URL provided")

    print(f"\nBuscando productos relacionados (WEB_DETECTION) para la imagen: {image_url}\n")

    try:
        # Inicializar el cliente de Vision y construir la petición de WEB_DETECTION
        client = get_vision_client()
        image = vision.Image(source=vision.ImageSource(image_uri=image_url))
        request = {
            "image": image,
            "features": [{"type_": vision.Feature.Type.WEB_DETECTION, "max_results": max_results}]
        }

        response = client.annotate_image(request)
        web_detection = response.web_detection

        top_products = []

        # Usar pages_with_matching_images si están disponibles
        if web_detection.pages_with_matching_images:
            for page in web_detection.pages_with_matching_images[:max_results]:
                title = page.page_title if page.page_title else "Sin título"
                link = page.url if page.url else "Link no disponible"
                if page.full_matching_images:
                    thumbnail = page.full_matching_images[0].url
                else:
                    thumbnail = "Thumbnail no disponible"
                price = "Precio no disponible"

                top_products.append({
                    "title": title,
                    "link": link,
                    "thumbnail": thumbnail,
                    "price": price
                })
        else:
            # Fallback: usar full_matching_images si no hay pages
            if web_detection.full_matching_images:
                for img in web_detection.full_matching_images[:max_results]:
                    top_products.append({
                        "title": "Sin título",
                        "link": img.url if img.url else "Link no disponible",
                        "thumbnail": img.url if img.url else "Thumbnail no disponible",
                        "price": "Precio no disponible"
                    })
            else:
                print("No se encontraron coincidencias en WEB_DETECTION.")

        # Imprimir en consola los resultados formateados.
        print("Resultados de productos relacionados:")
        for product in top_products:
            print(f"Title: {product['title']}")
            print(f"Link: {product['link']}")
            print(f"Thumbnail: {product['thumbnail']}")
            print(f"Price: {product['price']}")
            print("-------------")
    
        return top_products

    except Exception as e:
        logger.error(f"Error in related_search: {str(e)}")
        raise