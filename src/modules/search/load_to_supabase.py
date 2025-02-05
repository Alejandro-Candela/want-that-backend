import os
import time
import requests
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

import os
import time
import base64
import requests
from datetime import datetime
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

def load_to_supabase(image_path: str, bucket_name: str, score: float, text_prompt: str) -> str:
    """
    Sube una imagen en formato .webp a un bucket de Supabase Storage junto con el score y el text_prompt 
    como metadata, y a continuación la sube a Imgur. Retorna la URL pública de la imagen en Imgur.
    La imagen se sube con un nombre generado en el formato: fecha_prompt_score.webp

    Parámetros:
        image_path (str): Ruta local de la imagen .webp a subir.
        bucket_name (str): Nombre del bucket en Supabase.
        score (float): Score asociado a la imagen.
        text_prompt (str): Texto del prompt utilizado.
    
    Retorna:
        str: URL pública de la imagen en Imgur si ambas subidas son exitosas; una cadena vacía en caso de error.
    """
    # Instanciar el cliente de Supabase.
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Generar un nombre de archivo basado en la fecha actual, el prompt y el score.
    date_str = datetime.now().strftime("%Y%m%d%H%M%S")
    sanitized_prompt = text_prompt.replace(" ", "_")
    formatted_score = f"{score:.2f}"
    object_name = f"{date_str}_{sanitized_prompt}_{formatted_score}.webp"

    try:
        # Leer la imagen en modo binario.
        with open(image_path, "rb") as file:
            file_data = file.read()

        # Opciones de subida: se define el contentType y se incluye la metadata.
        options = {
            "contentType": "image/webp",
            "metadata": {
                "score": str(score),
                "text_prompt": text_prompt
            }
        }

        # Subir la imagen al bucket de Supabase.
        upload_response = supabase.storage.from_(bucket_name).upload(object_name, file_data, options)
        if not upload_response:
            print("Error al subir la imagen a Supabase.")
            return ""

        # Subir la imagen a Imgur para obtener una URL pública.
        encoded_image = base64.b64encode(file_data).decode("utf-8")
        url_imgur = "https://api.imgur.com/3/upload"
        headers_imgur = {
            "Authorization": f"Client-ID {os.getenv('IMGUR_CLIENT_ID')}"
        }
        data_imgur = {
            "image": encoded_image,
            "type": "base64"
        }
        response_imgur = requests.post(url_imgur, headers=headers_imgur, data=data_imgur, timeout=10)
        if response_imgur.status_code != 200:
            print(f"Error al subir la imagen a Imgur: {response_imgur.text}")
            return ""
        
        imgur_json = response_imgur.json()
        if not imgur_json.get("success"):
            print(f"Error en la respuesta de Imgur: {imgur_json}")
            return ""
        
        imgur_url = imgur_json["data"]["link"]
        return imgur_url

    except FileNotFoundError:
        print(f"El archivo {image_path} no fue encontrado.")
    except Exception as e:
        print(f"Ocurrió un error al subir la imagen: {e}")

    return ""

