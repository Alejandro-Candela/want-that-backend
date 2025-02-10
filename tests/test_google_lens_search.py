import pytest
from src.modules.search.google_lens_search import product_search

# --- Clases Dummy para simular respuestas del API ---

class DummyProduct:
    def __init__(self, name: str, display_name: str, description: str):
        self.name = name
        self.display_name = display_name
        self.description = description

class DummyResult:
    def __init__(self, product: DummyProduct, score: float, image: str):
        self.product = product
        self.score = score
        self.image = image

class DummyProductSearchResults:
    def __init__(self, results: list):
        self.results = results

class DummyResponse:
    def __init__(self, product_search_results: DummyProductSearchResults):
        self.product_search_results = product_search_results

class DummyVisionClient:
    def product_search(self, image, image_context):
        # Se imprime para debugear el llamado
        print("Llamado a DummyVisionClient.product_search")
        # Se crea un producto dummy
        dummy_product = DummyProduct(
            name="projects/item-finder-alejandro/locations/us-west1/products/dummy-product",
            display_name="Dummy Product",
            description="Un producto dummy para pruebas"
        )
        dummy_result = DummyResult(dummy_product, 0.95, "https://dummy.com/product.jpg")
        dummy_results = DummyProductSearchResults([dummy_result])
        return DummyResponse(dummy_results)

class DummyProductSearchClient:
    def product_set_path(self, *, project: str, location: str, product_set: str) -> str:
        path = f"projects/{project}/locations/{location}/productSets/{product_set}"
        print("Llamado a DummyProductSearchClient.product_set_path; devuelve:", path)
        return path

# --- Fixture para sobrescribir los clientes reales con los dummy ---
@pytest.fixture(autouse=True)
def override_clients(monkeypatch: pytest.MonkeyPatch):
    # Sobrescribimos las funciones get_vision_client y get_product_search_client
    import src.modules.search.google_lens_search as gls
    monkeypatch.setattr(gls, "get_vision_client", lambda: DummyVisionClient())
    monkeypatch.setattr(gls, "get_product_search_client", lambda: DummyProductSearchClient())

# --- Test para el flujo exitoso ---
def test_product_search_valid():
    valid_image_url = "https://dummy.com/image.jpg"
    # Llamamos a product_search con parámetros dummy
    result = product_search(
        valid_image_url,
        product_set_id="dummy_set",
        product_category="dummy-category",
        max_results=1
    )
    # Verificamos que el resultado es una lista con un diccionario que contiene las claves esperadas
    assert isinstance(result, list)
    assert len(result) == 1
    match = result[0]
    for key in ("name", "display_name", "description", "score", "image_url"):
        assert key in match
    assert isinstance(match["score"], float)

# --- Test para validar URL inválida ---
def test_product_search_invalid_url():
    invalid_image_url = "not_a_valid_url"
    with pytest.raises(ValueError) as exc_info:
        product_search(invalid_image_url)
    assert "Invalid image URL provided" in str(exc_info.value)

# --- Test para simular una excepción durante la llamada al API ---
def test_product_search_exception(monkeypatch: pytest.MonkeyPatch):
    # Creamos un cliente faulty para simular un error en product_search
    class FaultyVisionClient:
        def product_search(self, image, image_context):
            raise Exception("Simulated API failure")
    import src.modules.search.google_lens_search as gls
    monkeypatch.setattr(gls, "get_vision_client", lambda: FaultyVisionClient())
    
    with pytest.raises(Exception) as exc_info:
        product_search("https://dummy.com/image.jpg")
    assert "Simulated API failure" in str(exc_info.value) 