import os
import pytest
import requests
from PIL import Image
from openai import AuthenticationError
from pydantic import BaseModel

from polymage.model.model import Model
from polymage.media.image_media import ImageMedia
from polymage.platform.lmstudio import LMStudioPlatform


# Define a schema for structured output testing
class WeatherResponse(BaseModel):
    city: str
    temperature: int
    condition: str

@pytest.fixture(scope="module")
def platform():
    """Provides a real LMStudioPlatform instance pointing to localhost."""
    return LMStudioPlatform(host="localhost:1234")

@pytest.fixture(scope="module", autouse=True)
def check_server_health():
    """Skips tests if LM Studio is not running locally."""
    try:
        response = requests.get("http://localhost:1234/v1/models")
        if response.status_code != 200:
            pytest.skip("LM Studio server is not responding at localhost:1234")
    except requests.exceptions.ConnectionError:
        pytest.skip("LM Studio server is not running.")

# --- Integration Tests ---

def test_integration_text_generation(platform):
    """Tests basic text completion with a real local model."""
    prompt = "Reply with exactly the word 'SUCCESS' and nothing else."
    # Using the gemma model defined in your lmstudio.py
    result = platform.text2text(model="qwen3-vl-30b", prompt=prompt)
    
    assert isinstance(result, str)
    assert "SUCCESS" in result.upper()

def test_integration_structured_data(platform):
    """Tests the JSON schema extraction and pydantic parsing."""
    prompt = "The weather in Tokyo is 25 degrees and sunny."
    
    result = platform.text2text(
        model="gemma-3-27b",
        prompt=prompt,
        response_model=WeatherResponse
    )
    
    assert isinstance(result, dict)
    assert result["city"].lower() == "tokyo"
    assert result["temperature"] == 25
    assert "sunny" in result["condition"].lower()

def test_integration_image_to_text(platform):
    """Tests multi-modal capabilities (requires a VL model like Qwen3-VL)."""
    # Create a tiny 1x1 black pixel image for testing
    from PIL import Image
    import io
    
    img = Image.new('RGB', (10, 10), color='black')
    media = ImageMedia(img)
    
    prompt = "Describe the color of this image."
    result = platform.image2text(
        model="qwen3-vl-30b",
        prompt=prompt,
        media=[media]
    )
    
    assert isinstance(result, str)
    assert len(result) > 0
