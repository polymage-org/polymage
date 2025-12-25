import os
import io
import base64
import pytest
import requests
from PIL import Image
from openai import AuthenticationError
from pydantic import BaseModel

from polymage.model.model import Model
from polymage.media.image_media import ImageMedia
from polymage.platform.drawthings import DrawThingsPlatform


# Configuration for the local server
SERVER_HOST = "192.168.42.16:7860"

@pytest.fixture
def platform():
    """Provides an instance of the DrawThingsPlatform."""
    return DrawThingsPlatform(host=SERVER_HOST)

@pytest.fixture
def sample_image_media():
    """Creates a small dummy ImageMedia object for img2img tests."""
    img = Image.new('RGB', (512, 512), color='red')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return ImageMedia(img_str)

class TestDrawThingsPlatform:

    def test_platform_initialization(self, platform):
        """Verify the platform loads with the correct name and models."""
        assert platform.platform_name() == 'drawthings'  # Based on your provided code
        models = platform.platform_models()
        assert len(models) > 0
        assert any(m.model_name() == "flux-1-schnell" for m in models)

    def test_get_model_by_name(self, platform):
        """Verify we can retrieve a specific model configuration."""
        model = platform.getModelByName("flux-1-schnell")
        assert model.model_name() == "flux-1-schnell"
        
    def test_text2image_integration(self, platform):
        """
        Integration test: Performs a real request to the local DrawThings server.
        Requires 'flux-1-schnell' to be available on the server.
        """
        prompt = "a simple red cube on a white background"
        try:
            result = platform.text2image(model="flux-1-schnell", prompt=prompt)
            
            assert isinstance(result, ImageMedia)
            assert result._image is not None
            # Verify metadata was attached
            assert result._metadata['Description'] == prompt
        except Exception as e:
            pytest.fail(f"text2image request failed: {e}")

    def test_image2image_integration(self, platform, sample_image_media):
        """
        Integration test: Performs a real img2img request.
        Requires 'qwen-image-edit-4-steps' to be available.
        """
        prompt = "turn the red cube blue"
        try:
            # Note: platform.py's image2image expects a List[ImageMedia]
            result = platform.image2image(
                model="qwen-image-edit-4-steps", 
                prompt=prompt, 
                media=[sample_image_media]
            )
            
            assert isinstance(result, ImageMedia)
            assert result._image.size[0] > 0
        except Exception as e:
            pytest.fail(f"image2image request failed: {e}")

    def test_invalid_model_throws_error(self, platform):
        """Verify that requesting a non-existent model raises ValueError."""
        with pytest.raises(ValueError):
            platform.text2image(model="non-existent-model", prompt="test")

    def test_unsupported_methods(self, platform):
        """Verify that unimplemented methods return None or pass as defined."""
        # The current implementation of _text2text in DrawThingsPlatform just 'pass'es
        model_obj = platform.getModelByName("flux-1-schnell")
        assert platform._text2text(model_obj, "hello") is None
