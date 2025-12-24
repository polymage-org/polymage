import pytest
import io
import base64
from PIL import Image

from unittest.mock import MagicMock, patch
from polymage.media.image_media import ImageMedia
from polymage.utils.image_utils  import image_to_base64


@pytest.fixture
def sample_pil_image():
    """Provides a basic 10x10 RGB image."""
    return Image.new("RGB", (10, 10), color="red")

@pytest.fixture
def sample_base64_image(sample_pil_image):
    """Provides a base64 string of a PNG image."""
    buffer = io.BytesIO()
    sample_pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

@pytest.fixture
def sample_bytes_image(sample_pil_image):
    """Provides raw bytes of a PNG image."""
    buffer = io.BytesIO()
    sample_pil_image.save(buffer, format="PNG")
    return buffer.getvalue()

class TestImageMedia:
    
    # --- Initialization Tests ---

    def test_init_with_pil_image(self, sample_pil_image):
        """Tests initialization using a direct PIL Image object."""
        media = ImageMedia(sample_pil_image)
        assert media._image == sample_pil_image
        assert isinstance(media._image, Image.Image)

    def test_init_with_base64(self, sample_base64_image):
        """Tests initialization using a base64 string."""
        media = ImageMedia(sample_base64_image)
        assert isinstance(media._image, Image.Image)
        assert media._image.size == (10, 10)

    def test_init_with_bytes(self, sample_bytes_image):
        """Tests initialization using raw bytes."""
        media = ImageMedia(sample_bytes_image)
        assert isinstance(media._image, Image.Image)
        assert media._image.size == (10, 10)

    def test_init_invalid_type(self):
        """Tests that passing an unsupported type raises a TypeError."""
        with pytest.raises(TypeError, match="image_data must be"):
            ImageMedia(12345)

    def test_init_with_metadata(self, sample_pil_image):
        """Tests that metadata is correctly stored during init."""
        meta = {"author": "Gemini", "version": "1.0"}
        media = ImageMedia(sample_pil_image, metadata=meta)
        assert media._metadata == meta

    # --- Conversion Tests ---

    def test_to_base64(self, sample_pil_image):
        """Tests conversion back to base64 string."""
        media = ImageMedia(sample_pil_image)
        b64_output = media.to_base64(format="PNG")
        
        assert isinstance(b64_output, str)
        # Verify it's valid base64 by decoding it back
        decoded_bytes = base64.b64decode(b64_output)
        img = Image.open(io.BytesIO(decoded_bytes))
        assert img.size == (10, 10)

    # --- File I/O Tests ---

    def test_save_to_file(self, sample_pil_image, tmp_path):
        """Tests saving the image to a file path."""
        d = tmp_path / "output"
        d.mkdir()
        file_path = str(d / "test_image.png")
        
        media = ImageMedia(sample_pil_image, metadata={"title": "Test"})
        media.save_to_file(file_path)
        
        # Verify file exists
        assert (d / "test_image.png").exists()
        
        # Verify metadata was saved (PNG specific)
        saved_img = Image.open(file_path)
        assert saved_img.info["title"] == "Test"

    @patch("PIL.Image.Image.save")
    def test_save_to_file_calls_pil_save(self, mock_save, sample_pil_image):
        """Mocks the PIL save method to ensure internal calls are correct."""
        media = ImageMedia(sample_pil_image, metadata={"key": "value"})
        media.save_to_file("mock_path.png")
        
        # Check if save was called
        assert mock_save.called
        # Verify that pnginfo was passed to the save call
        args, kwargs = mock_save.call_args
        assert "pnginfo" in kwargs
        
