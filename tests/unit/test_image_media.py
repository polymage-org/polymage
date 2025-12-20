import base64
from io import BytesIO
import pytest
from PIL import Image, PngImagePlugin

from polymage.media.image_media import (
    base64_to_image,
    bytes_to_image,
    image_to_base64,
    ImageMedia,
)


# --------------------------
# Fixtures
# --------------------------

@pytest.fixture
def sample_image():
    """Returns a small red 10x10 PIL Image."""
    return Image.new("RGB", (10, 10), color=(255, 0, 0))

@pytest.fixture
def sample_metadata():
    return {"author": "pytest", "source": "unit_test"}

# --------------------------
# Utility Function Tests
# --------------------------

def test_base64_to_image(sample_image):
    b64_str = image_to_base64(sample_image, format="PNG")
    img = base64_to_image(b64_str)
    assert isinstance(img, Image.Image)
    assert img.size == sample_image.size

def test_bytes_to_image(sample_image):
    buf = BytesIO()
    sample_image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img = bytes_to_image(img_bytes)
    assert isinstance(img, Image.Image)
    assert img.size == sample_image.size

def test_image_to_base64(sample_image):
    b64_str = image_to_base64(sample_image, format="PNG")
    assert isinstance(b64_str, str)
    decoded = base64.b64decode(b64_str)
    assert len(decoded) > 0
    # Verify it can be reloaded
    img = Image.open(BytesIO(decoded))
    assert img.size == sample_image.size

# --------------------------
# ImageMedia Class Tests
# --------------------------

def test_imagmedia_from_pil_image(sample_image, sample_metadata):
    media = ImageMedia(sample_image, metadata=sample_metadata)
    assert isinstance(media._image, Image.Image)
    assert media._metadata == sample_metadata

def test_imagmedia_from_base64_string(sample_image, sample_metadata):
    b64_str = image_to_base64(sample_image, format="PNG")
    media = ImageMedia(b64_str, metadata=sample_metadata)
    assert isinstance(media._image, Image.Image)
    assert media._image.size == sample_image.size
    assert media._metadata == sample_metadata

def test_imagmedia_from_bytes(sample_image, sample_metadata):
    buf = BytesIO()
    sample_image.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    media = ImageMedia(img_bytes, metadata=sample_metadata)
    assert isinstance(media._image, Image.Image)
    assert media._image.size == sample_image.size
    assert media._metadata == sample_metadata

def test_imagmedia_invalid_input():
    with pytest.raises(TypeError, match="image_data must be either"):
        ImageMedia(12345)

def test_to_base64(sample_image):
    media = ImageMedia(sample_image)
    b64_str = media.to_base64(format="PNG")
    assert isinstance(b64_str, str)
    # Validate round-trip
    recovered = base64_to_image(b64_str)
    assert recovered.size == sample_image.size

def test_save_to_file_with_metadata(sample_image, sample_metadata, tmp_path):
    media = ImageMedia(sample_image, metadata=sample_metadata)
    output_path = tmp_path / "output.png"
    media.save_to_file(str(output_path))

    # Reload and check metadata
    reloaded = Image.open(output_path)
    assert hasattr(reloaded, "text")  # PNG metadata
    for k, v in sample_metadata.items():
        assert reloaded.text.get(k) == v

def test_save_to_file_no_metadata(sample_image, tmp_path):
    media = ImageMedia(sample_image)
    output_path = tmp_path / "output_no_meta.png"
    media.save_to_file(str(output_path))

    reloaded = Image.open(output_path)
    assert isinstance(reloaded, Image.Image)
    # Should not crash; metadata may be absent

# --------------------------
# Edge Cases
# --------------------------

def test_base64_invalid_string():
    with pytest.raises(Exception):
        base64_to_image("not_a_valid_base64_image_string")

def test_bytes_invalid():
    with pytest.raises(Exception):
        bytes_to_image(b"invalid_image_bytes")
