import os
import pytest
from PIL import Image
from openai import AuthenticationError
from pydantic import BaseModel

from polymage.model.model import Model
from polymage.media.image_media import ImageMedia
from polymage.platform.togetherai import TogetherAiPlatform  # adjust import as needed


class DummyResponseModel(BaseModel):
    answer: str


@pytest.fixture(scope="session")
def api_key():
    key = os.getenv("TOGETHERAI_TOKEN")
    if not key:
        pytest.skip("TOGETHERAI_TOKEN environment variable not set")
    return key


@pytest.fixture
def platform(api_key):
    return TogetherAiPlatform(api_key=api_key)


@pytest.mark.integration
def test_text2text(platform):
    result = platform.text2text("gpt-oss-20b", "What is 2 + 2?")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.integration
def test_text2data(platform):
    result = platform.text2text(
        "gpt-oss-20b",
        "Return a JSON with key 'answer' and value '42'.",
        response_model=DummyResponseModel
    )
    assert isinstance(result, dict)
    assert result["answer"] == "42"
    
    

@pytest.mark.integration
def test_image2text(platform, tmp_path):
    # Create a tiny test image
    image_path = tmp_path / "test.png"
    img = Image.new("RGB", (64, 64), color="red")

    image_media = ImageMedia(img)
    prompt = "What color is this image?"

    # Only qwen3-vl supports vision
    result = platform.image2text("Qwen3-VL-8B-Instruct", prompt, [image_media])
    assert isinstance(result, str)
    assert len(result) > 0
    assert "red" in result.lower()


@pytest.mark.integration
def test_invalid_api_key():
    bad_platform = TogetherAiPlatform(api_key="invalid-key")
    with pytest.raises(AuthenticationError):
        bad_platform.text2text("gpt-oss-20b", "Hello")

