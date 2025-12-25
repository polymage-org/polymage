import pytest
from unittest.mock import MagicMock
from pydantic import BaseModel
from polymage.agent.image_generator_agent import ImageGeneratorAgent

class MockResponseModel(BaseModel):
    url: str

@pytest.fixture
def mock_platform():
    """Fixture to provide a mocked platform instance."""
    return MagicMock()

@pytest.fixture
def agent(mock_platform):
    """Fixture to provide an ImageGeneratorAgent instance."""
    return ImageGeneratorAgent(
        platform=mock_platform,
        model="dall-e-3",
        system_prompt="You are a creative artist."
    )

class TestImageGeneratorAgent:

    def test_initialization(self, mock_platform):
        """Test if the agent initializes with correct attributes."""
        agent = ImageGeneratorAgent(
            platform=mock_platform,
            model="test-model",
            system_prompt="Test Prompt"
        )
        assert agent.platform == mock_platform
        assert agent.model == "test-model"
        assert agent.system_prompt == "Test Prompt"

    def test_run_text_to_image(self, agent, mock_platform):
        """Test that run calls text2image when media is None."""
        prompt = "A sunset over the mountains"
        mock_platform.text2image.return_value = "image_url_123"

        result = agent.run(prompt=prompt, media=None)

        # Verify platform call
        mock_platform.text2image.assert_called_once_with(
            model=agent.model,
            prompt=prompt
        )
        assert result == "image_url_123"

    def test_run_image_to_image(self, agent, mock_platform):
        """Test that run calls image2image when media is provided."""
        prompt = "Make this image look like a Van Gogh painting"
        mock_media = [MagicMock()]
        mock_platform.image2image.return_value = "edited_image_url"

        result = agent.run(prompt=prompt, media=mock_media)

        # Verify platform call
        mock_platform.image2image.assert_called_once_with(
            model=agent.model,
            prompt=prompt,
            media=mock_media
        )
        assert result == "edited_image_url"

    def test_run_with_kwargs(self, agent, mock_platform):
        """Test that extra kwargs are passed down to the platform."""
        prompt = "Abstract art"
        agent.run(prompt=prompt, media=None, quality="high", style="vivid")

        mock_platform.text2image.assert_called_once_with(
            model=agent.model,
            prompt=prompt,
            quality="high",
            style="vivid"
        )

    def test_run_with_response_model(self, agent, mock_platform):
        """Test that passing a response_model doesn't break the call (it's ignored by current logic)."""
        prompt = "A cat"
        agent.run(prompt=prompt, response_model=MockResponseModel)

        # Verify it still routes to text2image correctly
        mock_platform.text2image.assert_called_once()
