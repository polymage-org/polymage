import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel
from typing import List


from polymage.agent.image_captioner_agent import ImageCaptionerAgent
from polymage.platform.platform import Platform
from polymage.media.media import Media


class TestImageCaptionerAgent:
    @pytest.fixture
    def mock_platform(self):
        """Provides a mocked Platform instance."""
        return MagicMock(spec=Platform)

    @pytest.fixture
    def mock_media(self):
        """Provides a list containing a mocked Media instance."""
        return [MagicMock(spec=Media)]

    @pytest.fixture
    def agent(self, mock_platform):
        """Initializes the ImageCaptionerAgent with a mock platform."""
        return ImageCaptionerAgent(
            platform=mock_platform,
            model="test-vision-model",
            system_prompt="You are a helpful captioner."
        )

    def test_initialization(self, agent, mock_platform):
        """Tests if the agent correctly inherits and sets attributes."""
        assert agent.platform == mock_platform
        assert agent.model == "test-vision-model"
        assert agent.system_prompt == "You are a helpful captioner."

    def test_run_calls_platform_image2text(self, agent, mock_platform, mock_media):
        """Tests if run() correctly delegates to platform.image2text with right args."""
        # Setup
        expected_response = "A beautiful sunset over the mountains."
        mock_platform.image2text.return_value = expected_response
        
        test_prompt = "Describe this image."
        test_kwargs = {"temperature": 0.5, "max_tokens": 100}

        # Execute
        result = agent.run(
            prompt=test_prompt, 
            media=mock_media, 
            **test_kwargs
        )

        # Assert
        mock_platform.image2text.assert_called_once_with(
            model="test-vision-model",
            prompt=test_prompt,
            media=mock_media,
            **test_kwargs
        )
        assert result == expected_response

    def test_run_with_no_media(self, agent, mock_platform):
        """Tests the agent's behavior when media is None."""
        agent.run(prompt="Hello")
        
        mock_platform.image2text.assert_called_once_with(
            model="test-vision-model",
            prompt="Hello",
            media=None
        )
