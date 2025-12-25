import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel
from typing import List

from polymage.agent.instruct_agent import InstructAgent
from polymage.platform.platform import Platform
from polymage.media.media import Media


class MockResponseModel(BaseModel):
    answer: str

@pytest.fixture
def mock_platform():
    """Provides a mocked Platform instance."""
    return MagicMock(spec=Platform)

@pytest.fixture
def default_agent(mock_platform):
    """Provides an InstructAgent with basic configuration."""
    return InstructAgent(
        platform=mock_platform,
        model="test-model",
        system_prompt="You are a helpful assistant"
    )

class TestInstructAgent:

    def test_initialization(self, mock_platform):
        """Tests if the agent initializes with the correct attributes."""
        system_prompt = "Test System Prompt"
        model = "gpt-4"
        
        agent = InstructAgent(
            platform=mock_platform,
            model=model,
            system_prompt=system_prompt
        )
        
        assert agent.platform == mock_platform
        assert agent.model == model
        assert agent.system_prompt == system_prompt
        assert agent.response_model is None

    def test_run_basic_call(self, default_agent, mock_platform):
        """Tests a standard run call to ensure parameters are passed correctly."""
        prompt = "Hello, world!"
        mock_platform.text2text.return_value = "Mocked Response"

        result = default_agent.run(prompt=prompt)

        # Verify the platform's text2text was called with expected arguments
        mock_platform.text2text.assert_called_once_with(
            model="test-model",
            prompt=prompt,
            media=None,
            response_model=None,
            system_prompt="You are a helpful assistant"
        )
        assert result == "Mocked Response"

    def test_run_with_media(self, default_agent, mock_platform):
        """Tests that media list is correctly passed to the platform."""
        prompt = "Describe this image"
        mock_media = [MagicMock(spec=Media), MagicMock(spec=Media)]
        
        default_agent.run(prompt=prompt, media=mock_media)

        mock_platform.text2text.assert_called_once_with(
            model="test-model",
            prompt=prompt,
            media=mock_media,
            response_model=None,
            system_prompt="You are a helpful assistant"
        )

    def test_run_with_response_model(self, mock_platform):
        """Tests that response_model is passed when provided in init."""
        agent = InstructAgent(
            platform=mock_platform,
            model="test-model",
            response_model=MockResponseModel
        )
        
        agent.run(prompt="structured request")

        mock_platform.text2text.assert_called_once_with(
            model="test-model",
            prompt="structured request",
            media=None,
            response_model=MockResponseModel
        )

    def test_system_prompt_precedence(self, default_agent, mock_platform):
        """Tests that the agent's internal system_prompt overrides one passed in kwargs."""
        prompt = "test"
        # Passing a conflicting system_prompt via kwargs
        default_agent.run(prompt=prompt, system_prompt="I am a rogue AI")

        # It should still be the one from __init__ ("You are a helpful assistant")
        args, kwargs = mock_platform.text2text.call_args
        assert kwargs['system_prompt'] == "You are a helpful assistant"

    def test_no_system_prompt(self, mock_platform):
        """Tests behavior when no system prompt is provided."""
        agent = InstructAgent(platform=mock_platform, model="test-model")
        
        agent.run(prompt="Just a prompt")

        # kwargs should not contain system_prompt if it was None
        args, kwargs = mock_platform.text2text.call_args
        assert 'system_prompt' not in kwargs

    def test_additional_kwargs_passthrough(self, default_agent, mock_platform):
        """Tests that extra kwargs are passed to the platform."""
        default_agent.run(prompt="test", temperature=0.7, top_p=1.0)

        mock_platform.text2text.assert_called_once_with(
            model="test-model",
            prompt="test",
            media=None,
            response_model=None,
            system_prompt="You are a helpful assistant",
            temperature=0.7,
            top_p=1.0
        )