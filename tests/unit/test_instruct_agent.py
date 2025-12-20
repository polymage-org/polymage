# test_instruct_agent.py

import pytest
from unittest.mock import Mock, patch
from typing import Optional, List
from pydantic import BaseModel

from polymage.media.media import Media
from polymage.agent.instruct_agent import InstructAgent


# Define a mock response model for testing
class TestResponseModel(BaseModel):
    answer: str


# Define a mock platform that implements text2text
class MockPlatform:
    def text2text(self, model, prompt, media=None, response_model=None, **kwargs):
        # Just echo inputs for testing
        return {
            "model": model,
            "prompt": prompt,
            "media": media,
            "response_model": response_model.__name__ if response_model else None,
            "system_prompt": kwargs.get("system_prompt"),
        }


@pytest.fixture
def mock_platform():
    return MockPlatform()


def test_instruct_agent_initialization(mock_platform):
    """Test that InstructAgent initializes correctly."""
    agent = InstructAgent(
        platform=mock_platform,
        model="gpt-4",
        system_prompt="You are a helpful assistant",
        response_model=TestResponseModel
    )
    assert agent.platform == mock_platform
    assert agent.model == "gpt-4"
    assert agent.system_prompt == "You are a helpful assistant"
    assert agent.response_model == TestResponseModel


def test_instruct_agent_run_with_system_prompt(mock_platform):
    """Test that system_prompt is passed correctly to text2text."""
    agent = InstructAgent(
        platform=mock_platform,
        model="gpt-4",
        system_prompt="You are a helpful assistant"
    )

    result = agent.run("What is the capital of France?")

    assert result["model"] == "gpt-4"
    assert result["prompt"] == "What is the capital of France?"
    assert result["system_prompt"] == "You are a helpful assistant"
    assert result["response_model"] is None


def test_instruct_agent_run_without_system_prompt(mock_platform):
    """Test behavior when no system_prompt is provided."""
    agent = InstructAgent(platform=mock_platform, model="gpt-4")

    result = agent.run("What is the capital of France?")

    assert result["system_prompt"] is None


def test_instruct_agent_run_with_response_model(mock_platform):
    """Test that response_model is passed correctly."""
    agent = InstructAgent(
        platform=mock_platform,
        model="gpt-4",
        response_model=TestResponseModel
    )

    result = agent.run("What is the capital of France?")

    assert result["response_model"] == "TestResponseModel"


def test_instruct_agent_run_with_media(mock_platform):
    """Test that media is passed correctly."""
    mock_media = [Mock(spec=Media), Mock(spec=Media)]
    agent = InstructAgent(platform=mock_platform, model="gpt-4")

    result = agent.run("Describe this image.", media=mock_media)

    assert result["media"] == mock_media


def test_instruct_agent_run_with_extra_kwargs(mock_platform):
    """Test that extra kwargs are passed through to text2text."""
    agent = InstructAgent(
        platform=mock_platform,
        model="gpt-4",
        system_prompt="You are a helpful assistant"
    )

    result = agent.run("What is 2+2?", temperature=0.7, max_tokens=100)

    # system_prompt should override if passed in kwargs, but in current impl it always sets from self
    assert result["system_prompt"] == "You are a helpful assistant"
    # Extra kwargs like temperature are not captured in our mock, so we can't assert them
    # unless we modify MockPlatform to capture **kwargs. Let's enhance it for this test.

def test_instruct_agent_run_with_extra_kwargs_captured(mock_platform):
    """Test that extra kwargs are passed through by enhancing the mock."""
    # Enhance MockPlatform to capture all kwargs
    def enhanced_text2text(model, prompt, media=None, response_model=None, **kwargs):
        return {
            "model": model,
            "prompt": prompt,
            "media": media,
            "response_model": response_model.__name__ if response_model else None,
            "kwargs": kwargs,
        }

    mock_platform.text2text = enhanced_text2text
    agent = InstructAgent(
        platform=mock_platform,
        model="gpt-4",
        system_prompt="You are a helpful assistant"
    )

    result = agent.run("What is 2+2?", temperature=0.7, max_tokens=100)

    assert result["kwargs"]["system_prompt"] == "You are a helpful assistant"
    assert result["kwargs"]["temperature"] == 0.7
    assert result["kwargs"]["max_tokens"] == 100
    
