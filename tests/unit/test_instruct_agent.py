import pytest
from unittest.mock import Mock, MagicMock
from typing import Any, Optional, List

# Assuming the classes are in these modules
from polymage.agent.agent import Agent
from polymage.media.media import Media
from polymage.agent.instruct_agent import InstructAgent


class TestInstructAgent:
	def test_init(self):
		"""Test InstructAgent initialization"""
		mock_platform = Mock()
		agent = InstructAgent(
			platform=mock_platform,
			model="gpt-4",
			system_prompt="You are a helpful assistant"
		)

		assert agent.platform == mock_platform
		assert agent.model == "gpt-4"
		assert agent.system_prompt == "You are a helpful assistant"

	def test_run_with_system_prompt(self):
		"""Test run method with system prompt"""
		mock_platform = Mock()
		mock_platform.text2text.return_value = "Test response"

		agent = InstructAgent(
			platform=mock_platform,
			model="gpt-4",
			system_prompt="You are a helpful assistant"
		)

		result = agent.run("What is the capital of France?")

		# Verify text2text was called with correct parameters
		mock_platform.text2text.assert_called_once_with(
			model="gpt-4",
			prompt="What is the capital of France?",
			media=None,
			response_model=None,
			system_prompt="You are a helpful assistant"
		)

		assert result == "Test response"

	def test_run_without_system_prompt(self):
		"""Test run method without system prompt"""
		mock_platform = Mock()
		mock_platform.text2text.return_value = "Test response"

		agent = InstructAgent(
			platform=mock_platform,
			model="gpt-4"
		)

		result = agent.run("What is the capital of France?")

		# Verify text2text was called without system_prompt
		mock_platform.text2text.assert_called_once_with(
			model="gpt-4",
			prompt="What is the capital of France?",
			media=None,
			response_model=None
		)

		assert result == "Test response"

	def test_run_with_media(self):
		"""Test run method with media objects"""
		mock_platform = Mock()
		mock_platform.text2text.return_value = "Test response"

		media_obj = Mock(spec=Media)
		agent = InstructAgent(
			platform=mock_platform,
			model="gpt-4",
			system_prompt="You are a helpful assistant"
		)

		result = agent.run("What is the capital of France?", media=[media_obj])

		# Verify text2text was called with media parameter
		mock_platform.text2text.assert_called_once_with(
			model="gpt-4",
			prompt="What is the capital of France?",
			media=[media_obj],
			response_model=None,
			system_prompt="You are a helpful assistant"
		)

		assert result == "Test response"

	def test_run_with_response_model(self):
		"""Test run method with response model"""
		mock_platform = Mock()
		mock_response_model = Mock()
		mock_platform.text2text.return_value = "Test response"

		agent = InstructAgent(
			platform=mock_platform,
			model="gpt-4",
			system_prompt="You are a helpful assistant"
		)

		result = agent.run(
			"What is the capital of France?",
			response_model=mock_response_model
		)

		# Verify text2text was called with response_model
		mock_platform.text2text.assert_called_once_with(
			model="gpt-4",
			prompt="What is the capital of France?",
			media=None,
			response_model=mock_response_model,
			system_prompt="You are a helpful assistant"
		)

		assert result == "Test response"

	def test_run_with_additional_kwargs(self):
		"""Test run method with additional keyword arguments"""
		mock_platform = Mock()
		mock_platform.text2text.return_value = "Test response"

		agent = InstructAgent(
			platform=mock_platform,
			model="gpt-4",
			system_prompt="You are a helpful assistant"
		)

		result = agent.run(
			"What is the capital of France?",
			temperature=0.7,
			max_tokens=100
		)

		# Verify text2text was called with additional kwargs
		mock_platform.text2text.assert_called_once_with(
			model="gpt-4",
			prompt="What is the capital of France?",
			media=None,
			response_model=None,
			system_prompt="You are a helpful assistant",
			temperature=0.7,
			max_tokens=100
		)

		assert result == "Test response"

	def test_run_overrides_system_prompt(self):
		"""Test that system prompt in kwargs overrides the agent's system prompt"""
		mock_platform = Mock()
		mock_platform.text2text.return_value = "Test response"

		agent = InstructAgent(
			platform=mock_platform,
			model="gpt-4",
			system_prompt="Default system prompt"
		)

		# Pass a different system prompt in kwargs
		result = agent.run(
			"What is the capital of France?",
			system_prompt="Overridden system prompt"
		)

		# Verify that the kwargs version takes precedence
		mock_platform.text2text.assert_called_once_with(
			model="gpt-4",
			prompt="What is the capital of France?",
			media=None,
			response_model=None,
			system_prompt="Overridden system prompt"
		)

		assert result == "Test response"

	def test_run_with_none_system_prompt(self):
		"""Test run method when system_prompt is explicitly None"""
		mock_platform = Mock()
		mock_platform.text2text.return_value = "Test response"

		agent = InstructAgent(
			platform=mock_platform,
			model="gpt-4",
			system_prompt=None
		)

		result = agent.run("What is the capital of France?")

		# Verify text2text was called without system_prompt
		mock_platform.text2text.assert_called_once_with(
			model="gpt-4",
			prompt="What is the capital of France?",
			media=None,
			response_model=None
		)

		assert result == "Test response"

