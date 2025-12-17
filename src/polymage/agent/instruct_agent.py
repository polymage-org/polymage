from typing import Any, Optional, List
from pydantic import BaseModel

from .agent import Agent
from ..media.media import Media


class InstructAgent(Agent):
	"""
	An agent that handles text-to-text transformations with optional system prompts.

	This class extends the base Agent class to provide functionality for running
	text processing tasks where the platform's text2text method is used. It supports
	optional system prompts that can be injected into the prompt processing workflow.

	The agent is designed to work with various platforms that implement text2text
	functionality, allowing for flexible prompt handling and response modeling.

	Attributes:
		Inherits all attributes from Agent class including platform, model,
		response_model, and system_prompt.

	Example:
		# Create an InstructAgent instance
		agent = InstructAgent(
			platform=my_platform,
			model="gpt-4",
			system_prompt="You are a helpful assistant"
		)

		# Run with system prompt
		result = agent.run("What is the capital of France?")

		# Run without system prompt
		result = agent.run("What is the capital of France?",
						  response_model=MyResponseModel)
	"""

	def __init__(self, **kwargs):
		"""
		Initialize the InstructAgent with platform, model and optional parameters.

		Args:
			**kwargs: Keyword arguments passed to the parent Agent constructor
					 including platform, model, response_model, and system_prompt.
		"""
		super().__init__(**kwargs)

	def run(self, prompt: str, media: Optional[List[Media]] = None, **kwargs) -> Any:
		"""
		Execute a text-to-text transformation using the configured platform.

		This method processes a prompt through the platform's text2text capability,
		optionally incorporating a system prompt if one was provided during initialization.

		Args:
			prompt (str): The input text prompt to process
			media (Optional[List[Media]]): Optional list of media objects to include
										  in the processing (e.g., images, files)
			**kwargs: Additional keyword arguments to pass to the platform's text2text method

		Returns:
			Any: The result from the platform's text2text processing, typically
				 a string or structured data based on response_model parameter

		Note:
			If system_prompt was provided during initialization, it will be merged
			with any additional kwargs, where the system prompt takes precedence
			in case of key conflicts.
		"""

		if self.system_prompt is not None:
			kwargs['system_prompt'] = self.system_prompt

		return self.platform.text2text(
			model=self.model,
			prompt=prompt,
			media=media,
			response_model=self.response_model,
			**kwargs
		)
