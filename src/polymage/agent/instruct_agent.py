from typing import Any, Optional, List
from pydantic import BaseModel

from .agent import Agent
from ..media.media import Media

"""
a basic instruct agent
"""

class InstructAgent(Agent):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)


	def run(self, prompt: str, media: Optional[List[Media]] = None, **kwargs) -> Any:
		platform=self.platform
		model=self.model
		response_model=self.response_model
		system_prompt=self.system_prompt

		if system_prompt is None:
			return platform.text2text(model=model, prompt=prompt, media=media, response_model=response_model, **kwargs)
		else:
			# add the system_prompt to **kwargs
			extras = {'system_prompt': system_prompt}
			combined = {**kwargs, **extras}  # Merge: extras override duplicates
			return platform.text2text(model=model, prompt=prompt, media=media, response_model=response_model, **combined)

