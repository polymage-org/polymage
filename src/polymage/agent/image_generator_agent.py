from typing import Any, Optional, List
from pydantic import BaseModel

from .agent import Agent
from ..media.media import Media

"""
a basic instruct agent
"""

class ImageGeneratorAgent(Agent):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)


	def run(self, prompt: str, media: Optional[List[Media]] = None, response_model: Optional[BaseModel] = None, **kwargs) -> Any:
		platform=self.platform
		model=self.model
		system_prompt=self.system_prompt

		if media is None:
			return platform.text2image(model=model, prompt=prompt, **kwargs)
		else:
			return platform.image2image(model=model, prompt=prompt, **kwargs)
