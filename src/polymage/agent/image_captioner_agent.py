from typing import Any, Optional, List

from .agent import Agent
from ..media.media import Media

"""
an image captioner agent
"""

class ImageCaptionerAgent(Agent):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)


	def run(self, prompt: str, media: Optional[List[Media]] = None, **kwargs) -> Any:
		platform=self.platform
		model=self.model
		system_prompt=self.system_prompt

		return platform.image2text(model=model, prompt=prompt, media=media, **kwargs)
