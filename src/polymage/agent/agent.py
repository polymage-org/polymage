from abc import ABC, abstractmethod
from typing import Any, Optional, List
from pydantic import BaseModel

from ..media.media import Media
from ..platform.platform import Platform


class Agent(ABC):
	def __init__(
			self,
			platform: Platform,
			model: str,
			response_model: Optional[BaseModel] = None,
			system_prompt: Optional[str] = None,
	):
		self.platform=platform
		self.model=model
		self.response_model=response_model
		self.system_prompt=system_prompt


	@abstractmethod
	def run(self, prompt: str, media: Optional[List[Media]] = None, **kwargs) -> Any:
		pass
