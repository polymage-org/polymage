import logging
from typing import Any, Optional, List

from .agent import Agent
from ..media.media import Media

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

"""
an image captioner agent
"""

class ImageCaptionerAgent(Agent):
	"""
	An agent responsible for generating captions or descriptions for images.

	This class inherits from Agent and provides functionality to convert image
	content into text descriptions using a specified platform and model. It leverages
	the platform's image-to-text capabilities to process media files and generate
	relevant textual output based on the provided prompt.

	The agent utilizes the platform's `image2text` method to perform the actual
	image-to-text conversion, making it suitable for tasks such as image captioning,
	visual description generation, or content analysis of media files.

	Example usage:
		agent = ImageCaptionerAgent()
		result = agent.run(prompt="Describe this image", media=[image_file])

	Attributes:
		Inherits all attributes from the parent Agent class.
	"""
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def run(self, prompt: str, media: Optional[List[Media]] = None, **kwargs) -> Any:
		"""
		Execute the image captioning process.

		Args:
			prompt (str): The prompt or instruction for image description generation.
			media (Optional[List[Media]]): List of media objects to process.
										  Defaults to None.
			**kwargs: Additional keyword arguments passed to the platform's
					 image2text method.

		Returns:
			Any: The result from the platform's image-to-text conversion, typically
				 a string or structured data containing the generated caption or
				 description.
		"""
		platform=self.platform
		model=self.model
		system_prompt=self.system_prompt

		return platform.image2text(model=model, prompt=prompt, media=media, **kwargs)
