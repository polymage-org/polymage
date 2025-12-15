import instructor
from typing import Optional, List, Any
from openai import OpenAI
from pydantic import BaseModel
from PIL import Image

from ..media.media import Media
from ..media.image_media import ImageMedia
from ..platform.platform import Platform


class LMStudioPlatform(Platform):
	def __init__(self, host: str = "127.0.0.1:1234", **kwargs):
		super().__init__("lmstudio", **kwargs)
		self.host = host


	def _text2text(self, model: str, prompt: str, media: Optional[List[Media]] = None, response_model: Optional[BaseModel] = None, **kwargs) -> str:
		system_prompt: Optional[str] = kwargs.get("system_prompt", "")
		client = OpenAI(
				base_url="http://localhost:1234/v1",  # LM Studio's default endpoint
				api_key="lm-studio"  # Dummy key (LM Studio doesn't require real keys)
			)
		response = client.chat.completions.create(
			model=model,  # e.g., "gpt-4o" or local model like "llama-3"
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": prompt}
			],
			temperature=0.8
		)
		return response.choices[0].message.content.strip()


	def _text2data(self, model: str, prompt: str, media: Optional[List[Media]] = None, response_model: Optional[BaseModel] = None, **kwargs) -> BaseModel:
		system_prompt: Optional[str] = kwargs.get("system_prompt", "")
		client = OpenAI(
				base_url="http://localhost:1234/v1",  # LM Studio's default endpoint
				api_key="lm-studio"  # Dummy key (LM Studio doesn't require real keys)
		)
		response = client.chat.completions.create(
			model=model,  # e.g., "gpt-4o" or local model like "llama-3"
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": prompt}
			],
			response_format={
				"type": "json_schema",
				"json_schema": {
					"name": "UserProfile",
					"schema": response_model,
				},
			},
			temperature=0.8,
		)
		return response.choices[0].message.content.strip()


	"""
	#
	# does not work (for now) with Instructor
	#
	def _text2data(self, model: str, prompt: str, media: Optional[List[Media]] = None, response_model: Optional[BaseModel] = None, **kwargs) -> BaseModel:
		system_prompt: Optional[str] = kwargs.get("system_prompt", "")

		# Patch the OpenAI client with Instructor
		client = instructor.from_openai(
			OpenAI(
				base_url="http://localhost:1234/v1",  # LM Studio's default endpoint
				api_key="lm-studio"  # Dummy key (LM Studio doesn't require real keys)
			),
			tool_choice="auto",
			#mode=instructor.Mode.JSON,
		)
		# use structured response model
		response_data = client.chat.completions.create(
			model=model,
			# the data model to use
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": prompt}
			],
			response_model=response_model,
			max_retries=3,  # Auto-retry on validation failures
		)
		return response_data
	"""


	def _image2text(self, model: str, prompt: str, media: List[ImageMedia], **kwargs) -> str:
		client = OpenAI(
			base_url="http://localhost:1234/v1",  # LM Studio's default endpoint
			api_key="lm-studio"  # Dummy key (LM Studio doesn't require real keys)
		)
		if len(media) == 0:
			return ""
		else:
			image = media[0]
			base64_image = image.to_base64()

		response = client.responses.create(
			model=model,
			input=[
				{
					"role": "user",
					"content": [
						{"type": "input_text", "text": prompt},
						{"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"},
					],
				}
			],
		)
		return response.output[0].content[0].text




	def _text2image(self, model: str, prompt: str, **kwargs) -> Image.Image:
		"""Not supported"""
		pass

	def _image2image(self, model: str, prompt: str, image: Image.Image, **kwargs) -> Image.Image:
		"""Not supported"""
		pass

