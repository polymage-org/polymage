from abc import ABC, abstractmethod
from typing import Optional, List, Any
from openai import OpenAI
from pydantic import BaseModel
from PIL import Image

from ..model.model import Model
from ..media.media import Media
from ..media.image_media import ImageMedia
from ..platform.platform import Platform

gemma_3_27b = Model(
    name= "gemma-3-27b",
    internal_name= "gemma-3-27b-it-qat",
    capabilities = ["text2text", "text2image"],
    default_params = {
        "steps": 8,
    }
)

qwen3_vl_30b = Model(
    name= "qwen3-vl-30b",
    internal_name= "qwen/qwen3-vl-30b",
    capabilities = ["text2text", "text2image"],
    default_params = {
        "num_steps": 20,
        "height": 1024,
        "width": 1024,
    }
)

class LMStudioPlatform(Platform):
	def __init__(self, host: str = "127.0.0.1:1234", **kwargs):
		super().__init__('lmstudio', list((gemma_3_27b, qwen3_vl_30b)), **kwargs)
		self._host = host
		self._api_key = "lm-studio"  # Dummy key (LM Studio doesn't require real keys)


	def _text2text(self, model: Model, prompt: str, media: Optional[List[Media]] = None, response_model: Optional[BaseModel] = None, **kwargs) -> str:
		system_prompt: Optional[str] = kwargs.get("system_prompt", "")
		client = OpenAI(
				base_url=f"http://{self._host}/v1",  # LM Studio's default endpoint
				api_key=self._api_key
			)
		response = client.chat.completions.create(
			model=model.model_internal_name(),
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": prompt}
			],
			temperature=0.8
		)
		return response.choices[0].message.content.strip()


	def _text2data(self, model: Model, prompt: str, media: Optional[List[Media]] = None, response_model: Optional[BaseModel] = None, **kwargs) -> str:
		system_prompt: Optional[str] = kwargs.get("system_prompt", "")
		client = OpenAI(
				base_url=f"http://{self._host}/v1",  # LM Studio's default endpoint
				api_key=self._api_key
		)
		response = client.chat.completions.create(
			model=model.model_internal_name(),
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
	def _text2data(self, model: Model, prompt: str, media: Optional[List[Media]] = None, response_model: Optional[BaseModel] = None, **kwargs) -> BaseModel:
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
			model=model.model_internal_name(),
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


	def _image2text(self, model: Model, prompt: str, media: List[ImageMedia], **kwargs) -> str:
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
			model=model.model_internal_name(),
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

