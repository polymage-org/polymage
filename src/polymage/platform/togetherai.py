import json
import logging
from typing import Optional, List, Any
from openai import OpenAI
from pydantic import BaseModel
from PIL import Image
from tenacity import retry, stop_after_attempt, retry_if_exception_type

from ..model.model import Model
from ..media.media import Media
from ..media.image_media import ImageMedia
from ..platform.platform import Platform


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


gpt_oss_20b = Model(
    name= "gpt-oss-20b",
    internal_name= "openai/gpt-oss-20b",
    capabilities = ["text2text"],
    default_params = {
    }
)

class TogetherAiPlatform(Platform):
	def __init__(self, api_key: str, **kwargs: Any) -> None:
		super().__init__('togetherai', list((gpt_oss_20b,)), **kwargs)
		self._api_key = api_key


	def _text2text(self, model: Model, prompt: str, media: Optional[List[Media]] = None, response_model: Optional[BaseModel] = None, **kwargs: Any) -> str:
		system_prompt: Optional[str] = kwargs.get("system_prompt", "")
		client = OpenAI(
				base_url=f"https://api.together.xyz/v1",  # TogetherAi's default endpoint
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


	#
	# using structured data may sometime fail, because the result is not a valid JSON
	# if the JSON is not valid, retry 3 times
	#
	@retry(retry=retry_if_exception_type(json.JSONDecodeError), stop=stop_after_attempt(3))
	def _text2data(self, model: Model, prompt: str, response_model: BaseModel, media: Optional[List[Media]] = None, **kwargs: Any) -> str:
		system_prompt: Optional[str] = kwargs.get("system_prompt", "")
		client = OpenAI(
				base_url=f"http://{self._host}/v1",  # LM Studio's default endpoint
				api_key=self._api_key
		)

		json_schema = response_model.model_json_schema()
		json_schema_name = json_schema['title']

		chat_completion = client.chat.completions.create(
			model=model.model_internal_name(),
			messages=[
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": prompt}
			],
			response_format={
				"type": "json_schema",
				"json_schema": {
					"name": json_schema_name,
					"schema": json_schema,
				},
			},
			temperature=0.8,
		)
		json_string = chat_completion.choices[0].message.content.strip()
		# return a python Dict
		return json.loads(json_string)


	def _image2text(self, model: Model, prompt: str, media: List[ImageMedia], api_key=None, **kwargs: Any) -> str:
		client = OpenAI(
			base_url="https://api.together.xyz/v1",
			api_key=self._api_key,
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




	def _text2image(self, model: str, prompt: str, **kwargs: Any) -> Image.Image:
		"""Not supported"""
		pass

	def _image2image(self, model: str, prompt: str, image: Image.Image, **kwargs: Any) -> Image.Image:
		"""Not supported"""
		pass

