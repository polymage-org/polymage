import random
import logging
import requests
from typing import Any, List
from pydantic import BaseModel
from PIL import Image

from .platform import Platform
from ..model.model import Model
from ..media.image_media import ImageMedia

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

"""
Clouflare provide AI workers with some free tier

Clouflare support LLm and some other multimedia models
you can find the list of supported models here : https://developers.cloudflare.com/workers-ai/models/
"""

flux_1_schnell = Model(
	name="flux-1-schnell",
	internal_name="@cf/black-forest-labs/flux-1-schnell",
	capabilities=["text2image"],
	default_params={
		"steps": 8,
	},
	output_type="bytes",
)

dreamshaper_8_lcm = Model(
	name="dreamshaper-8-lcm",
	internal_name="@cf/lykon/dreamshaper-8-lcm",
	capabilities=["text2image"],
	default_params={
		"num_steps": 20,
		"height": 1024,
		"width": 1024,
		"guidance": 2.0,
	},
	output_type="bytes",
)

lucid_origin = Model(
	name="lucid-origin",
	internal_name="@cf/leonardo/lucid-origin",
	capabilities=["text2image"],
	default_params={
		"num_steps": 40,
		"height": 1120,
		"width": 1120,
		"guidance": 2.0,
	},
	output_type="base64",
)


class CloudflarePlatform(Platform):
	def __init__(self, api_id: str, api_key: str, **kwargs: Any) -> None:
		super().__init__('lmstudio', list((flux_1_schnell, dreamshaper_8_lcm, lucid_origin)), **kwargs)
		self._api_id = api_id
		self._api_key = api_key

	def _text2image(self, model: Model, prompt: str, **kwargs: Any) -> ImageMedia:
		CLOUDFLARE_ID = self._api_id
		CLOUDFLARE_TOKEN = self._api_key

		payload = model.model_default_params()
		# add the prompt to the params
		payload["prompt"] = prompt

		url = "https://api.cloudflare.com/client/v4/accounts/" + CLOUDFLARE_ID + "/ai/run/" + model.model_internal_name()
		headers = {
			'Content-Type': 'application/json',
			'Authorization': 'Bearer ' + CLOUDFLARE_TOKEN
		}
		data = payload
		try:
			response = requests.post(url, headers=headers, json=data)
			response.raise_for_status()  # Raise an exception for HTTP errors
			if model.model_output_type() == "bytes":
				# image is returned as binary
				image_data = response.content
				return ImageMedia(image_data,{'Software': f"{self.platform_name()}/{model.model_name()}", 'Description': prompt})
			else:
				# image is returned as base64
				result = response.json()
				image_data = result['result']['image']
				return ImageMedia(image_data,{'Software': f"{self.platform_name()}/{model.model_name()}", 'Description': prompt})
		except Exception as e:
			logging.error("API call failed", exc_info=True)
			raise

	def _image2image(self, model: str, prompt: str, image: Image.Image, **kwargs: Any) -> ImageMedia:
		pass

	def _text2text(self, model: str, prompt: str, **kwargs: Any) -> Any:
		"""Not supported"""
		pass

	def _text2data(self, model: str, response_model: BaseModel, prompt: str, **kwargs: Any) -> Any:
		"""Not supported"""
		pass

	def _image2text(self, model: str, prompt: str, image: Image.Image, **kwargs: Any) -> str:
		"""Not supported"""
		pass

	def _image2data(self, model: str, response_model: BaseModel, prompt: str, image: Image.Image, **kwargs: Any) -> Any:
		"""Not supported"""
		pass
