import random
import requests
from typing import Any, List
from pydantic import BaseModel
from PIL import Image

from .platform import Platform
from ..model.model import Model
from ..media.image_media import ImageMedia, base64_to_image, image_to_base64

"""
Clouflare provide AI workers with some free tier

Clouflare support LLm and some other multimedia models
you can find the list of supported models here : https://developers.cloudflare.com/workers-ai/models/
"""

flux_1_schnell = Model(
    name= "flux-1-schnell",
    internal_name= "@cf/black-forest-labs/flux-1-schnell",
    capabilities = ["text2image"],
    default_params = {
        "steps": 8,
    }
)


dreamshaper_8_lcm = Model(
    name= "dreamshaper-8-lcm",
    internal_name= "@cf/lykon/dreamshaper-8-lcm",
    capabilities = ["text2image"],
    default_params = {
        "num_steps": 20,
        "height": 1024,
        "width": 1024,
    }
)


class CloudflarePlatform(Platform):
    def __init__(self, api_id: str, api_key: str, **kwargs):
        super().__init__('lmstudio',  list((flux_1_schnell, dreamshaper_8_lcm)), **kwargs)
        self._api_id = api_id
        self._api_key = api_key


    def _text2image(self, model: Model, prompt: str, **kwargs) -> ImageMedia:
        CLOUDFLARE_ID = self._api_id
        CLOUDFLARE_TOKEN = self._api_key

        payload = model.model_default_params()
        random_seed = random.randint(0, 2 ** 32 - 1)

        payload["prompt"] = prompt

        url = "https://api.cloudflare.com/client/v4/accounts/" + CLOUDFLARE_ID + "/ai/run/" + model.model_internal_name()
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + CLOUDFLARE_TOKEN
        }
        data = {
            'steps': 5,
            'seed': random_seed,
            'prompt': prompt
        }
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()  # Raise an exception for HTTP errors
            result = response.json()
            #print(f"CloudFlare result = {result}")
            image_data = result['result']['image']
            #image = base64_to_image(base64_string)
            return ImageMedia(image_data, {'Software': f"{self.platform_name()}/{model.model_name()}", 'Description': prompt})
        except Exception as e:
            RuntimeError(f"CloudFlare error: {e}")


    def _image2image(self, model: str, prompt: str, image: Image.Image, **kwargs) -> ImageMedia:
        pass

    def _text2text(self, model: str, prompt: str, **kwargs) -> Any:
        """Not supported"""
        pass

    def _text2data(self, model: str, response_model: BaseModel, prompt: str, **kwargs) -> Any:
        """Not supported"""
        pass

    def _image2text(self, model: str, prompt: str, image: Image.Image, **kwargs) -> str:
        """Not supported"""
        pass

    def _image2data(self, model: str, response_model: BaseModel, prompt: str, image: Image.Image, **kwargs) -> Any:
        """Not supported"""
        pass

