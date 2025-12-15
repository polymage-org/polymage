from typing import Optional, List
from pydantic import BaseModel
from PIL import Image
from groq import Groq

from ..media.media import Media
from ..media.image_media import ImageMedia
from .platform import Platform

"""
groq platform

groq support LLm and audio2text models
you can find the list of supported models here : https://console.groq.com/docs/models
"""


class GroqPlatform(Platform):
    def __init__(self, api_key: str, **kwargs):
        super().__init__('groq', **kwargs)
        self._api_key = api_key

    def _text2text(self, model: str, prompt: str, media: Optional[List[Media]] = None, response_model: Optional[BaseModel] = None, **kwargs) -> str:
        system_prompt: Optional[str] = kwargs.get("system_prompt", "You are a helpful assistant.")
        client = Groq(api_key=self._api_key)
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        print(f"Groq text2text : {chat_completion.choices[0].message}")
        return chat_completion.choices[0].message.content.strip()


    def _text2data(self, model: str, prompt: str, media: Optional[List[Media]] = None, response_model: Optional[BaseModel] = None, **kwargs) -> str:
        system_prompt: Optional[str] = kwargs.get("system_prompt", "You are a helpful assistant.")
        client = Groq(api_key=self._api_key)
        chat_completion = client.chat.completions.create(
            model=model,
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
        print(f"Groq text2data : {chat_completion.choices[0].message}")
        return chat_completion.choices[0].message.content.strip()


    def _image2text(self, model: str, prompt: str, media: List[ImageMedia], **kwargs) -> str:
        client = Groq(api_key=self._api_key)
        if len(media) == 0:
            return ""
        else:
            image = media[0]
            base64_image = image.to_base64()

        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            temperature=1,
            max_completion_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        print(f"Groq image2text : {chat_completion.choices[0].message}")
        return chat_completion.choices[0].message.content.strip()


    def _text2image(self, model: str, prompt: str, **kwargs) -> Image.Image:
        """Not supported"""
        pass


    def _image2image(self, model: str, prompt: str, image: Image.Image, **kwargs) -> Image.Image:
        """Not supported"""
        pass


