import base64

import requests
from typing import Any
from pydantic import BaseModel
from PIL import Image

from .platform import Platform
from ..media.image_media import ImageMedia, base64_to_image, image_to_base64

#
# installed models default params
#
hidream_fast_settings = {
    "model": "hidream_i1_fast_q8p.ckpt",
    "negative_prompt": "",
    "steps": 14,
    "batch_count": 1,
    "sampler": "DPM++ 2M Trailing",
    "seed": -1,
    "hires_fix": False,
    "tiled_decoding": False,
    "clip_skip": 1,
    "shift": 1.0,
    "guidance_scale": 1.0,
    "resolution_dependent_shift": False
}

flux_kontext_settings = {
    "model": "flux_1_kontext_dev_q8p.ckpt",
    "negative_prompt": "",
    "steps": 8,
    "batch_count": 1,
    "sampler": "DPM++ 2M Trailing",
    "seed": -1,
    "hires_fix": False,
    "tiled_decoding": False,
    "clip_skip": 1,
    "shift": 1.0,
    "guidance_scale": 2.0,
    "resolution_dependent_shift": True,
    "loras": [{"file": "flux.1_turbo_alpha_lora_f16.ckpt", "weight": 1, "mode": "all"}],
}

qwen_image_edit_settings_4_steps = {
    "model": "qwen_image_edit_2509_q8p.ckpt",
    "negative_prompt": "",
    "steps": 5,
    "batch_count": 1,
    "sampler": "DPM++ 2M Trailing",
    "seed": -1,
    "hires_fix": False,
    "tiled_decoding": False,
    "clip_skip": 1,
    "shift": 1.0,
    "guidance_scale": 2.0,
    "resolution_dependent_shift": True,
    "loras": [{"file": "qwen_image_edit_2509_lightning_4_step_v1.0_lora_f16.ckpt", "weight": 1, "mode": "all"}],
}

qwen_image_edit_settings_8_steps = {
    "model": "qwen_image_edit_2509_q8p.ckpt",
    "negative_prompt": "",
    "steps": 8,
    "batch_count": 1,
    "sampler": "DPM++ 2M Trailing",
    "seed": -1,
    "hires_fix": False,
    "tiled_decoding": False,
    "clip_skip": 1,
    "shift": 1.0,
    "guidance_scale": 2.0,
    "resolution_dependent_shift": True,
    "loras": [{"file": "qwen_image_edit_2509_lightning_8_step_v1.0_lora_f16.ckpt", "weight": 1, "mode": "all"}],
}

zimage_turbo_settings = {
    "model": "z_image_turbo_1.0_q8p.ckpt",
    "negative_prompt": "",
    "steps": 10,
    "batch_count": 1,
    "sampler": "UniPC Trailing",
    "seed": -1,
    "hires_fix": False,
    "tiled_decoding": False,
    "clip_skip": 1,
    "shift": 1.0,
    "guidance_scale": 1.0,
    "resolution_dependent_shift": False,
    "loras": [],
}


MODELS_SETTINGS = {
    "hidream_fast": hidream_fast_settings,
    "flux_kontext": flux_kontext_settings,
    "qwen_image_edit_4_steps": qwen_image_edit_settings_4_steps,
    "qwen_image_edit_8_steps": qwen_image_edit_settings_8_steps,
    "zimage_turbo": zimage_turbo_settings,
}


class DrawThingsPlatform(Platform):
    def __init__(self, host: str = "127.0.0.1:7860", **kwargs):
        super().__init__("drawthing", **kwargs)
        self.host = host


    def _text2image(self, model: str, prompt: str, **kwargs) -> ImageMedia:
        payload = MODELS_SETTINGS[model]
        payload["prompt"] = prompt

        try:
            response = requests.post(f"http://{self.host}/sdapi/v1/txt2img", json=payload)
            response.raise_for_status()
            json_data = response.json()
            base64_string = json_data["images"][0]
            image = base64_to_image(base64_string)
            return ImageMedia(image, {'plaform': "DrawThings", 'model': model, 'prompt': prompt})
        except Exception as e:
            raise RuntimeError(f"DrawThings error: {e}")


    def _image2image(self, model: str, prompt: str, image: Image.Image, **kwargs) -> ImageMedia:
        payload = MODELS_SETTINGS[model]
        payload["prompt"] = prompt
        base64_image = image_to_base64(image)
        payload["init_images"] = [base64_image]

        try:
            response = requests.post(f"http://{self.host}/sdapi/v1/img2img", json=payload)
            response.raise_for_status()
            json_data = response.json()
            base64_string = json_data["images"][0]
            image = image_to_base64(base64_string)
            return ImageMedia(image, {'plaform': self._name, 'model': model, 'prompt': prompt})
        except Exception as e:
            raise RuntimeError(f"DrawThings error: {e}")


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
