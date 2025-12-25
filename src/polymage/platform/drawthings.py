import logging
import requests
from typing import List, Any
from pydantic import BaseModel
from PIL import Image

from .platform import Platform
from ..model.model import Model
from ..utils.image_utils import fit_to_nearest_aspect_ratio
from ..media.image_media import ImageMedia

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

#
# installed models default params
#
flux_1_schnell = Model(
	name="flux-1-schnell",
	internal_name="flux_1_schnell_q8p.ckpt",
	capabilities=["text2image"],
	default_params={
        "model": "flux_1_schnell_q8p.ckpt",
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
        "resolution_dependent_shift": False,
        "loras": [],
    },
)

flux_1_dev = Model(
	name="flux-1-dev",
	internal_name="flux_1_dev_q8p.ckpt",
	capabilities=["text2image"],
	default_params={
        "model": "flux_1_schnell_q8p.ckpt",
        "negative_prompt": "",
        "steps": 20,
        "batch_count": 1,
        "sampler": "DPM++ 2M Trailing",
        "seed": -1,
        "hires_fix": False,
        "tiled_decoding": False,
        "clip_skip": 1,
        "shift": 1.0,
        "guidance_scale": 2.0,
        "resolution_dependent_shift": True,
        "loras": [],
    },
)

flux_2_dev = Model(
	name="flux-2-dev",
	internal_name="flux_2_dev_q8p.ckpt",
	capabilities=["text2image"],
	default_params={
        "model": "flux_2_dev_q8p.ckpt",
        "negative_prompt": "",
        "steps": 20,
        "batch_count": 1,
        "sampler": "DPM++ 2M Trailing",
        "seed": -1,
        "hires_fix": False,
        "tiled_decoding": False,
        "clip_skip": 1,
        "shift": 1.0,
        "guidance_scale": 2.0,
        "resolution_dependent_shift": False,
        "loras": [],	},
)

hidream_fast = Model(
    name = "hidream-fast",
    internal_name = "hidream_i1_fast_q8p.ckpt",
    capabilities = ["text2image"],
    default_params = {
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
        "resolution_dependent_shift": False,
        "loras": [],
    },
)

hidream_dev = Model(
    name = "hidream-dev",
    internal_name = "hidream_i1_dev_q8p.ckpt",
    capabilities = ["text2image"],
    default_params = {
        "model": "hidream_i1_dev_q8p.ckpt",
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
        "resolution_dependent_shift": False,
        "loras": [],
    },
)

qwen_image_edit_4_steps = Model(
    name = "qwen-image-edit-4-steps",
    internal_name = "qwen_image_edit_2509_q8p.ckpt",
    capabilities = ["text2image", "image2image"],
    default_params = {
        "model": "qwen_image_edit_2509_q8p.ckpt",
        "negative_prompt": "",
        "steps": 4,
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
    },
)


qwen_image_edit_8_steps = Model(
    name = "qwen-image-edit-8-steps",
    internal_name = "qwen_image_edit_2509_q8p.ckpt",
    capabilities = ["text2image", "image2image"],
    default_params = {
        "model": "qwen_image_edit_2509_q8p.ckpt",
        "negative_prompt": "",
        "steps": 10,
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
    },
)

zimage_turbo = Model(
    name = "zimage-turbo",
    internal_name = "zimage_turbo_q8p.ckpt",
    capabilities = ["text2image"],
    default_params = {
        "model": "z_image_turbo_1.0_q8p.ckpt",
        "negative_prompt": "",
        "steps": 8,
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
    },
)

class DrawThingsPlatform(Platform):
    def __init__(self, host: str = "127.0.0.1:7860", **kwargs: Any) -> None:
        super().__init__('drawthings',  list((flux_1_schnell, flux_1_dev, flux_2_dev, hidream_fast, qwen_image_edit_4_steps, qwen_image_edit_8_steps, zimage_turbo)), **kwargs)
        self.host = host


    def _text2image(self, model: Model, prompt: str, **kwargs: Any) -> ImageMedia:
        payload = model.model_default_params()
        payload["prompt"] = prompt

        try:
            response = requests.post(f"http://{self.host}/sdapi/v1/txt2img", json=payload)
            response.raise_for_status()
            json_data = response.json()
            base64_string = json_data["images"][0]
            return ImageMedia(base64_string, {'Software': f"{self.platform_name()}/{model.model_name()}", 'Description': prompt})
        except Exception as e:
            logging.error("API call failed", exc_info=True)
            raise


    def _image2image(self, model: Model, prompt: str, media: ImageMedia, **kwargs: Any) -> ImageMedia:
        payload = model.model_default_params()
        payload["prompt"] = prompt
        # for image2image it's better to fit the to the nearest aspect ratio
        media._image = fit_to_nearest_aspect_ratio(media._image)
        # and we need to pass the image size
        width, height = media._image.size
        payload["width"] = width
        payload["height"]  = height
        # convert the image to base64
        base64_image = media.to_base64()
        payload["init_images"] = [base64_image]

        try:
            response = requests.post(f"http://{self.host}/sdapi/v1/img2img", json=payload)
            response.raise_for_status()
            json_data = response.json()
            base64_string = json_data["images"][0]
            return ImageMedia(base64_string, {'Software': f"{self.platform_name()}/{model.model_name()}"})
        except Exception as e:
            logging.error("API call failed", exc_info=True)
            raise


    def _text2text(self, model: Model, prompt: str, **kwargs: Any) -> Any:
        """Not supported"""
        pass

    def _text2data(self, model: Model, response_model: BaseModel, prompt: str, **kwargs: Any) -> Any:
        """Not supported"""
        pass

    def _image2text(self, model: Model, prompt: str, image: Image.Image, **kwargs: Any) -> str:
        """Not supported"""
        pass

    def _image2data(self, model: Model, response_model: BaseModel, prompt: str, image: Image.Image, **kwargs: Any) -> Any:
        """Not supported"""
        pass
