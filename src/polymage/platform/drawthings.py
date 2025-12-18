import requests
from typing import List, Any
from pydantic import BaseModel
from PIL import Image, ImageOps

from .platform import Platform
from ..model.model import Model
from ..media.image_media import ImageMedia, base64_to_image, image_to_base64

#
# drawthings enforce some image ratio size (multiple of 128)
#
ASPECT_RATIO_SIZE = {
    "1:1": [1024, 1024],
    "5:4": [1152, 896],
    "4:3": [1024, 768],
    "3:2": [1152, 768],
    "2:1": [1408, 704],
    "16,9": [1024, 576],
    "9,16": [576, 1024],
    "4,5": [896, 1152],
    "3,4": [768, 1024],
    "2,3": [768, 1152],
    "1,2": [704, 1408],
}

#
# for image2image it's better to fit the image size to the nearest aspect ratio
#
def fit_to_nearest_aspect_ratio(image: Image.Image) -> Image.Image:
    """
    Fits the input PIL image to the nearest aspect ratio defined in ASPECT_RATIO_SIZE,
    using ImageOps.fit. Returns the cropped and resized image.
    """
    img_width, img_height = image.size
    img_aspect = img_width / img_height

    min_diff = float('inf')
    best_size = None

    for size in ASPECT_RATIO_SIZE.values():
        w, h = size
        ar = w / h
        diff = abs(img_aspect - ar)
        if diff < min_diff:
            min_diff = diff
            best_size = (w, h)
    # Use ImageOps.fit to crop and resize to the best matching aspect ratio
    fitted_image = ImageOps.fit(image, best_size, method=Image.LANCZOS)
    return fitted_image


#
# installed models default params
#
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
    }
)

qwen_image_edit_8_steps = Model(
    name = "qwen-image-edit-8-steps",
    internal_name = "qwen_image_edit_2509_q8p.ckpt",
    capabilities = ["text2image", "image2image"],
    default_params = {
        "model": "qwen_image_edit_2509_q8p.ckpt",
        "negative_prompt": "",
        "steps": 9,
        "batch_count": 1,
        "sampler": "DPM++ 2M Trailing",
        "seed": -1,
        "hires_fix": False,
        "tiled_decoding": False,
        "clip_skip": 1,
        "shift": 1.0,
        "guidance_scale": 1.0,
        "resolution_dependent_shift": True,
        "loras": [{"file": "qwen_image_edit_2509_lightning_8_step_v1.0_lora_f16.ckpt", "weight": 1, "mode": "all"}],
    }
)

zimage_turbo = Model(
    name = "zimage-turbo",
    internal_name = "zimage_turbo_q8p.ckpt",
    capabilities = ["text2image"],
    default_params = {
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
)

class DrawThingsPlatform(Platform):
    def __init__(self, host: str = "127.0.0.1:7860", **kwargs):
        super().__init__('lmstudio',  list((hidream_fast, qwen_image_edit_8_steps, zimage_turbo)), **kwargs)
        self.host = host


    def _text2image(self, model: Model, prompt: str, **kwargs) -> ImageMedia:
        payload = model.model_default_params()
        payload["prompt"] = prompt

        try:
            response = requests.post(f"http://{self.host}/sdapi/v1/txt2img", json=payload)
            response.raise_for_status()
            json_data = response.json()
            base64_string = json_data["images"][0]
            return ImageMedia(base64_string, {'Software': f"{self.platform_name()}/{model.model_name()}", 'Description': prompt})
        except Exception as e:
            raise RuntimeError(f"DrawThings error: {e}")


    def _image2image(self, model: Model, prompt: str, media: ImageMedia, **kwargs) -> ImageMedia:
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
            raise RuntimeError(f"DrawThings error: {e}")


    def _text2text(self, model: Model, prompt: str, **kwargs) -> Any:
        """Not supported"""
        pass

    def _text2data(self, model: Model, response_model: BaseModel, prompt: str, **kwargs) -> Any:
        """Not supported"""
        pass

    def _image2text(self, model: Model, prompt: str, image: Image.Image, **kwargs) -> str:
        """Not supported"""
        pass

    def _image2data(self, model: Model, response_model: BaseModel, prompt: str, image: Image.Image, **kwargs) -> Any:
        """Not supported"""
        pass
