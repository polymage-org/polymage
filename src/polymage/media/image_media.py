# image_media.py
import base64
from io import BytesIO
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from typing import Optional
from .media import Media


#
# Conversion utils
#
def base64_to_image(base64_string: str) -> Image.Image:
    # Decode the base64 string to bytes
    image_bytes = base64.b64decode(base64_string)
    # Create a BytesIO object from the decoded bytes
    image_buffer = BytesIO(image_bytes)
    # Open the image using PIL
    image = Image.open(image_buffer)
    # return a PIL Image object
    return image


def image_to_base64(image: Image.Image, format='PNG') -> str:
    # Input is a PIL Image object
    buffered = BytesIO()
    image.save(buffered, format=format)
    encoded_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded_str


class ImageMedia(Media):
    """
    Image media class using Pillow as internal representation.
    """
    
    def __init__(self, pil_image: Image.Image, metadata: Optional[dict] = None, **kwargs):
        """
        Initialize with a PIL Image object.
        
        Args:
            pil_image: PIL Image object
        """
        if not isinstance(pil_image, Image.Image):
            raise TypeError("Expected PIL.Image.Image object")
        self._image = pil_image
        self._metadata = metadata


    def to_base64(self, format='PNG') -> str:
        # Create an in-memory bytes buffer
        buffer = BytesIO()
        # Save the image to the buffer in the specified format
        self._image.save(buffer, format=format)
        # Get the bytes from the buffer
        image_bytes = buffer.getvalue()
        # Encode the bytes as base64 and decode to a string
        base64_str = base64.b64encode(image_bytes).decode('utf-8')
        return base64_str


    def save_to_file(self, filepath: str) -> None:
        metadata = self._metadata
        self._image.load()  # Ensures image is fully loaded
        file_metameta = PngInfo()
        if metadata is not None:
            for key, value in metadata.items():
                file_metameta.add_text(key, value)
        # save to file
        self._image.save(filepath, pnginfo=file_metameta)


