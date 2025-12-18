# image_media.py
import base64
from io import BytesIO
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from typing import Optional, Dict, Any, Union
from .media import Media

#
# Conversion utils
#
def base64_to_image(base64_string: str) -> Image.Image:
    """
    Convert a base64 encoded string to a PIL Image object.

    This function decodes a base64 string representation of an image and converts it
    into a PIL (Pillow) Image object that can be manipulated or saved.

    Args:
        base64_string (str): Base64 encoded string representing an image

    Returns:
        PIL.Image.Image: A PIL Image object created from the base64 data

    Example:
        # Convert base64 string to image
        image = base64_to_image("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==")

        # Save the image
        image.save('output.png')

    Note:
        The function supports any image format that PIL can handle (PNG, JPEG, etc.)
    """
    # Decode the base64 string to bytes
    image_bytes = base64.b64decode(base64_string)
    # Create a BytesIO object from the decoded bytes
    image_buffer = BytesIO(image_bytes)
    # Open the image using PIL
    image = Image.open(image_buffer)
    # return a PIL Image object
    return image


def bytes_to_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(image_bytes))
    return image


def image_to_base64(image: Image.Image, format='PNG') -> str:
    """
    Convert a PIL Image object to a base64 encoded string.

    This function takes a PIL Image object and converts it to a base64 encoded string
    which can be used for embedding images in HTML, JSON responses, or other text-based formats.

    Args:
        image (PIL.Image.Image): The PIL Image object to convert
        format (str, optional): The image format to use for encoding. Defaults to 'PNG'.
                               Common formats include 'PNG', 'JPEG', 'GIF'.

    Returns:
        str: Base64 encoded string representation of the image

    Example:
        from PIL import Image
        # Load an image
        img = Image.open('example.jpg')
        # Convert to base64
        base64_string = image_to_base64(img, 'JPEG')

    Note:
        The function uses BytesIO for efficient in-memory image handling and
        returns a UTF-8 decoded base64 string ready for use in text-based protocols.
    """
    # Input is a PIL Image object
    buffered = BytesIO()
    image.save(buffered, format=format)
    encoded_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded_str


class ImageMedia(Media):
    """
    Image media class using Pillow as internal representation.

    This class provides a wrapper around PIL (Pillow) Image objects to handle image media
    with additional functionality for base64 encoding and metadata management.

    Attributes:
        _image (PIL.Image.Image): Internal PIL Image object
        _metadata (Optional[dict]): Metadata associated with the image

    Example:
        from PIL import Image
        pil_img = Image.open('image.jpg')
        image_media = ImageMedia(pil_img, {'author': 'John Doe'})

        # Convert to base64
        base64_data = image_media.to_base64()

        # Save with metadata
        image_media.save_to_file('output.png')
    """

    def __init__(self, image_data: Union[str, bytes, Image.Image], metadata: Optional[Dict[str, Any]] = None, **kwargs):
        self._metadata = metadata
        # Auto-detect based on type
        if isinstance(image_data, str):
            self._image = base64_to_image(image_data)
        elif isinstance(image_data, bytes):
            self._image = bytes_to_image(image_data)
        elif isinstance(image_data, Image.Image):
            self._image = image_data
        else:
            raise TypeError("image_data must be either a Pillow Image or base64-encoded string or raw bytes.")


    def to_base64(self, format='PNG') -> str:
        """
        Convert the image to a base64 encoded string.

        Args:
            format (str): Image format for encoding (default: 'PNG')

        Returns:
            str: Base64 encoded string representation of the image

        Example:
            base64_str = image_media.to_base64('JPEG')
        """
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
        """
        Save the image to a file with metadata support.

        Args:
            filepath (str): Path where the image should be saved

        Example:
            image_media.save_to_file('output.png')
        """
        metadata = self._metadata
        self._image.load()  # Ensures image is fully loaded
        file_metameta = PngInfo()
        if metadata is not None:
            for key, value in metadata.items():
                file_metameta.add_text(key, value)
        # save to file
        self._image.save(filepath, pnginfo=file_metameta)
