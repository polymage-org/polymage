from abc import ABC, abstractmethod
from typing import List, Optional, Any
from pydantic import BaseModel

from ..media.media import Media
from ..media.image_media import ImageMedia

class Platform(ABC):
    """Abstract base for AI platform connectors"""
    def __init__(self, name: str, **kwargs):
        self._name = name


    def text2text(self, model: str, prompt: str, media: Optional[List[Media]] = None, response_model: Optional[BaseModel] = None, **kwargs) -> Any:
        # basic text output
        if response_model is None:
            return self._text2text(model, prompt, media=media, response_model=response_model, **kwargs)
        # structured data output
        else:
            return self._text2data(model, prompt, media=media, response_model=response_model, **kwargs)


    @abstractmethod
    def _text2text(self, model: str, prompt: str, media: Optional[List[Media]] = None, response_model: Optional[BaseModel] = None, **kwargs) -> Any:
        """Platform-specific execution interface"""
        pass


    @abstractmethod
    def _text2data(self, model: str, prompt: str, media: Optional[List[Media]] = None, response_model: Optional[BaseModel] = None, **kwargs) -> Any:
        """Platform-specific execution interface"""
        pass




    def text2image(self, model: str, prompt: str, **kwargs) -> ImageMedia:
        return self._text2image(model, prompt, **kwargs)

    @abstractmethod
    def _text2image(self, model: str, prompt: str, **kwargs) -> ImageMedia:
        """Platform-specific execution interface"""
        pass


    def image2text(self, model: str, prompt: str, media: List[ImageMedia], **kwargs) -> str:
        return self._image2text(model, prompt, media=media, **kwargs)

    @abstractmethod
    def _image2text(self, model: str, prompt: str, media: List[ImageMedia], **kwargs) -> str:
        """Platform-specific execution interface"""
        pass


    def image2image(self, model: str, prompt: str, media: List[ImageMedia], **kwargs) -> ImageMedia:
        return self._image2image(model, prompt, media=media, **kwargs)

    @abstractmethod
    def _image2image(self, model: str, prompt: str, media: List[ImageMedia], **kwargs) -> ImageMedia:
        """Platform-specific execution interface"""
        pass


