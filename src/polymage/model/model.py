from abc import ABC
from typing import Dict, List, Any


class Model(ABC):

    def __init__(self, name: str, internal_name: str, capabilities: List[str], default_params:Dict[str, Any], **kwargs: Any) -> None:
        self._name = name
        self._internal_name = internal_name
        self._capabilities = capabilities
        self._default_params = default_params
        self._output_type = kwargs.get("output_type", "str")


    def model_name(self) -> str:
        return self._name

    def model_internal_name(self) -> str:
        return self._internal_name

    def model_output_type(self) -> str:
        return self._output_type

    def model_capabilities(self) -> List[str]:
        return self._capabilities

    def model_default_params(self) -> Dict[str, Any]:
        return self._default_params
