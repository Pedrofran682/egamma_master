from typing import Any, Dict

from pydantic import BaseModel


class DynamicConfiguration(BaseModel):
    object_name: str
    module: str
    parameters: Dict[str, Any] = {}
