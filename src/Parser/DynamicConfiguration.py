from typing import Any, Dict
from pydantic import BaseModel, Field


class DynamicConfiguration(BaseModel):
    """Pydantic schema for dynamic class reflection and parameter injection.

    Attributes:
        object_name: Name of the class or callable to import.
        module: Python module dotpath containing the object.
        parameters: Keyword arguments dictionary passed to the class constructor.
    """

    object_name: str = Field(..., description="Target class or function name.")
    module: str = Field(..., description="Python module path (e.g. 'torch.optim').")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Keyword arguments for constructor initialization."
    )
