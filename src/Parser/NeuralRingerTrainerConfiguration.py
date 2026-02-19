import importlib
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from src.Parser.DynamicConfiguration import DynamicConfiguration


class NeuralRingerTrainerConfiguration(BaseModel):
    batch_size: int
    epochs: int
    et_range_idx: list[int]
    eta_range_idx: list[int]
    n_initializations: int
    base_data_path: str
    num_workers: int
    debug: bool = False
    results_folder_path: str | None

    loss_function: DynamicConfiguration
    optimizer_function: DynamicConfiguration
    model: DynamicConfiguration
    dataset: DynamicConfiguration

    loss_fn_instance: Any = None
    optimizer_instance: Any = None
    model_instance: Any = None
    dataset_instance: Any = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def initialize_components(self):
        component_map = {
            "loss_function": "loss_fn_instance",
            "optimizer_function": "optimizer_instance",
            "model": "model_instance",
            "dataset": "dataset_instance",
        }
        for config_field, dest_field in component_map.items():
            config_obj = getattr(self, config_field)
            instance = self._get_instance(config_obj)
            setattr(self, dest_field, instance)
        return self

    def _get_instance(self, configuration: DynamicConfiguration):
        try:
            module = importlib.import_module(configuration.module)
            target_class = getattr(module, configuration.object_name)

            return target_class(**configuration.parameters)
        except ImportError:
            raise ValueError(f"Could not import module '{configuration.module}'")
        except AttributeError:
            raise ValueError(
                f"Could not find '{configuration.object_name}' in '{configuration.module}'"
            )
        except Exception as e:
            raise ValueError(f"Error initializing {configuration.object_name}: {e}")
