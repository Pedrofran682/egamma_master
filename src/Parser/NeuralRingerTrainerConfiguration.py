from pydantic import BaseModel, ConfigDict

from src.Parser.DynamicConfiguration import DynamicConfiguration


class NeuralRingerTrainerConfiguration(BaseModel):
    batch_size: int
    config_name: str
    epochs: int
    et_range_idx: list[int]
    eta_range_idx: list[int]
    n_initializations: int
    pred_target_limiar: int
    num_workers: int
    debug: bool = False
    results_folder_path: str | None = None

    loss_function: DynamicConfiguration
    optimizer_function: DynamicConfiguration
    model: DynamicConfiguration
    dataset: DynamicConfiguration
    kFold: DynamicConfiguration

    model_config = ConfigDict(arbitrary_types_allowed=True)
