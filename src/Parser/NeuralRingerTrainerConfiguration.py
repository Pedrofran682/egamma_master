from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

from src.Parser.DynamicConfiguration import DynamicConfiguration


class NeuralRingerTrainerConfiguration(BaseModel):
    """Pydantic validation schema for Neural Ringer model training YAML specifications.

    Attributes:
        batch_size: Mini-batch sample size for training and validation loaders.
        config_name: Unique tag identifier for the experiment run.
        epochs: Number of training epochs per cross-validation fold.
        et_range_idx: List of target transverse energy (ET) bin indices.
        eta_range_idx: List of target pseudorapidity (eta) bin indices.
        n_initializations: Number of random initialization repeats per fold.
        pred_target_limiar: Decision boundary cutoff for binary classification.
        use_trigger_filter: Flag to enable additional trigger filtering.
        trigger_filter: Trigger feature name when filter is active.
        num_workers: Subprocess workers count for PyTorch DataLoader.
        debug: Activates fast debug run mode (epochs=2, n_initializations=1).
        results_folder_path: Explicit output folder path (auto-generated if None).
        balance_data: Whether to compute and apply class balancing weights.
        loss_function: Dynamic specification for the loss criterion.
        optimizer_function: Dynamic specification for the optimizer.
        model: Dynamic specification for the PyTorch neural network.
        dataset: Dynamic specification for the dataset class.
        kFold: Dynamic specification for the cross-validation splitter.
    """

    batch_size: int = Field(..., description="Batch size per training step.")
    config_name: str = Field(..., description="Experiment run name.")
    epochs: int = Field(..., description="Number of training epochs.")
    et_range_idx: List[int] = Field(..., description="List of ET bin indices.")
    eta_range_idx: List[int] = Field(..., description="List of eta bin indices.")
    n_initializations: int = Field(..., description="Number of repeats per fold.")
    pred_target_limiar: float = Field(..., description="Probability threshold for positive class.")
    use_trigger_filter: bool = Field(default=False, description="Enable trigger filter.")
    trigger_filter: str = Field(default="", description="Trigger column name.")
    num_workers: int = Field(default=0, description="DataLoader worker subprocesses.")
    debug: bool = Field(default=False, description="Debug mode flag.")
    results_folder_path: Optional[str] = Field(default=None, description="Output folder path.")
    balance_data: bool = Field(default=True, description="Enable class weighting sampler.")
    loss_function: DynamicConfiguration = Field(..., description="Loss function config.")
    optimizer_function: DynamicConfiguration = Field(..., description="Optimizer config.")
    model: DynamicConfiguration = Field(..., description="Neural network architecture config.")
    dataset: DynamicConfiguration = Field(..., description="Dataset loader config.")
    kFold: DynamicConfiguration = Field(..., description="Cross validation config.")

    model_config = ConfigDict(arbitrary_types_allowed=True)
