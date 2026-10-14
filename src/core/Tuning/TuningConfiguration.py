import os
from typing import Any, Dict, List, Optional, Tuple
import yaml
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import ParameterGrid

from src.Parser.DynamicConfiguration import DynamicConfiguration
from src.Parser.NeuralRingerTrainerConfiguration import NeuralRingerTrainerConfiguration


class OptimizerSearchItem(BaseModel):
    """Specification for an optimizer search candidate."""

    module: str = Field(default="torch.optim", description="Python module path.")
    object_name: str = Field(..., description="Optimizer class name.")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Optimizer hyperparameter grid."
    )


class LossSearchItem(BaseModel):
    """Specification for a loss function search candidate."""

    module: str = Field(default="torch.nn", description="Python module path.")
    object_name: str = Field(default="BCELoss", description="Loss class name.")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Loss function parameter grid."
    )


class TuningSearchSpace(BaseModel):
    """Composite hyperparameter search space."""

    optimizers: List[OptimizerSearchItem] = Field(
        ..., description="List of optimizers with parameter values to explore."
    )
    loss_functions: Optional[List[LossSearchItem]] = Field(
        default=None, description="Optional loss functions to explore."
    )


class TuningConfiguration(BaseModel):
    """Top-level schema for hyperparameter search configuration."""

    base_config: str = Field(..., description="Path to base YAML training configuration.")
    overrides: Dict[str, Any] = Field(
        default_factory=dict, description="Parameters overriding base config."
    )
    search_space: TuningSearchSpace = Field(..., description="Hyperparameter search space.")
    output_dir: Optional[str] = Field(
        default=None, description="Base directory to store trial outputs."
    )
    split_manifest_path: Optional[str] = Field(
        default=None, description="Path to pre-existing split manifest JSON."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TuningConfiguration":
        """Loads tuning configuration from YAML file.

        Args:
            yaml_path: Path to tuning YAML configuration.

        Returns:
            Validated TuningConfiguration instance.
        """
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def build_trial_configurations(
        self,
    ) -> List[Tuple[str, NeuralRingerTrainerConfiguration, Dict[str, Any]]]:
        """Generates trial configurations across the declared hyperparameter search space.

        Returns:
            List of tuples: (trial_name, validated_config, metadata_dict).
        """
        with open(self.base_config, "r") as f:
            base_dict = yaml.safe_load(f)

        base_dict.update(self.overrides)
        base_output_dir = self.output_dir or f"results/{base_dict.get('config_name', 'model')}"

        loss_candidates: List[Tuple[str, str, Dict[str, Any]]] = []
        if self.search_space.loss_functions:
            for loss_spec in self.search_space.loss_functions:
                grid = {
                    k: (v if isinstance(v, list) else [v])
                    for k, v in loss_spec.parameters.items()
                }
                param_combinations = list(ParameterGrid(grid)) if grid else [{}]
                for p in param_combinations:
                    loss_candidates.append((loss_spec.module, loss_spec.object_name, p))
        else:
            base_loss = base_dict.get("loss_function", {})
            loss_candidates.append(
                (
                    base_loss.get("module", "torch.nn"),
                    base_loss.get("object_name", "BCELoss"),
                    base_loss.get("parameters", {}),
                )
            )

        trials: List[Tuple[str, NeuralRingerTrainerConfiguration, Dict[str, Any]]] = []
        trial_counter = 1

        for loss_mod, loss_name, loss_params in loss_candidates:
            for opt_spec in self.search_space.optimizers:
                opt_grid = {
                    k: (v if isinstance(v, list) else [v])
                    for k, v in opt_spec.parameters.items()
                }
                opt_param_combinations = list(ParameterGrid(opt_grid)) if opt_grid else [{}]

                for opt_params in opt_param_combinations:
                    opt_params_tag = "_".join(f"{k}{v}" for k, v in sorted(opt_params.items()))
                    trial_tag = (
                        f"trial_{trial_counter:03d}_{opt_spec.object_name}"
                        f"{'_' + opt_params_tag if opt_params_tag else ''}_{loss_name}"
                    )

                    trial_dict = dict(base_dict)
                    trial_dict["loss_function"] = {
                        "module": loss_mod,
                        "object_name": loss_name,
                        "parameters": loss_params,
                    }
                    trial_dict["optimizer_function"] = {
                        "module": opt_spec.module,
                        "object_name": opt_spec.object_name,
                        "parameters": opt_params,
                    }
                    trial_dict["config_name"] = f"{base_dict.get('config_name', 'tuning')}_{trial_tag}"
                    trial_dict["results_folder_path"] = os.path.join(base_output_dir, trial_tag)
                    if self.split_manifest_path:
                        trial_dict["split_manifest_path"] = self.split_manifest_path

                    config_instance = NeuralRingerTrainerConfiguration.model_validate(trial_dict)

                    metadata = {
                        "trial_name": trial_tag,
                        "trial_index": trial_counter,
                        "optimizer": opt_spec.object_name,
                        "optimizer_module": opt_spec.module,
                        "optimizer_parameters": opt_params,
                        "loss_function": loss_name,
                        "loss_module": loss_mod,
                        "loss_parameters": loss_params,
                        "results_folder_path": trial_dict["results_folder_path"],
                    }

                    trials.append((trial_tag, config_instance, metadata))
                    trial_counter += 1

        return trials
