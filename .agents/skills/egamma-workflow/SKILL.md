---
name: egamma-workflow
description: >-
  Workflows, execution commands, and codebase navigation for egamma_master machine learning pipeline.
  Activate when running training, unit tests, plotting, model evaluations, or adding new models in WSL conda environment egamma.
---

# Egamma Workflow & Command Execution Guide

This skill documents how to execute tasks, navigate the repository, and add components to the `egamma_master` codebase.

---

## 1. Environment & Python Execution

Always run Python commands and scripts using the `egamma` Conda environment in WSL:

```bash
conda run -n egamma <command>
```

Key environment paths:
- Python interpreter: `/home/pmourafr/anaconda3/envs/egamma/bin/python`
- Pytest executable: `/home/pmourafr/anaconda3/envs/egamma/bin/pytest`

---

## 2. Standard Command Workflows

### Running Tests
Execute the full test suite (pytest configuration is defined in `pytest.ini` with `pythonpath = .`):
```bash
conda run -n egamma pytest -v
```
Execute individual test files or packages:
```bash
conda run -n egamma pytest tests/core/trainers/test_trainers.py -v
conda run -n egamma pytest tests/models/test_model_registry.py -v
conda run -n egamma pytest tests/core/validation/test_validation.py -v
conda run -n egamma pytest tests/core/validation/test_quadrant_analysis.py -v
```

### Running Model Training
Train neural ringer models with cross-validation using a YAML config:
```bash
conda run -n egamma python main.py --config config/NeuralRinger/ModelV1.yaml
```
Output results and checkpoints are stored in `results/` and execution logs in `log/`.

### Plotting
- **Metric and ROC curves**:
  ```bash
  conda run -n egamma python scripts/run_plots.py results/<run_folder>
  ```
- **Ringer energy profiles**:
  ```bash
  conda run -n egamma python scripts/run_plots.py results/<run_folder> --plot_ringer --percentage 0.5
  ```
- **Multi-model comparison ROC grid**:
  ```bash
  conda run -n egamma python scripts/run_roc_plots.py --results_path results/
  ```

### Validating Models
Compare model performance against benchmark threshold cuts (supports optional `--efficiencies_csv` and `--target_pd`):
```bash
conda run -n egamma python scripts/validate_pd.py --config config/NeuralRinger/ModelV1.yaml --data_path data/ [--efficiencies_csv <csv_path>] [--target_pd 0.9424]
```
Calculate ATLAS baseline fast photon cut efficiencies:
```bash
conda run -n egamma python scripts/calculate_photon_cut_efficiency.py --data_path data/consolidated/consolidated.et2.eta2.npz --threshold 20.0
```

### Regional Data Distribution Analysis
Analyze and visualize class distributions across $E_T$ and $\eta$ regions:
```bash
conda run -n egamma python scripts/plot_data_distribution.py --config config/NeuralRinger/ModelV1.yaml
```

### Quadrant Analysis (Model Comparison & Cut-Based Benchmark)
Compare two classification models or a model against Athena fast photon cut-based trigger selection on holdout test partitions across $E_T$ and $\eta$ regions:
```bash
# Model vs Model
conda run -n egamma python scripts/run_quadrant_analysis.py \
    --mode model_vs_model \
    --config1 config/NeuralRinger/ModelV1.yaml \
    --data_path1 results/<model1_results_dir> \
    --config2 config/NeuralRinger/ModelV2.yaml \
    --data_path2 results/<model2_results_dir>

# Model vs Cut-Based Selection
conda run -n egamma python scripts/run_quadrant_analysis.py \
    --mode model_vs_cut \
    --config1 config/NeuralRinger/ModelV1.yaml \
    --data_path1 results/<model1_results_dir> \
    --working_point loose
```

### Batch Runs
Run batch sequential trainings via bash runner:
```bash
bash scripts/runner.sh
```

---

## 3. Codebase File Map & Architecture

When searching for or modifying functionality, refer to the following locations:

| Subsystem | Key Files | Description |
| :--- | :--- | :--- |
| **Model Registry** | `src/Models/ModelRegistry.py` | Auto-discovers all `nn.Module` classes in `src/Models/egamma/` by class name |
| **Model Architectures** | `src/Models/egamma/*.py` | Individual models (`ModelV1` through `ModelV6`, `Run2_ModelV1`, `Run2_ModelV1_2`) |
| **Model Factory** | `src/Models/Models.py` | `get_model(tag, input_dim)` delegating to `ModelRegistry` |
| **Interfaces** | `src/core/Interfaces/` | Abstract base classes (`BaseEgammaDataset`, `BaseTrainer`, `BaseResultAggregator`, `BaseEvaluator`, `BasePlotter`, `BaseMetricPlotter`) |
| **Dataset & Splits** | `src/core/Datasets/` | `EgammaNpzDataset.py` (reads `.npz` rings, exposes `feature_names`), `EgammaNpzDatasetNoTargetOrigin.py`, `SplitManifest.py` (K-fold split persistence with logging) |
| **Trainers** | `src/core/Trainers/` | `NeuralRingerTrainer.py` (coordinator), `FoldTrainer.py` (single fold), `TrainingFactory.py`, `ResultsRecorder.py` |
| **Callbacks** | `src/core/Callbacks/` | `SPCallbackPyTorch.py` (tracks SP index, knee point, early stopping, best weights) |
| **Plotting** | `src/core/Plotting/` | `PlotManager.py`, `BasePlotter.py`, `QuadrantPlotter.py` (2x2 heatmaps & score scatter), `ProfilePlotter.py`, `MetricPlotter.py`, `RegionDistributionPlotter.py` (2D heatmaps), `LegacyPlotter.py`, `Context.py` |
| **Validation** | `src/core/Validation/` | `ModelValidator.py` (global & regional evaluation), `QuadrantAnalyzer.py` (4-quadrant model comparison), `RegionDataDistributionAnalyzer.py`, `HoldoutEvaluator.py`, `EfficiencyPlotter.py`, `ResultAggregator.py` |
| **Configuration** | `src/Parser/` | `NeuralRingerTrainerConfiguration.py` (Pydantic schema), `DynamicConfiguration.py` |
| **Utilities** | `src/utils.py` | L1-norm, metric aggregation, bin extraction, dynamic instantiation |
| **Configs** | `config/NeuralRinger/` | YAML training specifications |
| **Notebooks** | `notebooks/` | Interactive analysis (`validate_results.ipynb`) |
| **Entrypoints** | `main.py`, `scripts/*.py` | CLI execution scripts (`validate_pd.py`, `plot_data_distribution.py`, `run_plots.py`, `run_roc_plots.py`, `calculate_photon_cut_efficiency.py`) |
| **Unit Tests** | `tests/` | Pytest test cases (`test_region_distribution_plotter.py`, `test_validation.py`, etc.) |

---

## 4. How to Add a New Model Architecture

1. Create a new file `src/Models/egamma/<ModelClassName>.py` (e.g., `ModelV7.py`).
2. Implement your class subclassing `torch.nn.Module` using clean OOP principles and type hints:
   ```python
   import torch
   import torch.nn as nn

   class ModelV7(nn.Module):
       def __init__(self, input_dim: int) -> None:
           super().__init__()
           self.fc = nn.Linear(input_dim, 1)

       def forward(self, x: torch.Tensor) -> torch.Tensor:
           return torch.sigmoid(self.fc(x))
   ```
3. `ModelRegistry` will automatically discover and register `ModelV7` under `"ModelV7"`.
4. No edits to `src/Models/Models.py` or `src/Models/egamma/__init__.py` are needed.
