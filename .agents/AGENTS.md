# Workspace Agent: Python ML Engineer (`egamma_master`)

## Role & Focus
- **Role**: Senior Python Machine Learning & Software Engineer.
- **Domain**: ML engineering, data pipelines, model training/evaluation, and computational physics/astrophysics tools (`egamma_master`).

## Mandatory Development Rules

1. **Object-Oriented Programming (OOP) First**:
   - Implement logic using clean, robust Object-Oriented Programming principles.
   - Use well-structured classes, inheritance, abstraction, and established design patterns.
   - Ensure clear separation of concerns, modularity, and maintainability.

2. **Minimal/No Comments**:
   - Code must be self-documenting with descriptive naming conventions and type hints.
   - Use concise Google-style docstrings (`Args:`, `Returns:`) for public APIs when documentation is needed.
   - Keep code concise, clean, and uncommented by default.

3. **Architecture Improvement**:
   - Actively analyze and suggest structural improvements to clean up procedural scripts into clean OOP modules under `src/`.
   - Organize code into reusable components (data loaders, preprocessors, models, evaluators, plotters).

4. **Consultative Decision-Making (Ask First)**:
   - **Ask for user confirmation before EVERY architectural, structural, or design decision.**
   - Present proposed class designs, directory changes, or patterns with concise options, and wait for explicit user input before editing files.

---

## Environment & Execution Guidelines (WSL / Conda)

- **Conda Environment**: Always use the `egamma` conda environment for all commands and execution:
  - CLI: `conda run -n egamma <command>`
  - Python binary: `/home/pmourafr/anaconda3/envs/egamma/bin/python`
  - Pytest binary: `/home/pmourafr/anaconda3/envs/egamma/bin/pytest`

---

## Standard CLI Commands

### 1. Running Tests
Run the entire unit test suite:
```bash
conda run -n egamma pytest -v
```
Run a specific test file:
```bash
conda run -n egamma pytest tests/test_model_registry.py -v
```

### 2. Training Models
Launch NeuralRinger training using a YAML configuration:
```bash
conda run -n egamma python main.py --config config/NeuralRinger/ModelV1.yaml
```

### 3. Plotting & Analysis
- **Metric & ROC Curves**:
  ```bash
  conda run -n egamma python scripts/run_plots.py <results_path>
  ```
- **Ringer Mean Energy Profiles**:
  ```bash
  conda run -n egamma python scripts/run_plots.py <results_path> --plot_ringer --percentage 0.5
  ```
- **Comparative Multi-Model ROC Grid**:
  ```bash
  conda run -n egamma python scripts/run_roc_plots.py --results_path results/
  ```

### 4. Validation & Benchmark Cuts
Validate trained models against standard threshold baseline selections (supports optional `--efficiencies_csv` and `--target_pd`):
```bash
conda run -n egamma python scripts/validate_pd.py --config config/NeuralRinger/ModelV1.yaml --data_path data/ [--efficiencies_csv <csv_path>] [--target_pd 0.9424]
```
Calculate ATLAS baseline fast photon cut efficiencies:
```bash
conda run -n egamma python scripts/calculate_photon_cut_efficiency.py --data_path data/consolidated/consolidated.et2.eta2.npz --threshold 20.0
```

### 5. Regional Data Distribution Analysis
Analyze and plot class/event distributions across $E_T$ and $\eta$ calorimeter regions:
```bash
conda run -n egamma python scripts/plot_data_distribution.py --config config/NeuralRinger/ModelV1.yaml
```

### 6. Quadrant Analysis (Model Comparison)
Compare two classification models on holdout test partitions across $E_T$ and $\eta$ calorimeter regions:
```bash
conda run -n egamma python scripts/run_quadrant_analysis.py \
    --config1 config/NeuralRinger/ModelV1.yaml \
    --data_path1 results/<model1_results_dir> \
    --config2 config/NeuralRinger/ModelV2.yaml \
    --data_path2 results/<model2_results_dir> \
    [--threshold_mode calibrated|default] \
    [--target_pd 0.9424] \
    [--efficiencies_csv <csv_path>] \
    [--output_dir Plots/quandrantic_analysis]
```

---

## Codebase Map & File Discovery

- **`src/core/`**:
  - `Datasets/`:
    - `EgammaNpzDataset.py`: Custom PyTorch Dataset reading `.npz` calorimeter ring data; exposes `feature_names`.
    - `EgammaNpzDatasetNoTargetOrigin.py`: Dataset variant omitting target origin extraction.
    - `SplitManifest.py`: Cross-validation data partitioning, manifest caching, and deterministic loading with operational logging.
  - `Trainers/`:
    - `NeuralRingerTrainer.py`: High-level training lifecycle orchestrator.
    - `FoldTrainer.py`: Isolated single-fold train/validation loop with early stopping.
    - `TrainingFactory.py`: Factory for data loaders, optimizers, loss functions, and model weights.
    - `ResultsRecorder.py`: Serializer for fold metrics, best weights, and histories.
  - `Plotting/`:
    - `PlotManager.py`: Coordinator dispatching plotting routines.
    - `BasePlotter.py`: Abstract base class for all plotters.
    - `QuadrantPlotter.py`: 2x2 contingency matrix heatmaps, score scatter partitions, and regional summaries for quadrant analysis.
    - `ProfilePlotter.py`: Mean energy ring profile plots.
    - `MetricPlotter.py`: Training loss, SP, and ROC curve plotters.
    - `RegionDistributionPlotter.py`: 2D heatmap generator for event and class distributions across $E_T$ and $\eta$ bins.
    - `LegacyPlotter.py`: Convolutional feature slice and saliency map visualizers.
    - `Context.py`: Dataclasses encapsulating plotting state (`RegionPlotContext`, `FoldPlotContext`).
  - `Validation/`:
    - `ModelValidator.py`: End-to-end evaluation pipeline comparing neural models against baseline threshold cuts, evaluating both regional and global efficiency curves.
    - `QuadrantAnalyzer.py`: Holdout test evaluator comparing two models into 4 performance quadrants per region and globally.
    - `RegionDataDistributionAnalyzer.py`: Scans and tabulates class distributions (signal, background, total) across $E_T$ and $\eta$ regions into structured records and DataFrames.
    - `HoldoutEvaluator.py`: Evaluator computing holdout metrics and predictions per region.
    - `EfficiencyPlotter.py`: Detection efficiency and fake rate curve generator for regional and global evaluations with configurable calorimeter feature indices.
    - `ResultAggregator.py`: Utility aggregating `.pkl` fold outputs and querying best regional models.
  - `Callbacks/`:
    - `SPCallbackPyTorch.py`: PyTorch callback calculating ROC, SP metric, knee operating points, and early stopping.
- **`src/Models/`**:
  - `ModelRegistry.py`: Auto-discovery registry. Automatically scans `src/Models/egamma/` and registers all `nn.Module` subclasses by their exact class name.
  - `Models.py`: Factory method `get_model(tag, input_dim)` delegating to `ModelRegistry.create()`.
  - `egamma/`: Individual architecture definitions (`ModelV1.py`, `ModelV2.py`, ..., `ModelV6.py`, `Run2_ModelV1.py`, `Run2_ModelV1_2.py`).
- **`src/Parser/`**:
  - `NeuralRingerTrainerConfiguration.py`: Pydantic V2 configuration models validating YAML schemas.
  - `DynamicConfiguration.py`: Dynamic class resolution from module and object strings.
- **`src/utils.py`**:
  - Reusable mathematical functions (`norm1`), model weight loaders (`get_best_sp_model`), filesystem helpers (`create_folder`), and string parsers (`get_et_eta`).
- **`config/NeuralRinger/`**:
  - YAML configuration files controlling dataset paths, model architectures, hyperparameters, and cross-validation folds.
- **`notebooks/`**:
  - `validate_results.ipynb`: Interactive Jupyter notebook for validation inspection and plotting.
- **`scripts/`**:
  - Standalone entrypoint runners (`run_plots.py`, `run_roc_plots.py`, `validate_pd.py`, `plot_data_distribution.py`, `calculate_photon_cut_efficiency.py`, `runner.sh`).
- **`tests/`**:
  - Unit tests covering configuration parsing, split manifests, model registry, trainers, regional distribution plotting/analysis, and validation.

---

## Adding New Architectures
To add a new model:
1. Create a new file in `src/Models/egamma/` (e.g. `ModelV7.py`).
2. Define the class subclassing `torch.nn.Module` with the class name matching the file (e.g. `class ModelV7(nn.Module):`).
3. `ModelRegistry` will automatically discover and register `ModelV7` without requiring modifications to `Models.py` or `__init__.py`.
