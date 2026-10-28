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

5. **Conda Environment (`egamma`)**:
   - Always use the `egamma` Conda environment for all tool executions, Python scripts, CLI tasks, and tests.
   - Run commands using `conda run -n egamma <command>` or direct binaries `/home/pmourafr/anaconda3/envs/egamma/bin/python` and `/home/pmourafr/anaconda3/envs/egamma/bin/pytest`. Never execute commands using the system or base Python.

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
conda run -n egamma pytest tests/models/test_model_registry.py -v
```

### 2. Training Models
Launch NeuralRinger training using a YAML configuration and a split manifest (`split_manifest.json` under `./results` or `None` to auto-generate inside the run folder):
```bash
conda run -n egamma python main.py --config config/NeuralRinger/ModelV1.yaml --split_manifest split_manifest.json
```
Output folders in `results/` are timestamped automatically (`config_name<Name>_id<YYYYMMDDHHMMSS>`).

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
conda run -n egamma python scripts/calculate_photon_cut_efficiency.py --data_path data/consolidated/consolidated.et2.eta2.npz [--output_dir <dir>]
```

### 5. Regional Data Distribution Analysis
Analyze and plot class/event distributions across $E_T$ and $\eta$ calorimeter regions:
```bash
conda run -n egamma python scripts/plot_data_distribution.py --config config/NeuralRinger/ModelV1.yaml
```

### 6. Quadrant Analysis (Model Comparison & Cut-Based Benchmark)
Compare two classification models or a model against Athena fast photon cut-based trigger selection on holdout test partitions across $E_T$ and $\eta$ calorimeter regions:
```bash
# Model vs Model comparison
conda run -n egamma python scripts/run_quadrant_analysis.py \
    --mode model_vs_model \
    --config1 config/NeuralRinger/ModelV1.yaml \
    --data_path1 results/<model1_results_dir> \
    --config2 config/NeuralRinger/ModelV2.yaml \
    --data_path2 results/<model2_results_dir> \
    [--threshold_mode calibrated|default] \
    [--target_pd 0.9424] \
    [--efficiencies_csv <csv_path>] \
    [--output_dir Plots/quandrantic_analysis]

# Model vs Cut-Based Selection comparison
conda run -n egamma python scripts/run_quadrant_analysis.py \
    --mode model_vs_cut \
    --config1 config/NeuralRinger/ModelV1.yaml \
    --data_path1 results/<model1_results_dir> \
    [--working_point loose|medium|tight] \
    [--threshold_mode calibrated|default] \
    [--target_pd 0.9424] \
    [--efficiencies_csv <csv_path>] \
    [--output_dir Plots/quandrantic_analysis_cut]
```

### 7. Hyperparameter Tuning
Run grid search tuning over optimizers and hyperparameters for a target model configuration:
```bash
conda run -n egamma python scripts/run_hyperparameter_tuning.py \
    --tuning_config config/tuning/ModelV6_50Rings_region1_1_tuning.yaml \
    [--output_dir results/tuning_modelv6] \
    [--debug]
```

### 8. Pileup Sensitivity Analysis
Evaluate classifier stability, efficiency vs. $\langle\mu\rangle$, observable drifts, and $N_{\text{PV}}$ vertex calibration:
```bash
conda run -n egamma python scripts/run_pileup_analysis.py \
    --config config/NeuralRinger/ModelV6_HighBatch_newExtraction_20_regions_100Rings.yaml \
    --results_path results/<model_results_dir> \
    --target_et 1 --target_eta 1 \
    [--output_dir Plots/pileup_analysis] \
    [--format png|pdf]
```

---

## Codebase Map & File Discovery

- **`src/core/`**:
  - `Interfaces/`:
    - `BaseEgammaDataset.py`: Abstract base class for calorimeter NPZ datasets exposing template methods and `filter_events()`.
    - `BaseTrainer.py`: Abstract base class for trainers defining `device`, `full_dataset`, and `run()`.
    - `BaseResultAggregator.py`: Abstract base class for result aggregation and model discovery.
    - `BaseEvaluator.py`: Abstract base class for holdout evaluation.
    - `BasePlotter.py`: Abstract base classes `BasePlotter` and `BaseMetricPlotter`.
    - `BaseHyperparameterTuner.py`: Abstract base class for hyperparameter search orchestrators.
  - `Datasets/`:
    - `EgammaNpzDataset.py`: Custom PyTorch Dataset reading `.npz` calorimeter ring data; exposes `feature_names`.
    - `EgammaNpzDatasetNoTargetOrigin.py`: Dataset variant omitting target origin extraction.
    - `SplitManifest.py`: Cross-validation data partitioning, manifest caching, and deterministic loading with operational logging.
  - `Trainers/`:
    - `NeuralRingerTrainer.py`: High-level training lifecycle orchestrator.
    - `FoldTrainer.py`: Isolated single-fold train/validation loop with early stopping.
    - `TrainingFactory.py`: Factory for data loaders, optimizers, loss functions, and model weights.
    - `ResultsRecorder.py`: Serializer for fold metrics, best weights, and histories.
  - `Tuning/`:
    - `GridSearchTuner.py`: Orchestrator executing hyperparameter search trials and summarizing ranked outcomes.
    - `TuningConfiguration.py`: Pydantic schema validating tuning configuration YAML and building discrete trial configurations.
  - `Plotting/`:
    - `PlotManager.py`: Coordinator dispatching plotting routines.
    - `BasePlotter.py`: Abstract base class for all plotters.
    - `QuadrantPlotter.py`: 2x2 contingency matrix heatmaps, score scatter partitions, and regional summaries for quadrant analysis.
    - `FeatureDistributionPlotter.py`: 2-panel feature distributions with synchronized selection ratio subplots ($N_{\text{sel}} / N_{\text{true}}$) and overview grid generator.
    - `PileupPlotter.py`: 2-panel efficiency vs. $\langle\mu\rangle$ curves with ratio panel, shower shape & ring profile drift vs $\langle\mu\rangle$, and 2D $N_{\text{PV}}$ vs $\langle\mu\rangle$ density correlation heatmap.
    - `ProfilePlotter.py`: Mean energy ring profile plots.
    - `MetricPlotter.py`: Training loss, SP, and ROC curve plotters.
    - `RegionDistributionPlotter.py`: 2D heatmap generator for event and class distributions across $E_T$ and $\eta$ bins.
    - `LegacyPlotter.py`: Convolutional feature slice and saliency map visualizers.
    - `Context.py`: Dataclasses encapsulating plotting state (`RegionPlotContext`, `FoldPlotContext`).
  - `Validation/`:
    - `ModelValidator.py`: End-to-end evaluation pipeline comparing neural models against baseline threshold cuts, evaluating both regional and global efficiency curves.
    - `QuadrantAnalyzer.py`: Holdout test evaluator comparing two models into 4 performance quadrants per region and globally (supports `strategy1_alias` and `strategy2_alias`).
    - `DistributionComparator.py`: Statistical distance engine computing Kolmogorov-Smirnov ($D$), Wasserstein ($W_1$), Kullback-Leibler ($D_{\text{KL}}$), and Jensen-Shannon ($D_{\text{JS}}$) divergences with bootstrap uncertainties across True Data, Athena Cut, and Model vs Model baselines.
    - `PileupAnalyzer.py`: Statistical computation of detection efficiency & fake rate vs $\langle\mu\rangle$ with binomial uncertainties, observable mean drift ($\text{SEM}$), and primary vertex calibration fit ($N_{\text{PV}} = a\langle\mu\rangle + b$).
    - `RegionDataDistributionAnalyzer.py`: Scans and tabulates class distributions (signal, background, total) across $E_T$ and $\eta$ regions into structured records and DataFrames.
    - `HoldoutEvaluator.py`: Evaluator computing holdout metrics and predictions per region.
    - `EfficiencyPlotter.py`: Detection efficiency and fake rate curve generator for regional and global evaluations with configurable calorimeter feature indices.
    - `ResultAggregator.py`: Utility aggregating `.pkl` fold outputs and querying best regional models.
    - `QuadrantTableConsolidator.py`: Consolidates regional quadrant metrics (`SP`, `PF`, `PD` as percentages) into 2D $E_T \times |\eta|$ LaTeX grid tables and CSV exports.
  - `Callbacks/`:
    - `SPCallbackPyTorch.py`: PyTorch callback calculating ROC, SP metric, knee operating points, and early stopping.
- **`src/Models/`**:
  - `ModelRegistry.py`: Auto-discovery registry. Automatically scans `src/Models/egamma/` and registers all `nn.Module` subclasses by their exact class name.
  - `Models.py`: Factory method `get_model(tag, input_dim)` delegating to `ModelRegistry.create()`.
  - `egamma/`: Individual architecture definitions (`ModelV1.py`, `ModelV2.py`, ..., `ModelV10.py`, `Run2_ModelV1.py`, `Run2_ModelV1_2.py`).
- **`src/Parser/`**:
  - `NeuralRingerTrainerConfiguration.py`: Pydantic V2 configuration models validating YAML schemas.
  - `DynamicConfiguration.py`: Dynamic class resolution from module and object strings.
- **`src/utils.py`**:
  - Reusable mathematical functions (`norm1`), statistical distance metrics (`compute_ks_test_with_uncertainty`, `compute_wasserstein_with_uncertainty`, `compute_kl_divergence_with_uncertainty`, `compute_js_divergence_with_uncertainty`), feature column extractors (`extract_feature_values`), table serializers (`save_table`), model weight loaders (`get_best_sp_model`), filesystem helpers (`create_folder`), and string parsers (`get_et_eta`).
- **`config/NeuralRinger/`**:
  - YAML configuration files controlling dataset paths, model architectures, hyperparameters, and cross-validation folds.
- **`notebooks/`**:
  - `single_region_full_validation.ipynb`: Comprehensive single-region validation pipeline featuring operational response curves, quadrant analysis with alias support, 2-panel feature distribution ratios, multi-reference statistical distance tables, and pileup sensitivity suite.
  - `validate_results.ipynb`: Interactive Jupyter notebook for validation inspection and plotting.
  - `consolidate_quadrant_tables.ipynb`: Interactive Jupyter notebook consolidating quadrant metrics into 2D LaTeX grid tables.
- **`scripts/`**:
  - Standalone entrypoint runners (`run_plots.py`, `run_roc_plots.py`, `validate_pd.py`, `plot_data_distribution.py`, `calculate_photon_cut_efficiency.py`, `run_pileup_analysis.py`, `runner.sh`).
- **`tests/`**:
  - Organized to mirror `src/`:
    - `core/datasets/`: Manifest, consolidation, and ROOT generator tests.
    - `core/trainers/`: Training loop and factory tests.
    - `core/plotting/`: Metric, regional distribution, and quadrant plotting tests.
    - `core/validation/`: Benchmark cut, validation pipeline, and quadrant analysis tests.
    - `models/`: Auto-discovery and model registry tests.
    - `parser/`: Dynamic configuration and schema parsing tests.
    - `integration/`: End-to-end training pipeline, CLI entrypoint, and process lifecycle tests.

---

## Adding New Architectures
To add a new model:
1. Create a new file in `src/Models/egamma/` (e.g. `ModelV7.py`).
2. Define the class subclassing `torch.nn.Module` with the class name matching the file (e.g. `class ModelV7(nn.Module):`).
3. `ModelRegistry` will automatically discover and register `ModelV7` without requiring modifications to `Models.py` or `__init__.py`.
