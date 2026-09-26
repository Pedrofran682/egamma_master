# egamma_master: Machine Learning Ringer Architecture for Electron/Gamma Identification

## Overview
`egamma_master` is a computational astrophysics and high-energy physics machine learning framework designed for training, validating, and evaluating Neural Ringer models for electron/gamma identification and energy ring classification.

## Project Architecture

```
egamma_master/
├── config/                  # YAML configurations for Neural Ringer model training runs
│   └── NeuralRinger/        # Model configuration files (e.g. ModelV1.yaml, ModelV5.yaml)
├── src/                     # Core Object-Oriented source code
│   ├── core/
│   │   ├── Callbacks/       # Custom PyTorch training callbacks (e.g. SPCallbackPyTorch)
│   │   ├── Datasets/        # PyTorch / NumPy dataset wrappers & SplitManifest
│   │   ├── Trainers/        # Modular training components (Factory, FoldTrainer, ResultsRecorder)
│   │   ├── Plotting/        # Modular plotting engine (PlotManager, MetricPlotter, ProfilePlotter)
│   │   └── Validation/      # Holdout evaluation & model validation pipeline
│   ├── Models/              # Neural network architecture definitions
│   │   ├── Models.py        # PyTorch model implementations
│   │   └── egamma/          # Specific egamma model variants
│   ├── Parser/              # Pydantic configuration schemas and dynamic parsing models
│   └── utils.py             # Reusable helper functions
├── scripts/                 # CLI entrypoint scripts (run_plots, run_roc_plots, validate_pd, runner.sh)
├── tests/                   # Pytest OOP unit test suite
├── notebooks/               # Jupyter exploration and analysis notebooks
├── main.py                  # Primary training execution entrypoint
├── pytest.ini               # Pytest configuration file
└── environment.yml          # Conda environment dependency definitions
```

## Key Components

### 1. Model Training Orchestration (`src/core/Trainers/`)
- `NeuralRingerTrainer`: High-level orchestrator coordinating regions, folds, and repeats.
- `TrainingFactory`: Decouples creation of PyTorch models, optimizers, loss functions, and DataLoaders.
- `FoldTrainer`: Encapsulates epoch training, validation steps, and callback monitoring.
- `ResultsRecorder`: Manages metric accumulation, history tracking, and disk serialization.

### 2. Dataset & Split Management (`src/core/Datasets/`)
- `EgammaNpzDataset`: Reads particle physics `.npz` binary data, extracts calorimeter rings, and handles event filtering.
- `SplitManifest`: Manages persistent JSON manifests (`split_manifest.json`) for deterministic train/val/test partitions without relying on random seeds.

### 3. Plotting & Evaluation Engine (`src/core/Plotting/`)
- `PlotManager`: Orchestrates active visualizers and metric plotters.
- `ProfileMeanEnergyPlotter`: Generates average ring energy profile curves.
- `ModelMetricsPlotter` & `RocPlotter`: Produces loss/accuracy curves, ROC curves, and performance metrics.
- `LegacyPlotter`: Keeps `ConvLayerPlotter` and `SaliencyMapPlotter` code archived in the codebase without active pipeline execution.

### 4. Validation & Holdout Pipeline (`src/core/Validation/`)
- `ModelValidator`: Main pipeline orchestrator selecting top-performing models and coordinating holdout evaluations.
- `HoldoutEvaluator`: Computes accuracy, TPR, TNR, FPR, FNR, and establishes target $P_D$ cut thresholds.
- `ResultAggregator`: Discovers and groups fold/repeat statistics from serialized result archives.
- `EfficiencyPlotter`: Plots binned signal detection efficiency curves across $E_T$ and $\eta$ distributions.

### 5. Dynamic Configuration (`src/Parser/`)
- Type-validated YAML configuration deserialization via Pydantic (`NeuralRingerTrainerConfiguration`).

## Development Guidelines & Refactoring Focus
- **OOP First**: Pure Object-Oriented design across trainers, datasets, evaluators, and models.
- **Minimal Commentary**: Self-documenting, concise Python code.
- **Modularity**: Clean separation between data loading, model definition, training loop, and evaluation.

## Running Tests
Run all unit tests with `pytest`:
```bash
conda activate egamma
pytest -v
```
Or run directly using standard library:
```bash
python -m unittest discover -s tests -v
```

