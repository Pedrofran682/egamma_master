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
- `FastPhotonCutEvaluator`: Evaluates baseline detection efficiency ($P_D$) using ATLAS Athena `TrigFastPhotonCutMaps` (`loose`, `medium`, `tight`, `etcut`).
- `HoldoutEvaluator`: Computes accuracy, TPR, TNR, FPR, FNR, and establishes target $P_D$ cut thresholds.
- `ResultAggregator`: Discovers and groups fold/repeat statistics from serialized result archives.
- `EfficiencyPlotter`: Plots binned signal detection efficiency curves across $E_T$ and $\eta$ distributions.

### 5. Dynamic Configuration (`src/Parser/`)
- Type-validated YAML configuration deserialization via Pydantic (`NeuralRingerTrainerConfiguration`).

## Fast Photon Trigger Cut Efficiency Calculation

Calculate baseline trigger cut efficiencies across your custom $(E_T, |\eta|)$ kinematic regions using `scripts/calculate_photon_cut_efficiency.py`. The calculation does not rely on truth labels or `target` columns, evaluating pure calorimeter cut selections from ATLAS Athena `TrigFastPhotonCutMaps`. To avoid memory exhaustion, files are processed sequentially with a memory-efficient accumulator.

### Kinematic Granularity
- **$E_T$ intervals (GeV)**: `[15, 20, 30, 40, 50, inf]` (bins: `et0` to `et4`)
- **$|\eta|$ intervals**: `[0, 0.8, 1.37, 1.54, 2.37, 2.50, inf]` (bins: `eta0` to `eta5`)

For each event in an $E_T$ bin, the corresponding lower-bound ATLAS threshold map (15, 20, 30, 40, 40 GeV) is automatically selected to evaluate $R_{\text{core}} = E^{3\times 7}/E^{7\times 7}$ and $\text{HadEmRatio} = E_T^{\text{had}}/E_T^{\text{EM}}$ across detector $|\eta|$.

### 1. Evaluating All Region Files (Streaming Accumulator)
```bash
conda run -n egamma python scripts/calculate_photon_cut_efficiency.py \
  --data_path data/consolidated/ \
  --pattern "*.npz" \
  --output_csv efficiencies_by_region.csv
```

### 2. Evaluating a Single Region File
```bash
conda run -n egamma python scripts/calculate_photon_cut_efficiency.py \
  --data_path data/consolidated/consolidated.et2.eta2.npz \
  --output_csv efficiencies_by_region.csv
```

### 3. Available CLI Options
- `--data_path <path>`: Path to a single `.npz` file or a directory of files.
- `--pattern <glob>`: File glob pattern when `--data_path` is a directory (default: `*.npz`).
- `--working_points loose medium tight`: Space-separated working points to evaluate (default: `loose medium tight`).
- `--apply_et_cut`: Enforce minimum $E_T \ge (\text{threshold} - 3)\text{ GeV}$ cut (default: `False`).
- `--output_csv <path>`: Output CSV path for region efficiencies compatible with `ModelValidator` (default: `efficiencies_by_region.csv`).

## Development Guidelines & Refactoring Focus
- **OOP First**: Pure Object-Oriented design across trainers, datasets, evaluators, and models.
- **Minimal Commentary**: Self-documenting, concise Python code.
- **Modularity**: Clean separation between data loading, model definition, training loop, and evaluation.

## Running Tests
Run all unit tests with `pytest`:
```bash
conda run -n egamma pytest -v
```
Run the Fast Photon Cut Evaluator test suite specifically:
```bash
conda run -n egamma pytest tests/test_fast_photon_cut_evaluator.py -v
```

