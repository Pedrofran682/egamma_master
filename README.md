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
├── scripts/                 # CLI entrypoints (dataGen, merge_regions_data, run_plots, run_roc_plots, validate_pd, runner.sh)
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
- `RootDatasetGenerator`: Extracts, vectorizes, and partitions calorimeter events from ROOT trees into binned `.npz` files in a single vectorized pass with multi-threaded branch decompression.
- `RegionDataConsolidator`: Merges partitioned `.npz` dataset files into consolidated kinematic region files with multi-core parallel processing.
- `EgammaNpzDataset`: Reads particle physics `.npz` binary data, extracts calorimeter rings, and handles event filtering.
- `SplitManifest`: Manages persistent JSON manifests (`split_manifest.json`) for deterministic train/val/test partitions without relying on random seeds.

### 3. Plotting & Evaluation Engine (`src/core/Plotting/`)
- `PlotManager`: Orchestrates active visualizers and metric plotters.
- `ProfileMeanEnergyPlotter`: Generates average ring energy profile curves.
- `ModelMetricsPlotter` & `RocPlotter`: Produces loss/accuracy curves, ROC curves, and performance metrics.
- `RegionDistributionPlotter`: Generates 2D heatmap grids and multi-panel figures of event distributions across $E_T$ and $\eta$ regions.
- `QuadrantPlotter`: Renders 2x2 contingency matrix heatmaps, score scatter partitions, and regional summaries for model comparison.
- `LegacyPlotter`: Keeps `ConvLayerPlotter` and `SaliencyMapPlotter` code archived in the codebase without active pipeline execution.

### 4. Validation & Holdout Pipeline (`src/core/Validation/`)
- `ModelValidator`: Main pipeline orchestrator selecting top-performing models and coordinating holdout evaluations.
- `QuadrantAnalyzer`: Holdout test evaluator comparing two models into 4 performance quadrants per region and globally with McNemar significance testing.
- `RegionDataDistributionAnalyzer`: Summarizes regional event counts by class (background, signal, total events, signal fraction) and coordinates heatmap exports.
- `FastPhotonCutEvaluator`: Evaluates baseline detection efficiency ($P_D$) using ATLAS Athena `TrigFastPhotonCutMaps` (`loose`, `medium`, `tight`, `etcut`).
- `HoldoutEvaluator`: Computes accuracy, TPR, TNR, FPR, FNR, and establishes target $P_D$ cut thresholds.
- `ResultAggregator`: Discovers and groups fold/repeat statistics from serialized result archives.
- `EfficiencyPlotter`: Plots binned signal detection efficiency curves across $E_T$ and $\eta$ distributions.

### 5. Dynamic Configuration (`src/Parser/`)
- Type-validated YAML configuration deserialization via Pydantic (`NeuralRingerTrainerConfiguration`).

## Data Preparation Pipeline

The framework provides an end-to-end data pipeline for converting raw CERN ROOT ntuples into consolidated kinematic NPZ datasets.

### 1. Extracting & Partitioning ROOT Files (`scripts/dataGen.py`)
Processes raw ROOT files, extracts calorimeter ring features and kinematic variables, vectorizes events into 2D NumPy arrays in a single batch-level pass, and partitions events into kinematic $(E_T, |\eta|)$ regions.

```bash
# Run extraction across default sample directories using 4 worker threads
conda run -n egamma python scripts/dataGen.py --workers 4

# Run with custom input and output locations
conda run -n egamma python scripts/dataGen.py \
  --base_dir /eos/user/p/pmourafr/RootFiles \
  --output_dir ./data \
  --splits 2 \
  --workers 8
```

**Key CLI Options**:
- `--base_dir <path>`: Base directory containing ROOT sample directories.
- `--output_dir <path>`: Destination directory where partitioned NPZ folders are created.
- `--tree_name <path>`: Path to TTree inside ROOT files (default: `run_450000/HLT/EgammaMon/summary/events`).
- `--vector_col <name>`: Branch name containing ring vector data (default: `trig_L2_calo_rings`).
- `--splits <int>`: Number of batches to split input files into (default: `2`).
- `--workers <int>`: Parallel threads for `uproot` branch decompression (default: `4`).
- `--compress`: Save output archives using compression (default: uncompressed).

### 2. Merging Partitions into Consolidated Datasets (`scripts/merge_regions_data.py`)
Discovers matching partition files across sample directories, extracts `data`, `target`, `source`, and `feature` arrays in a single pass, and writes consolidated region archives `consolidated.et{et}.eta{eta}.npz`.

```bash
# Run consolidation with multi-core parallel processing
conda run -n egamma python scripts/merge_regions_data.py \
  --input_dir ./data \
  --output_dir ./data/consolidated

# Run specifying worker processes and uncompressed output for maximum write speed
conda run -n egamma python scripts/merge_regions_data.py \
  --input_dir ./data \
  --output_dir ./data/consolidated \
  --workers 4 \
  --no_compress
```

**Key CLI Options**:
- `--input_dir, -i <path>`: Source directory containing partitioned NPZ files (default: `./data`).
- `--output_dir, -o <path>`: Destination directory for consolidated files (default: `./data/consolidated`).
- `--workers, -w <int>`: Number of parallel worker processes (default: all available CPU cores).
- `--no_compress`: Save uncompressed NPZ archives instead of `np.savez_compressed`.
- `--pattern <regex>`: Regex pattern matching `et` and `eta` indices from filenames.
- `--ignore_pattern <regex>`: Regex pattern identifying files to ignore (default: `\.sys\.v\d+`).

## Model Training Workflow

The Neural Ringer training pipeline is modular, fully declarative via YAML configs, and orchestrated by `NeuralRingerTrainer`.

### 1. Training Configuration (`config/NeuralRinger/`)
Each training run is defined in a YAML configuration file. Key parameters include:
- `model`: Model architecture name under `src/Models/egamma/` (e.g. `ModelV1`, `ModelV5`, `ModelV6`) and `input_dim` (100, 50, or 25 rings).
- `dataset`: Dataset loader (`EgammaNpzDataset`), path to consolidated `.npz` files, and ring percentage (`1.0`, `0.5`, `0.25`).
- `et_range_idx` & `eta_range_idx`: Kinematic bins to train on (e.g., `[0, 1, 2, 3, 4]`).
- `kFold`: Stratified K-Fold cross-validation configuration (default: 10 folds).
- `epochs`, `batch_size`, `n_initializations`: Number of training epochs, batch size, and weight initializations per fold.
- `optimizer_function` & `loss_function`: Learning rate, optimizer parameters (e.g., Adam), and loss formulation.

### 2. Launching Training
Execute model training via `main.py`:

```bash
conda run -n egamma python main.py --config config/NeuralRinger/ModelV5_HighBatch_newExtraction_20_regions_100Rings.yaml
```

**Training Execution Lifecycle**:
1. **Config Validation**: Validates the YAML file against the Pydantic `NeuralRingerTrainerConfiguration` schema.
2. **Deterministic Data Partitioning**: `SplitManifest` reads or creates deterministic K-Fold cross-validation splits persisted in `split_manifest.json`.
3. **Cross-Validation Training**: For each kinematic region, fold, and repeat initialization:
   - Data is loaded and ring features are normalized ($L_1$-norm) via `EgammaNpzDataset`.
   - `FoldTrainer` executes the training loop with early stopping.
   - `SPCallbackPyTorch` evaluates the ROC curve, tracks the $SP$ metric ($SP = \sqrt{\sqrt{P_D \times (1 - P_F)} \times \frac{P_D + (1 - P_F)}{2}}$), and computes the optimal operating knee point.
4. **Serialization & Checkpoints**: `ResultsRecorder` writes model weights, metrics, and training histories:
   - Checkpoints & results: `results/<config_name>_id<timestamp>/`
   - Detailed execution log: `log/TrainerRunner_<timestamp>.log`

### 3. Post-Training Evaluation & Visualization

After training completes, analyze performance and generate plots:

- **Loss Curves, ROC Curves, and Metric Histories**:
  ```bash
  conda run -n egamma python scripts/run_plots.py results/<run_folder>/
  ```

- **Calorimeter Ringer Mean Energy Profiles**:
  ```bash
  conda run -n egamma python scripts/run_plots.py results/<run_folder>/ --plot_ringer --percentage 0.5
  ```

- **Comparative Multi-Model ROC Grid**:
  ```bash
  conda run -n egamma python scripts/run_roc_plots.py --results_path results/
  ```

- **Validation Against Benchmark Threshold Cuts**:
  ```bash
  conda run -n egamma python scripts/validate_pd.py \
    --config config/NeuralRinger/ModelV5_HighBatch_newExtraction_20_regions_100Rings.yaml \
    --data_path results/<run_folder>/
  ```
  *(Plots default to `Plots/<yaml_name>/` with dedicated subfolders for `ROC/`, `RingsMeanProfiles/`, and `Validation/` containing `signal_efficiency_vs_*` curves)*

- **Regional Class Distribution Heatmaps (Signal vs. Background)**:
  ```bash
  conda run -n egamma python scripts/plot_data_distribution.py \
    --config config/NeuralRinger/ModelV6_HighBatch_newExtraction_20_regions_100RingsNoFilter.yaml \
    --data_path data/consolidated \
    --format png
  ```
  *(Generates 2D heatmap grids for background counts, signal counts, total events, and signal fraction in `Plots/<config_name>/DataDistribution/` alongside `region_data_distribution.csv`)*

- **Comparative Quadrant Analysis (Event-by-Event Model Comparison)**:
  ```bash
  # Direct python execution (with conda activate egamma):
  python scripts/run_quadrant_analysis.py \
    --config1 config/NeuralRinger/ModelV5_HighBatch_newExtraction_20_regions_100Rings.yaml \
    --data_path1 results/<model1_results_folder>/ \
    --config2 config/NeuralRinger/ModelV6_HighBatch_newExtraction_20_regions_100Rings.yaml \
    --data_path2 results/<model2_results_folder>/

  # Or using conda run:
  conda run -n egamma python scripts/run_quadrant_analysis.py \
    --config1 config/NeuralRinger/ModelV5_HighBatch_newExtraction_20_regions_100Rings.yaml \
    --data_path1 results/<model1_results_folder>/ \
    --config2 config/NeuralRinger/ModelV6_HighBatch_newExtraction_20_regions_100Rings.yaml \
    --data_path2 results/<model2_results_folder>/
  ```
  *(Generates 2x2 contingency matrix heatmaps, score scatter partitions, and McNemar test summaries under `Plots/quandrantic_analysis/<model1>_<rings1>rings_vs_<model2>_<rings2>rings/`)*

### 4. Batch Training Automation
For running multiple sequential training runs, configure and execute `scripts/runner.sh`:
```bash
bash scripts/runner.sh
```

## Regional Class Distribution Analysis

Analyze and visualize event counts and class balance (Signal vs. Background) across all $(E_T, \eta)$ kinematic regions using `scripts/plot_data_distribution.py`. The tool performs fast event filtering without calorimeter ring normalization overhead, exports a summary CSV, and renders 2D heatmap grids:

```bash
# Generate 2D heatmaps using YAML configuration and consolidated NPZ directory
conda run -n egamma python scripts/plot_data_distribution.py \
  --config config/NeuralRinger/ModelV6_HighBatch_newExtraction_20_regions_100RingsNoFilter.yaml \
  --data_path data/consolidated \
  --format png
```

### Generated Visualizations & Outputs
- **Composite 2D Grid (`data_distribution_2d_grid.png`)**: 2x2 multi-panel figure displaying Background counts, Signal counts, Total events, and Signal fraction ($S / \text{Total}$) across $E_T \times \eta$.
- **Individual Metric Heatmaps**:
  - `distribution_background_count.png`: Background (Class 0) counts.
  - `distribution_signal_count.png`: Signal (Class 1) counts.
  - `distribution_total_events.png`: Total events per region.
  - `distribution_signal_fraction.png`: Signal fraction percentage ($S / \text{Total}$).
- **Summary Metrics CSV (`region_data_distribution.csv`)**: Tabular export containing columns `et`, `eta`, `background_count`, `signal_count`, `total_events`, `signal_fraction`, `file_path`.

### CLI Options
- `--config <path>`: Path to YAML training configuration (required).
- `--data_path <path>`: Optional path to directory containing consolidated `.npz` files (overrides config `drive_path`).
- `--output_dir <path>`: Custom destination path for figures and CSV (default: `Plots/<config_name>/DataDistribution/`).
- `--format {pdf,png}`: Image format for generated figures (default: `pdf`).

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
- `--output_csv <path>`: Output CSV path for region efficiencies compatible with `ModelValidator` (default: `efficiencies_by_region.csv`).

## Quadrant Analysis (Model Comparison)

Compare two trained neural classification models event-by-event on holdout test partitions across all $(E_T, \eta)$ kinematic regions and globally using `scripts/run_quadrant_analysis.py`.

The analysis evaluates model predictions on test sets, computes optimal decision thresholds, and groups events into four performance quadrants:
1. **Both Correct**: Events where both models predicted the correct ground-truth class.
2. **Model 1 Only Correct**: Events where Model 1 was correct and Model 2 misclassified.
3. **Model 2 Only Correct**: Events where Model 2 was correct and Model 1 misclassified.
4. **Both Wrong**: Events where both models misclassified.

Statistical discordance between models is quantified via **McNemar's test** (exact two-sided binomial test for discordant counts $< 25$, continuity-corrected $\chi^2$ test otherwise) to identify statistically significant advantages.

### 1. Basic Execution (Calibrated Thresholds)
Compare two models using calibrated decision thresholds derived from the target signal efficiency:

```bash
# Direct python execution (with conda activate egamma):
python scripts/run_quadrant_analysis.py \
  --config1 config/NeuralRinger/ModelV5_HighBatch_newExtraction_20_regions_100Rings.yaml \
  --data_path1 results/ModelV5_run/ \
  --config2 config/NeuralRinger/ModelV6_HighBatch_newExtraction_20_regions_100Rings.yaml \
  --data_path2 results/ModelV6_run/ \
  --file_format png
```

### 2. Using ATLAS Reference Efficiencies CSV
Provide a reference operating point table so each kinematic region uses its exact baseline target $P_D$:

```bash
python scripts/run_quadrant_analysis.py \
  --config1 config/NeuralRinger/ModelV5_HighBatch_newExtraction_20_regions_100Rings.yaml \
  --data_path1 results/ModelV5_run/ \
  --config2 config/NeuralRinger/ModelV6_HighBatch_newExtraction_20_regions_100Rings.yaml \
  --data_path2 results/ModelV6_run/ \
  --threshold_mode calibrated \
  --efficiencies_csv efficiencies_by_region.csv \
  --file_format pdf
```

### 3. Comparing Models at Default 0.5 Decision Boundary
```bash
python scripts/run_quadrant_analysis.py \
  --config1 config/NeuralRinger/ModelV5_HighBatch_newExtraction_20_regions_100Rings.yaml \
  --data_path1 results/ModelV5_run/ \
  --config2 config/NeuralRinger/ModelV6_HighBatch_newExtraction_20_regions_100Rings.yaml \
  --data_path2 results/ModelV6_run/ \
  --threshold_mode default
```

### Generated Visualizations & Output Directory
Outputs are automatically placed inside a dedicated model comparison folder under the base plots directory:
`Plots/quandrantic_analysis/<model1>_<rings1>rings_vs_<model2>_<rings2>rings/` (e.g. `Plots/quandrantic_analysis/ModelV5_100rings_vs_ModelV6_100rings/`):

- **`MatrixHeatmaps/`**: 3-panel 2x2 contingency matrix heatmaps (`iet{et}_ieta{eta}_quadrant_matrix.{format}` and `global_quadrant_matrix.{format}`) displaying All Events, Signal (Class 1), and Background (Class 0) counts, ratios, and McNemar p-values.
- **`ScoreScatter/`**: Predicted probability scatter plots comparing Model 1 vs. Model 2 with decision threshold boundary lines (`iet{et}_ieta{eta}_score_scatter.{format}` and `global_score_scatter.{format}`).
- **`Summary/`**: Summary comparisons across kinematic bins.

### CLI Options
- `--config1 <path>`: Path to Model 1 configuration YAML (required).
- `--data_path1 <path>`: Path to Model 1 results directory with `.pkl` files and `split_manifest.json` (required).
- `--config2 <path>`: Path to Model 2 configuration YAML (optional, defaults to `--config1`).
- `--data_path2 <path>`: Path to Model 2 results directory with `.pkl` files (required).
- `--manifest_path <path>`: Optional explicit path to split manifest JSON (defaults to `<data_path1>/split_manifest.json`).
- `--threshold_mode {calibrated,default}`: Decision threshold mode (`calibrated` for target $P_D$ cut, `default` for 0.5 cut; default: `calibrated`).
- `--target_pd <float>`: Target signal detection efficiency (default: `0.9424`).
- `--efficiencies_csv <path>`: Optional path to reference operating point CSV table.
- `--output_dir <path>`: Base output directory (the comparison subfolder is automatically appended inside; default: `Plots/quandrantic_analysis`).
- `--file_format {pdf,png}`: Output graphic format (default: `pdf`).

## Development Guidelines & Refactoring Focus
- **OOP First**: Pure Object-Oriented design across trainers, datasets, evaluators, and models.
- **Minimal Commentary**: Self-documenting, concise Python code.
- **Modularity**: Clean separation between data loading, model definition, training loop, and evaluation.

## Running Tests
Run all unit tests with `pytest`:
```bash
conda run -n egamma pytest -v
```
Run the Regional Distribution Plotter test suite specifically:
```bash
conda run -n egamma pytest tests/test_region_distribution_plotter.py -v
```
Run the Fast Photon Cut Evaluator test suite specifically:
```bash
conda run -n egamma pytest tests/test_fast_photon_cut_evaluator.py -v
```
Run the Quadrant Analysis test suite specifically:
```bash
conda run -n egamma pytest tests/core/validation/test_quadrant_analysis.py -v
```
Run the Data Preparation test suites:
```bash
conda run -n egamma pytest tests/test_root_dataset_generator.py -v
conda run -n egamma pytest tests/test_region_data_consolidator.py -v
```
