#!/usr/bin/env sh

# plot ringer 
# python scripts/run_plots.py results/modelV3.dim0.5.folds10_id20251117001936/ --plot_ringer --drive_path /eos/user/j/jlieberm/photonRinger/datasets/notIso --percentage 1.0

################################ New data extraction

## 100 rings
# python scripts/run_plots.py results/config_nameModelV1_HighBatch_newExtraction_20_regions_100Rings_id20260629222107/ --plot_ringer --drive_path /eos/user/p/pmourafr/RootFiles/dataSetup/concatenated_results/ --percentage 1.0

## 50 rings
# python scripts/run_plots.py results/config_nameModelV1_HighBatch_newExtraction_20_regions_50Rings_id20260630043428/ --plot_ringer --drive_path /eos/user/p/pmourafr/RootFiles/dataSetup/concatenated_results/ --percentage 0.5

## 25 rings
# python scripts/run_plots.py results/config_nameModelV1_HighBatch_newExtraction_20_regions_25Rings_id20260630122538/ --plot_ringer --drive_path /eos/user/p/pmourafr/RootFiles/dataSetup/concatenated_results/ --percentage 0.25

# plot ringer
# python scripts/run_plots.py results/modelV3.dim0.5.folds10_id20251117001936/ 


# python main.py --config config/NeuralRinger/ModelV5_HighBatch_newExtraction_20_regions_100Rings.yaml
# python main.py --config config/NeuralRinger/ModelV5_HighBatch_newExtraction_20_regions_50Rings.yaml
# python main.py --config config/NeuralRinger/ModelV5_HighBatch_newExtraction_20_regions_25Rings.yaml

# python main.py --config config/NeuralRinger/ModelV1_HighBatch_newExtraction_20_regions_100Rings.yaml
# python main.py --config config/NeuralRinger/ModelV1_HighBatch_newExtraction_20_regions_50Rings.yaml
# python main.py --config config/NeuralRinger/ModelV1_HighBatch_newExtraction_20_regions_25Rings.yaml


# python main.py --config config/NeuralRinger/ModelV1_HighBatch_newExtraction_20_regions_50Rings_loose.yaml
# python main.py --config config/NeuralRinger/ModelV1_HighBatch_newExtraction_20_regions_50Rings_medium.yaml
# python main.py --config config/NeuralRinger/ModelV1_HighBatch_newExtraction_20_regions_50Rings_tight.yaml

# python scripts/validate_pd.py --config config/NeuralRinger/ModelV1_HighBatch_newExtraction_20_regions_25Rings.yaml  --data_path results/config_nameModelV1_HighBatch_newExtraction_20_regions_25Rings_id20260630122538
# python scripts/validate_pd.py --config config/NeuralRinger/ModelV1_HighBatch_newExtraction_20_regions_50Rings.yaml  --data_path results/config_nameModelV1_HighBatch_newExtraction_20_regions_50Rings_id20260630043428
# python scripts/validate_pd.py --config config/NeuralRinger/ModelV1_HighBatch_newExtraction_20_regions_100Rings.yaml  --data_path results/config_nameModelV1_HighBatch_newExtraction_20_regions_100Rings_id20260629222107