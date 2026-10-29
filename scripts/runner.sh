#!/usr/bin/env sh

# ## Training
# python main.py --config config/NeuralRinger/ModelV6_HighBatch_newExtraction_20_regions_100Rings.yaml --results_path results/config_nameModelV6_HighBatch_newExtraction_20_regions_100Rings_id20260930044931/

# python main.py --config config/NeuralRinger/ModelV6_HighBatch_newExtraction_20_regions_50Rings.yaml --results_path results/config_nameModelV6_HighBatch_newExtraction_20_regions_50Rings_id20261005183226

# # validate_pd
# python scripts/validate_pd.py   --config config/NeuralRinger/ModelV6_HighBatch_newExtraction_20_regions_100Rings.yaml   --data_path results/config_nameModelV6_HighBatch_newExtraction_20_regions_100Rings_id20260930044931/

# python scripts/run_plots.py results/results/config_nameModelV6_HighBatch_newExtraction_20_regions_50Rings_id20261005183226/


# python scripts/plot_data_distribution.py \
#   --config config/NeuralRinger/ModelV6_HighBatch_newExtraction_20_regions_100RingsNoFilter.yaml \
#   --data_path data/consolidated \
#   --format png



# python scripts/run_hyperparameter_tuning.py \
#   --tuning_config config/tuning/ModelV6_50Rings_region1_1_tuning.yaml \
#   --split_manifest results/config_nameModelV6_HighBatch_newExtraction_20_regions_50Rings_id20261005183226/split_manifest.json



# python main.py --config config/NeuralRinger/ModelV9_HighBatch_100Rings.yaml --split_manifest results/split_manifest.json
python main.py --config config/NeuralRinger/ModelV9_HighBatch_50Rings.yaml --split_manifest results/split_manifest.json

python main.py --config config/NeuralRinger/ModelV10_HighBatch_100Rings.yaml --split_manifest results/split_manifest.json
python main.py --config config/NeuralRinger/ModelV10_HighBatch_50Rings.yaml --split_manifest results/split_manifest.json
