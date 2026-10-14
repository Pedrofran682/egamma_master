#!/usr/bin/env sh

## Training
python main.py --config config/NeuralRinger/ModelV6_HighBatch_newExtraction_20_regions_100Rings.yaml --results_path results/config_nameModelV6_HighBatch_newExtraction_20_regions_100Rings_id20260930044931/

python main.py --config config/NeuralRinger/ModelV6_HighBatch_newExtraction_20_regions_50Rings.yaml --results_path results/config_nameModelV6_HighBatch_newExtraction_20_regions_50Rings_id20261005183226

# validate_pd
python scripts/validate_pd.py   --config config/NeuralRinger/ModelV6_HighBatch_newExtraction_20_regions_100Rings.yaml   --data_path results/config_nameModelV6_HighBatch_newExtraction_20_regions_100Rings_id20260930044931/

python scripts/run_plots.py results/results/config_nameModelV6_HighBatch_newExtraction_20_regions_50Rings_id20261005183226/


python scripts/plot_data_distribution.py \
  --config config/NeuralRinger/ModelV6_HighBatch_newExtraction_20_regions_100RingsNoFilter.yaml \
  --data_path data/consolidated \
  --format png
