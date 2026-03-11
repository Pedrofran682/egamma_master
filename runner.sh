#!/usr/bin/env sh

# plot ringer 
# python run_plots.py results/modelV3.dim0.5.folds10_id20251117001936/ --plot_ringer --drive_path /eos/user/j/jlieberm/photonRinger/datasets/notIso --percentage 1.0

# plot ringer
# python run_plots.py results/modelV3.dim0.5.folds10_id20251117001936/ 


# python main.py --config config/NeuralRinger/ModelV1.yaml
python main.py --config config/NeuralRinger/Run2_ModelV1.yaml

