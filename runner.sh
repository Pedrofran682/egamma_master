#!/usr/bin/env sh

# plot ringer 
# python run_plots.py results/modelV3.dim0.5.folds10_id20251117001936/ --plot_ringer --drive_path /eos/user/j/jlieberm/photonRinger/datasets/notIso --percentage 0.5

# plot ringer
# python run_plots.py results/modelV3.dim0.5.folds10_id20251117001936/ 


python main.py --percentage 0.5 --model_tag V2 --folder_path results/modelV2.dim0.5.folds10_id20251120131323

