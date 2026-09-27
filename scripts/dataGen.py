import os
import glob
import gc
import uproot
import awkward as ak
import numpy as np
import pandas as pd
from pathlib import Path


def process_root_files(input_dir,
                       output_dir,
                       tree_name,
                       vector_col,
                       eta_bins,
                       et_bins,
                       target_columns,
                       file_prefix,
                       num_splits=2):
    """
    Processes ROOT files in batches, applies eta/et and mc_type filters, and saves them.
    """

    os.makedirs(output_dir, exist_ok=True)
    data_output = Path(output_dir) / file_prefix
    os.makedirs(data_output, exist_ok=True)

    is_perf_jf = "perf_JF" in file_prefix

    all_files = sorted(glob.glob(f"{input_dir}/*.root"))
    if not all_files:
        print(f"Nenhum arquivo .root encontrado em {input_dir}")
        return

    file_batches = np.array_split(all_files, num_splits)

    for batch_idx, batch in enumerate(file_batches):
        if len(batch) == 0:
            continue

        print(f"\n--- Processando Metade {batch_idx + 1}/{num_splits} ---")
        file_paths = [f"{f}:{tree_name}" for f in batch]
        print(f"Carregando {len(file_paths)} arquivos...")

        events = uproot.concatenate(file_paths, filter_name=target_columns)

        features_list = []
        valid_fields = []
        for field in events.fields:
            if field == vector_col:
                for i in range(100):
                    features_list.append(f"{field}_{i}")
            else:
                features_list.append(field)
                valid_fields.append(field)

        for eta_idx in range(len(eta_bins) - 1):
            eta_min, eta_max = eta_bins[eta_idx], eta_bins[eta_idx + 1]

            for et_idx in range(len(et_bins) - 1):
                et_min, et_max = et_bins[et_idx], et_bins[et_idx + 1]

                mask = (abs(events['trig_L2_calo_eta']) >= eta_min) & (abs(events['trig_L2_calo_eta']) < eta_max) & \
                       (events['trig_L2_calo_et'] >= et_min) & (events['trig_L2_calo_et'] < et_max)

                events_filtered = events[mask]

                if len(events_filtered) == 0:
                    print(f"No events for: {et_min} <= et < {et_max} | {eta_min} <= eta < {eta_max}")
                    continue

                mc_types = ak.to_numpy(events_filtered['mc_type'])

                if is_perf_jf:
                    type_mask = ~np.isin(mc_types, [13, 14, 15])
                    target_val = 0
                else:
                    type_mask = np.isin(mc_types, [13, 14, 15])
                    target_val = 1

                events_filtered = events_filtered[type_mask]

                if len(events_filtered) == 0:
                    print(f"0 events after mc_type filter for: {et_min} <= et < {et_max} | {eta_min} <= eta < {eta_max}")
                    continue

                target_array = np.full(len(events_filtered), target_val, dtype=np.int32)
                source_array = np.full(len(events_filtered), file_prefix, dtype=object)

                padded_vector = ak.fill_none(ak.pad_none(events_filtered[vector_col], 100, axis=1), 0)[:, :100]

                columns_data = []
                for feat in features_list:
                    if feat.startswith(f"{vector_col}_"):
                        k = int(feat.split("_")[-1])
                        columns_data.append(ak.to_numpy(padded_vector[:, k]))
                    else:
                        columns_data.append(ak.to_numpy(events_filtered[feat]))

                data_array = np.column_stack(columns_data)

                filename = data_output / f"{file_prefix}_part{batch_idx}.et{et_idx}.eta{eta_idx}.npz"
                np.savez(
                    filename,
                    feature=np.array(features_list, dtype=str),
                    data=data_array,
                    target=target_array,
                    source=source_array
                )
                print(f"Saved: {filename} ")

                del mask, events_filtered, padded_vector, columns_data, data_array, target_array, source_array, type_mask, mc_types
                gc.collect()

        del events
        gc.collect()


cols = [
    'avgmu', 'ph_et', 'ph_eta', 'ph_phi', 'ph_Reta', 'ph_Rphi',
       'ph_e237', 'ph_e277', 'ph_Rhad', 'ph_Rhad1', 'ph_weta1',
       'ph_weta2', 'ph_f1', 'ph_f3', 'ph_f1core', 'ph_fracs1',
       'ph_f3core', 'ph_Eratio', 'ph_deltaE', 'ph_ethad', 'ph_wtots1',
       'ph_ethad1', 'ph_calo_eta', 'ph_calo_phi', 'ph_calo_et',
       'ph_calo_etaBE2', 'ph_calo_e', 'ph_hasCalo', 'ph_ptcone20',
       'ph_ptcone30', 'ph_ptcone40', 'ph_ptvarcone20', 'ph_ptvarcone30',
       'ph_ptvarcone40', 'ph_isIsoFixedLoose', 'ph_passCaloIso',
       'ph_passTrkIso', 'ph_tight', 'ph_medium', 'ph_loose',
       'ph_nGoodVtx', 'ph_nPileupPrimaryVtx', 'trig_L1_ph_eta',
       'trig_L1_ph_phi', 'trig_L1_ph_emClus', 'trig_L1_ph_roi_et',
       'trig_L1_ph_emIso', 'trig_L1_ph_hadCore', 'trig_L1_ph_tauClus',
       'trig_L1eFex_ph_eta', 'trig_L1eFex_ph_phi',
       'trig_L1eFex_ph_roi_et', 'trig_L1eFex_ph_wstot',
       'trig_L1eFex_ph_reta', 'trig_L1eFex_ph_rhad', 'trig_L2_calo_et',
       'trig_L2_calo_eta', 'trig_L2_calo_phi', 'trig_L2_calo_e237',
       'trig_L2_calo_e277', 'trig_L2_calo_fracs1', 'trig_L2_calo_weta2',
       'trig_L2_calo_ehad1', 'trig_L2_calo_emaxs1',
       'trig_L2_calo_e2tsts1', 'trig_L2_calo_wstot',
       'trig_L2_calo_rings', 'mc_hasMC',
       'mc_eta', 'mc_phi', 'mc_pt', 'mc_isTop', 'mc_isQuark',
       'mc_isParton', 'mc_isMeson', 'mc_isTau', 'mc_isMuon',
       'mc_isPhoton', 'mc_isElectron', 'mc_origin', 'mc_type', 'mc_pdgID',
       'mc_status'
]

if __name__ == '__main__':
    BASE_DIR = Path("/eos/user/p/pmourafr/RootFiles")
    INPUT_DIR = [
        "user.pmourafr.mc23_valid.801278.Py8EG_A14NNPDF23LO_perf_JF17.re.photonRinger184230042026_XYZ.root.tgz",
        "user.pmourafr.mc23_valid.801279.Py8EG_A14NNPDF23LO_perf_JF35.re.photonRinger184230042026_XYZ.root.tgz",
        "user.pmourafr.mc23_valid.801280.Py8EG_A14NNPDF23LO_perf_JF50.re.photonRinger184230042026_XYZ.root.tgz",
        "user.pmourafr.mc23_valid.801650.Py8_gammajet_frag_DP17_35_FullS.photonRinger183230042026_XYZ.root.tgz",
        "user.pmourafr.mc23_valid.801651.Py8_gammajet_frag_DP35_50_FullS.photonRinger085801052026_XYZ.root.tgz",
        "user.pmourafr.mc23_valid.801651.Py8_gammajet_frag_DP35_50_FullS.photonRinger183230042026_XYZ.root.tgz",
        "user.pmourafr.mc23_valid.801652.Py8_gammajet_frag_DP50_70_FullS.photonRinger010501052026_XYZ.root.tgz",
        "user.pmourafr.mc23_valid.801652.Py8_gammajet_frag_DP50_70_FullS.photonRinger090501052026_XYZ.root.tgz",
        "user.pmourafr.mc23_valid.801652.Py8_gammajet_frag_DP50_70_FullS.photonRinger183230042026_XYZ.root.tgz",
        "user.pmourafr.mc23_valid.801653.Py8_gammajet_frag_DP70_140_Full.photonRinger183230042026_XYZ.root.tgz"
    ]

    OUTPUT_DIR = "results"
    TREE_NAME = "run_450000/HLT/EgammaMon/summary/events"
    VECTOR_COLUMN_NAME = "trig_L2_calo_rings"

    ############ 72 regions
    # ETA_BINS = [0,0.6,0.8,1.15,1.37,1.52,1.81,2.01, 2.37,2.47]
    # ET_BINS = [25,30,35,40,45,50,60,80,100]
    # ET_BINS = np.array(ET_BINS) * 1000

    ############ 30 regions (5 ET x 6 eta)
    ETA_BINS = [0, 0.8, 1.37, 1.54, 2.37, 2.5, 999.0]
    ET_BINS = [15, 20, 30, 40, 50, 999999]
    ET_BINS = np.array(ET_BINS) * 1000

    for input_dir in INPUT_DIR:
        file_prefix = input_dir.split('user.pmourafr.')[1].split('.root.tgz')[0]
        process_root_files(
            input_dir=BASE_DIR / input_dir,
            output_dir=OUTPUT_DIR,
            tree_name=TREE_NAME,
            vector_col=VECTOR_COLUMN_NAME,
            eta_bins=ETA_BINS,
            et_bins=ET_BINS,
            target_columns=cols,
            file_prefix=file_prefix)
