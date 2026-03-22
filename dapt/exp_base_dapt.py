import os
import csv
import torch
import math
import setup_paths
import numpy as np
from model_dapt import Dapt
from constellation import Constellation
from utils_io import load_yaml, save_pkl
import pandas as pd
from utils_score import score

dataset_name = 'norman'  # norman, adamson, dixit, k562, rpe1
exp_mode_a = 'single'  # single, double
exp_mode_b = 'see0'  # see0, see1, see2


exp_mode = f'{exp_mode_a}_{exp_mode_b}' if exp_mode_a != 'single' else exp_mode_a
dir_exp_config_data = f'C:/Users/NKH/Gears/exp_config_data/exp_base/{exp_mode}_{dataset_name}/'
dir_exp_config_model = f'C:/Users/NKH/Gears/exp_config_model/gears/'
dir_write = f'C:/Users/NKH/Gears/exp/result_base_gears_default/{exp_mode}_{dataset_name}/'
path_write_tst = f'{dir_write}aggregate.csv'
os.makedirs(dir_write, exist_ok=True)

for dir in [d for d in os.listdir(dir_exp_config_data)]:
    if not os.path.isdir(os.path.join(dir_exp_config_data, dir)):
        continue
    if not dir.startswith('exp_'):
        continue
    exp_index = dir.split('_')[-1]

    path_config_data_init = f'{dir_exp_config_data}{dir}/config_data_init_default.yaml'
    path_config_data_run = f'{dir_exp_config_data}{dir}/config_data_run.yaml'
    path_config_model = f'{dir_exp_config_model}default_{dataset_name}.yaml'

    constellation = Constellation(path_config_data_init)

    # build the per-perturbation feature matrix for pert_features
    go_path = "C:/Users/NKH/Gears/data_processed/go.csv"
    go_df = pd.read_csv(go_path)
    go_df = go_df[go_df["source"] != go_df["target"]].copy()

    n = int(max(go_df["source"].max(), go_df["target"].max())) + 1
    A = np.zeros((n, n), dtype=np.float32)
    s = go_df["source"].to_numpy()
    t = go_df["target"].to_numpy()
    w = go_df["importance"].to_numpy(np.float32)

    # keep the strongest edge per pair
    np.maximum.at(A, (s, t), w)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0.0)
    A /= (A.sum(axis=1, keepdims=True) + 1e-6)

    pert_features = torch.from_numpy(A)  # .shape = [n_perts, n_perts]

    # Use the torch tensor you already created
    A_dense = pert_features.float().cpu()  # [n_perts, n_perts]
    n_perts = A_dense.shape[0]
    d = 128  

    g = torch.Generator().manual_seed(12345)
    R = torch.randn(n_perts, d, generator=g) / math.sqrt(d)
    pert_descriptor_tensor = A_dense @ R
    # column-wise standardization (optional but helps)
    mu = pert_descriptor_tensor.mean(0, keepdim=True)
    sigma = pert_descriptor_tensor.std(0, unbiased=False, keepdim=True) + 1e-6
    pert_descriptor_tensor = (pert_descriptor_tensor - mu) / sigma



    helper = constellation.exp_standard(**load_yaml(path_config_data_run))

    loader_trn = helper["loader_trn"]
    loader_val = helper["loader_val"]
    loader_tst = helper["loader_tst"]
    edges_g = helper["edges_g"]
    edges_p = helper["edges_p"]
    edge_weights_g = helper["edge_weights_g"]
    edge_weights_p = helper["edge_weights_p"]


    config_model = load_yaml(path_config_model)
    config_model["edges_genes"] = edges_g
    config_model["edges_perts"] = edges_p
    config_model["edges_weights_genes"] = edge_weights_g
    config_model["edges_weights_perts"] = edge_weights_p
    config_model["pert_descriptor_tensor"] = pert_descriptor_tensor
    config_model["descriptor_dim"] = int(pert_descriptor_tensor.shape[1])
    config_model["use_adapter"] = True
    config_model["fusion_mode"] = "hybrid"



    model = Dapt(**config_model)
    results = model.exp_standard(loader_trn, loader_val, loader_tst, "cuda", constellation)


    include_keys = [key for key in results.keys() if key not in ["cond_names", "y_pred", "y_cond"]]
    if not os.path.exists(path_write_tst):
        with open(path_write_tst, "w", newline="") as file:
            writer = csv.writer(file)
            header = ["index"] + include_keys
            writer.writerow(header)
            row = [str(exp_index)] + [np.mean(results[key]) for key in include_keys]
            writer.writerow(row)
    else:
        with open(path_write_tst, "a", newline="") as file:
            writer = csv.writer(file)
            row = [str(exp_index)] + [np.mean(results[key]) for key in include_keys]
            writer.writerow(row)

    save_pkl(results, dir_write+f"dapt_{exp_index}.pkl")

    del constellation, helper, loader_trn, loader_val, loader_tst, edges_g, edges_p, edge_weights_g, edge_weights_p
    del model, results
    torch.cuda.empty_cache()
