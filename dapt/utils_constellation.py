import torch
import random
import warnings
import numpy as np
import pandas as pd


def match(n_cond: int, n_ctrl: int, n_samp: int, ctrl_mask=False):
    indices_cond = None
    indices_ctrl = None

    n_samp = n_cond if n_samp == -1 else n_samp

    if ctrl_mask:
        indices_cond = np.random.choice(n_ctrl, n_samp, replace=True)
        indices_ctrl = indices_cond
    else:
        if n_samp <= min(n_cond, n_ctrl):
            indices_cond = np.random.choice(n_cond, n_samp, replace=False)
            indices_ctrl = np.random.choice(n_ctrl, n_samp, replace=False)
        elif n_samp <= max(n_cond, n_ctrl):
            n_large, n_small = max(n_cond, n_ctrl), min(n_cond, n_ctrl)
            indices_large = np.random.permutation(n_large)[:n_samp]
            indices_small = [np.random.permutation(n_small) for _ in range(n_samp // n_small)]
            indices_small.append(np.random.choice(n_small, n_samp % n_small, replace=False))
            indices_small = np.concatenate(indices_small)
            if n_cond > n_ctrl:
                indices_cond, indices_ctrl = indices_large, indices_small
            else:
                indices_cond, indices_ctrl = indices_small, indices_large
        else:
            selected = set()
            n_large, n_small = max(n_cond, n_ctrl), min(n_cond, n_ctrl)
            indices_large = np.random.permutation(n_large)
            indices_small = [np.random.permutation(n_small) for _ in range(n_large // n_small)]
            indices_small.append(np.random.choice(n_small, n_large % n_small, replace=False))
            indices_small = np.concatenate(indices_small)
            if n_cond >= n_ctrl:
                indices_cond_a, indices_ctrl_a = indices_large, indices_small
            else:
                indices_cond_a, indices_ctrl_a = indices_small, indices_large
            for index_cond, index_ctrl in zip(indices_cond_a, indices_ctrl_a):
                selected.add((index_cond, index_ctrl))
            indices_cond_b, indices_ctrl_b = [], []
            while len(selected) < n_samp:
                index_cond = np.random.choice(n_cond)
                index_ctrl = np.random.choice(n_ctrl)
                if (index_cond, index_ctrl) not in selected:
                    selected.add((index_cond, index_ctrl))
                    indices_cond_b.append(index_cond)
                    indices_ctrl_b.append(index_ctrl)
            indices_cond = np.concatenate([indices_cond_a, np.array(indices_cond_b)])
            indices_ctrl = np.concatenate([indices_ctrl_a, np.array(indices_ctrl_b)])

    return indices_cond, indices_ctrl


def assign(conds, r_val, r_tst):
    conds.remove((-1, -1)) if (-1, -1) in conds else None
    perts_single, perts_double = [], []
    conds_single, conds_double = [], []
    for cond in conds:
        if -1 in cond:
            conds_single.append(cond)
            perts_single.extend([pert for pert in cond])
        else:
            conds_double.append(cond)
            perts_double.extend([pert for pert in cond])
    perts_single = list(set(perts_single))
    perts_double = list(set(perts_double))
    perts_single.remove(-1)
    random.shuffle(perts_single)
    random.shuffle(perts_double)
    n_perts_single, n_perts_double = len(perts_single), len(perts_double)

    assignments = {
        "single": None,
        "double_see0": None,
        "double_see1": None,
        "double_see2": None
    }

    def count(perts_trn, perts_val, perts_tst, cond):
        perts = [pert for pert in cond]
        trn_pert = [pert for pert in perts if pert in perts_trn]
        val_pert = [pert for pert in perts if pert in perts_val]
        tst_pert = [pert for pert in perts if pert in perts_tst]

        return len(trn_pert), len(val_pert), len(tst_pert), len(perts)

    # single assignment
    n_val = int(r_val * n_perts_single)
    n_tst = int(r_tst * n_perts_single)
    perts_val = set(perts_single[:n_val])
    perts_tst = set(perts_single[n_val:n_val + n_tst])
    perts_trn = set(perts_single[n_val + n_tst:])

    conds_tst, conds_val, conds_trn = [], [], []
    valid_prefixes = [(1, 0, 0, 2), (0, 1, 0, 2), (0, 0, 1, 2)]
    for cond in conds_single:
        prefix = count(perts_trn, perts_val, perts_tst, cond)
        if prefix not in valid_prefixes:
            raise ValueError('Invalid prefix: {}'.format(prefix))
        elif prefix == (1, 0, 0, 2):
            conds_trn.append(cond)
        elif prefix == (0, 1, 0, 2):
            conds_val.append(cond)
        elif prefix == (0, 0, 1, 2):
            conds_tst.append(cond)

    assignments["single"] = {
        "trn": conds_trn + [(-1, -1)] if (-1, -1) not in conds_trn else conds_trn,
        "val": conds_val,
        "tst": conds_tst,
    }

    # double assignment
    n_val = int(r_val * n_perts_double)
    n_val = int(r_tst * n_perts_double)
    perts_val = set(perts_double[:n_val])
    perts_tst = set(perts_double[n_val:n_val + n_tst])
    perts_trn = set(perts_double[n_val + n_tst:])

    conds_tst_0, conds_val_0, conds_trn_0 = [], [], []
    conds_tst_1, conds_val_1, conds_trn_1 = [], [], []
    conds_tst_2, conds_val_2, conds_trn_2 = [], [], []
    for cond in conds:
        prefix = count(perts_trn, perts_val, perts_tst, cond)
        if prefix in [(2, 0, 0, 2), (1, 0, 0, 2)]:
            conds_trn_0.append(cond)
            conds_trn_1.append(cond)
            conds_trn_2.append(cond)
        elif prefix in [(0, 2, 0, 2)]:
            conds_val_0.append(cond)
            conds_val_2.append(cond)
        elif prefix in [(0, 0, 2, 2)]:
            conds_tst_0.append(cond)
            conds_tst_2.append(cond)
        elif prefix in [(1, 1, 0, 2)]:
            conds_val_1.append(cond)
            conds_trn_2.append(cond)
        elif prefix in [(1, 0, 1, 2)]:
            conds_tst_1.append(cond)
            conds_trn_2.append(cond)
        elif prefix in [(0, 1, 0, 2), (0, 0, 1, 2)]:
            conds_trn_2.append(cond)
        elif prefix in [(0, 1, 1, 2)]:
            continue
        else:
            ValueError('Invalid prefix: {}'.format(prefix))
    assignments
    assignments["double_see0"] = {
        'trn': conds_trn_0 + [(-1, -1)] if (-1, -1) not in conds_trn_0 else conds_trn_0,
        'val': conds_val_0,
        'tst': conds_tst_0,
    }
    assignments["double_see1"] = {
        'trn': conds_trn_1 + [(-1, -1)] if (-1, -1) not in conds_trn_1 else conds_trn_1,
        'val': conds_val_1,
        'tst': conds_tst_1,
    }
    assignments["double_see2"] = {
        'trn': conds_trn_2 + [(-1, -1)] if (-1, -1) not in conds_trn_2 else conds_trn_2,
        'val': conds_val_2,
        'tst': conds_tst_2,
    }

    return assignments


def relate(df, threshold=None, max_degree=None):
    # NOTE: Self-connections should be added in advance if needed.
    if max_degree is not None:
        df_filtered = df[df['importance'] >= threshold]
    else:
        df_filtered = df

    if max_degree is not None:
        df_sorted = df_filtered.sort_values(by=['target', 'importance'], ascending=[True, False])
        df_selected = df_sorted.groupby('target').head(max_degree)
    else:
        df_selected = df_filtered

    edges = torch.tensor(df_selected[['source', 'target']].values.T, dtype=torch.long)
    edge_weights = torch.tensor(df_selected['importance'].values, dtype=torch.float)

    return edges, edge_weights


def correlate(npy):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        corr_matrix = np.corrcoef(npy, rowvar=False)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    num_features = npy.shape[1]
    corr_df = pd.DataFrame(corr_matrix, index=range(num_features), columns=range(num_features))
    corr_series = corr_df.stack().rename('importance')
    result_df = corr_series.reset_index()
    result_df.columns = ['source', 'target', 'importance']

    return result_df
