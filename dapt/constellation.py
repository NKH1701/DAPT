import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils_constellation import relate, correlate
from utils_io import load_pkl, load_yaml, load_json


class Messenger:
    def __init__(
            self,
            map_pert_perturbation,
            map_perturbation_pert,
            map_gene_loc_gene_name,
            map_cond_non_zero_gene_locs_asc,
            map_cond_non_drop_gene_locs_asc,
            map_cond_non_zero_gene_locs_rank,
            map_cond_non_drop_gene_locs_rank,
            map_cond_complete_gene_locs_rank
    ):
        self.map_pert_perturbation = map_pert_perturbation
        self.map_perturbation_pert = map_perturbation_pert
        self.map_gene_loc_gene_name = map_gene_loc_gene_name
        self.map_cond_non_zero_gene_locs_asc = map_cond_non_zero_gene_locs_asc
        self.map_cond_non_drop_gene_locs_asc = map_cond_non_drop_gene_locs_asc
        self.map_cond_non_zero_gene_locs_rank = map_cond_non_zero_gene_locs_rank
        self.map_cond_non_drop_gene_locs_rank = map_cond_non_drop_gene_locs_rank
        self.map_cond_complete_gene_locs_rank = map_cond_complete_gene_locs_rank

        self.map_pert_perturbation[-1] = "ctrl"
        self.map_perturbation_pert["ctrl"] = -1

        self.valid_modes = {
            "pert": self._get_pert_from_perturbation,
            "perturbation": self._get_perturbation_from_pert,
            "cond": self._get_cond_from_condition,
            "condition": self._get_condition_from_cond,
            "non_zero": self._get_non_zero_gene_locs_asc_from_cond,
            "rank": self._get_complete_gene_locs_rank_from_cond,
            "gene": self._get_gene_name_from_loc,
        }

    def _get_pert_from_perturbation(self, perturbation):
        # return self.map_perturbation_pert.get(perturbation)
        return self.map_perturbation_pert[perturbation]

    def _get_perturbation_from_pert(self, pert):
        # return self.map_pert_perturbation.get(pert)
        return self.map_pert_perturbation[pert]

    def _get_cond_from_condition(self, condition):
        if condition == "ctrl":
            return (-1, -1)
        perturbation_a, perturbation_b = condition.split("+")
        pert_a = self.map_perturbation_pert[perturbation_a]
        pert_b = self.map_perturbation_pert[perturbation_b]
        return (pert_a, pert_b)

    def _get_condition_from_cond(self, cond):
        if cond == (-1, -1):
            return "ctrl"
        pert_a, pert_b = cond
        perturbation_a = self.map_pert_perturbation[pert_a]
        perturbation_b = self.map_pert_perturbation[pert_b]
        return f"{perturbation_a}+{perturbation_b}"

    def _get_non_zero_gene_locs_asc_from_cond(self, cond):
        return self.map_cond_non_zero_gene_locs_asc[cond]

    def _get_complete_gene_locs_rank_from_cond(self, cond):
        return self.map_cond_complete_gene_locs_rank[cond]

    def _get_gene_name_from_loc(self, loc):
        return self.map_gene_loc_gene_name[loc]

    def helper(self, mode, *args, **kwargs):
        if mode not in self.valid_modes:
            raise ValueError(f"Invalid mode: {mode}.")
        else:
            output = self.valid_modes[mode](*args, **kwargs)

        return output


class Sanctuary:
    def __init__(self, map_cond_responses, fingerprint):
        self.map_cond_responses = map_cond_responses
        self.fingerprint = fingerprint
        self._summary()

        self.valid_modes = {
            "dataset": self._get_dataset_info,
            "statistics": self._get_statistics_from_cond,
            "sample_all": self._get_samples_from_conds_all,
            "sample_avg": self._get_samples_from_conds_avg,
            "response_g": self._get_responses_for_gene_corr,
        }

    def _summary(self):
        conds = list(self.map_cond_responses.keys())
        perts = [pert for cond in conds for pert in cond]
        perts = list(set(perts))
        num_genes = self.map_cond_responses[list(self.map_cond_responses.keys())[0]].shape[1]
        num_responses = sum([responses.shape[0] for responses in self.map_cond_responses.values()])

        self.dataset_info = {
            "num_conds": len(conds),
            "num_perts": len(perts),
            "num_genes": num_genes,
            "num_responses": num_responses,
            "conds": conds,
            "perts": perts
        }

    def _get_dataset_info(self):
        return self.dataset_info

    def _get_statistics_from_cond(self, cond):
        if cond not in self.map_cond_responses:
            return None
        responses_cond = self.map_cond_responses[cond]
        responses_ctrl = self.map_cond_responses[(-1, -1)]
        avg_response_cond = responses_cond.mean(axis=0)
        avg_response_ctrl = responses_ctrl.mean(axis=0)
        avg_response_diff = avg_response_cond - avg_response_ctrl

        return {
            "avg_response_cond": avg_response_cond,
            "avg_response_ctrl": avg_response_ctrl,
            "avg_response_diff": avg_response_diff,
            "responses_cond": responses_cond,
            "responses_ctrl": responses_ctrl
        }

    def _get_responses_for_gene_corr(self, conds):
        conds_valid = [cond for cond in conds if -1 in cond]

        responses = []
        for cond in conds_valid:
            responses_cond = self.map_cond_responses[cond]
            responses.append(responses_cond)
        responses = np.concatenate(responses, axis=0)

        return responses

    def _get_samples_from_conds_all(self, conds):
        responses_ctrl = self.map_cond_responses[(-1, -1)]

        samples = []
        for cond in conds:
            responses_cond = self.map_cond_responses[cond]
            indices_cond, indices_ctrl = self.fingerprint[cond]
            npy_cond = responses_cond[indices_cond]
            npy_ctrl = responses_ctrl[indices_ctrl]

            for y_cond, y_ctrl in zip(npy_cond, npy_ctrl):
                x = torch.tensor(cond, dtype=torch.long)
                y_cond = torch.tensor(y_cond, dtype=torch.float32)
                y_ctrl = torch.tensor(y_ctrl, dtype=torch.float32)
                samples.append((x, y_cond, y_ctrl))

        random.shuffle(samples)
        return samples

    def _get_samples_from_conds_avg(self, conds):
        responses_ctrl = self.map_cond_responses[(-1, -1)]
        avg_response_ctrl = responses_ctrl.mean(axis=0)

        samples = []
        for cond in conds:
            responses_cond = self.map_cond_responses[cond]
            avg_response_cond = responses_cond.mean(axis=0)

            x = torch.tensor(cond, dtype=torch.long)
            y_cond = torch.tensor(avg_response_cond, dtype=torch.float32)
            y_ctrl = torch.tensor(avg_response_ctrl, dtype=torch.float32)
            samples.append((x, y_cond, y_ctrl))

        return samples

    def helper(self, mode, *args, **kwargs):
        if mode not in self.valid_modes:
            raise ValueError(f"Invalid mode: {mode}.")
        else:
            output = self.valid_modes[mode](*args, **kwargs)

        return output


class Eureka:
    def __init__(self, go):
        self.go = go

    def _inspire_pert(self, threshold, max_degree):
        return relate(self.go, threshold, max_degree)

    def _inspire_gene(self, threshold, max_degree, responses):
        return relate(correlate(responses), threshold, max_degree)

    def inspire(self, threshold_g, threshold_p, max_degree_g, max_degree_p, responses):
        edges_g, weights_g = self._inspire_gene(threshold_g, max_degree_g, responses)
        edges_p, weights_p = self._inspire_pert(threshold_p, max_degree_p)

        return edges_g, weights_g, edges_p, weights_p


class Postman(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cond_mask, y_cond, y_ctrl = self.samples[idx]
        return cond_mask, y_cond, y_ctrl


class Constellation:
    def __init__(self, path_config=None, **kwargs):
        if path_config is not None:
            self._construct(**load_yaml(path_config))
        else:
            self._construct(**kwargs)

    def _construct(self, **kwargs):
        go = pd.read_csv("C:/Users/NKH/Gears/" + kwargs["path_go"])
        meta = load_pkl("C:/Users/NKH/Gears/" + kwargs["path_meta"])
        map_cond_responses = meta["map_cond_responses"]
        fingerprint = load_pkl("C:/Users/NKH/Gears/" + kwargs["path_fingerprint"])
        args_messenger = {
            "map_pert_perturbation": meta["map_pert_perturbation"],
            "map_perturbation_pert": meta["map_perturbation_pert"],
            "map_gene_loc_gene_name": meta["map_gene_loc_gene_name"],
            "map_cond_non_zero_gene_locs_asc": meta["map_cond_non_zero_gene_locs_asc"],
            "map_cond_non_drop_gene_locs_asc": meta["map_cond_non_drop_gene_locs_asc"],
            "map_cond_non_zero_gene_locs_rank": meta["map_cond_non_zero_gene_locs_rank"],
            "map_cond_non_drop_gene_locs_rank": meta["map_cond_non_drop_gene_locs_rank"],
            "map_cond_complete_gene_locs_rank": meta["map_cond_complete_gene_locs_rank"]
        }

        self.messenger = Messenger(**args_messenger)
        self.sanctuary = Sanctuary(map_cond_responses, fingerprint)
        self.eureka = Eureka(go)

    def exp_standard(self, **kwargs):
        batch_size = kwargs["batch_size"]
        threshold_g = kwargs["threshold_g"]
        threshold_p = kwargs["threshold_p"]
        max_degree_g = kwargs["max_degree_g"]
        max_degree_p = kwargs["max_degree_p"]
        assignment = load_json("C:/Users/NKH/Gears/" + kwargs["path_assignment"])

        # Convert list of lists to list of tuples
        # NOTE: This is a workaround for the issue with the JSON incompatibility with tuples
        for key, conds in assignment.items():
            assignment[key] = [tuple(cond) for cond in conds]

        responses_corr = self.sanctuary.helper("response_g", conds=assignment["trn"])
        eg, wg, ep, wp = self.eureka.inspire(
            threshold_g, threshold_p, max_degree_g, max_degree_p, responses_corr)

        mode_trn = "sample_all"
        mode_val = "sample_all" if kwargs["compatible_mode"] else "sample_avg"
        mode_tst = "sample_avg"

        samples_trn = self.sanctuary.helper(mode_trn, conds=assignment["trn"])
        samples_val = self.sanctuary.helper(mode_val, conds=assignment["val"])
        samples_tst = self.sanctuary.helper(mode_tst, conds=assignment["tst"])
        loader_trn = DataLoader(Postman(samples_trn), batch_size=batch_size, shuffle=True)
        loader_val = DataLoader(Postman(samples_val), batch_size=batch_size, shuffle=False)
        loader_tst = DataLoader(Postman(samples_tst), batch_size=batch_size, shuffle=False)

        return {
            "edges_p": ep,
            "edges_g": eg,
            "edge_weights_p": wp,
            "edge_weights_g": wg,
            "loader_trn": loader_trn,
            "loader_val": loader_val,
            "loader_tst": loader_tst
        }

    def helper(self, mode, *args, **kwargs):
        key_a = self.messenger.valid_modes.keys()
        key_b = self.sanctuary.valid_modes.keys()
        key = list(key_a) + list(key_b)

        if mode not in key:
            raise ValueError(f"Invalid mode: {mode}.")
        elif mode in key_a:
            output = self.messenger.helper(mode, *args, **kwargs)
        elif mode in key_b:
            output = self.sanctuary.helper(mode, *args, **kwargs)

        return output
