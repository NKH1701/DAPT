The DAPT framework is organized into three primary modules based on its architecture: the Modeling Engine, the Data Management Engine, and the Data Loading Interface.



---



## 1. The Modeling Engine



This module contains the core classes responsible for the neural network logic, graph message passing, and perturbation embeddings.



### `Dapt`

The high-level Controller class wrapping the neural network. It acts as the primary interface for the user or main script.

```python
class Dapt:
    def __init__(self, **kwargs):
        self.model = None
        if kwargs.get('load_path'):
            self.load(kwargs['load_path'])
        else:
            self.model = GearsModel(**kwargs)

    def save(self, path):
        path_state = os.path.join(path, 'model.pth')
        path_config = os.path.join(path, 'config.pkl')

        torch.save(self.model.state_dict(), path_state)
        save_pkl(self.model.get_config(), path_config)

    def load(self, path):
        path_state = os.path.join(path, 'model.pth')
        path_config = os.path.join(path, 'config.pkl')

        model_config = load_pkl(path_config)
        self.model = GearsModel(**model_config)
        self.model.load_state_dict(torch.load(path_state, map_location='cpu'))

    def train(self, loader_trn, loader_val, device, constellation):
        self.model = self.model.to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

        # --- AUTOENCODER SETTINGS ---
        recon_loss_fcn = nn.MSELoss()
        lambda_recon = 0.2  # Weight of the reconstruction loss

        best_model = None
        best_score = np.inf

        print('Training...')
        for epoch in range(20):
            self.model.train()


            for step, (x, y_cond, y_ctrl) in enumerate(loader_trn):
                x = x.to(device)
                y_cond = y_cond.to(device)
                y_ctrl = y_ctrl.to(device)

                optimizer.zero_grad()
                y_pred, x_recon, x_target = self.model(x, y_ctrl)

                loss = loss_fcn(x, y_pred, y_cond, constellation)

                # Structural Loss (Autoencoder Reconstruction)
                loss_recon = recon_loss_fcn(x_recon, x_target)

                # Combine
                loss_total = loss + (lambda_recon * loss_recon)

                loss_total.backward()

                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
                optimizer.step()


                print(f'Epoch {epoch}, Step {step} done') if step % 100 == 0 and step != 0 else None

            scheduler.step()

            model_score = self.validate(loader_val, device, constellation)
            if model_score < best_score:
                best_score = model_score
                best_model = deepcopy(self.model)
            print(f'Epoch {epoch} done')

        self.model = best_model
        print('Training done')

    def validate(self, loader_val, device, constellation):
        print("Validating...")
        cond_masks_all, y_pred_all, y_ctrl_all, y_cond_all = predict_epoch(self.model, loader_val, device)
        model_score = score(cond_masks_all, y_pred_all, y_ctrl_all, y_cond_all, constellation, True)
        print("Validation done")
        return model_score

    def evaluate(self, loader_tst, device, constellation):
        print('Evaluating...')
        cond_masks_all, y_pred_all, y_ctrl_all, y_cond_all = predict_epoch(self.model, loader_tst, device)
        evaluation_score = score(cond_masks_all, y_pred_all, y_ctrl_all, y_cond_all, constellation)
        print('Evaluation done')
        return evaluation_score

    def exp_standard(self, loader_trn, loader_val, loader_tst, device, constellation):
        self.train(loader_trn, loader_val, device, constellation)

        evaluation_result = self.evaluate(loader_tst, device, constellation)
        return evaluation_result
```


- **Responsibilities:** Manages abstract interfaces for training, validation, evaluation, and model state persistence (saving/loading weights).

- **Key Methods:** `train()`, `predict()`, `save()`, `load()`.



### `DaptModel`

The primary PyTorch `nn.Module` that defines the structural layers of the network.

```python
class DaptModel(nn.Module):
    def __init__(self, **kwargs):
        super(GearsModel, self).__init__()

        # Set up parameters regarding the data
        self.n_genes = kwargs['n_genes']
        self.n_perts = kwargs['n_perts']

        oov_mask = kwargs.get("oov_pert_mask", None)
        if oov_mask is not None:
            self.register_buffer("oov_pert_mask", oov_mask.bool())  # [n_perts]
        else:
            self.oov_pert_mask = None

        # initialize modules
        self.eb_gene_indv = nn.Embedding(self.n_genes, 64, max_norm=True)
        self.bn_gene_indv = nn.BatchNorm1d(64)

        self.register_buffer('edges_genes', kwargs.get('edges_genes'))
        self.register_buffer('edges_weights_genes', kwargs.get('edges_weights_genes'))
        self.eb_gene_gaph = nn.Embedding(self.n_genes, 64, max_norm=True)
        self.sg_gene_grah = SGConv(64, 64, 1)

        self.pj_gene_fnal = ModuleMlp([64, 64, 64])

        self.register_buffer('edges_perts', kwargs.get('edges_perts'))
        self.register_buffer('edges_weights_perts', kwargs.get('edges_weights_perts'))
        self.eb_pert_gaph = nn.Embedding(self.n_perts, 64, max_norm=True)
        self.sg_pert_grah = SGConv(64, 64, 1)
```

- **Responsibilities:** Manages the gene embeddings, initializes the Perturbation Regularized Autoencoder (RAE), and executes the Graph Convolutional (SGConv) networks over the biological knowledge graphs.



### `ModuleMlp`

A specialized neural network subcomponent utilized heavily by `DaptModel`.

```python
class ModuleMlp(nn.Module):
    def __init__(self, dim_sequence):
        super(ModuleMlp, self).__init__()
        sequence = []
        for dim_in, dim_out in zip(dim_sequence[:-1], dim_sequence[1:]):
            sequence.extend([
                nn.Linear(dim_in, dim_out),
                nn.BatchNorm1d(dim_out),
                nn.ReLU()
            ])
        self.sequence = nn.Sequential(*sequence)

    def forward(self, x):
        if self.training and x.size(0) == 1:
            return self.sequence(torch.cat([x, x], dim=0))[0:1]
        return self.sequence(x)
```

- **Responsibilities:** Processes flattened feature vectors through sequential linear layers and non-linear activations to generate the final transcriptomic predictions.



---



## 2. The Data Management Engine



This engine is designed using a **Facade Pattern**, hiding the complex data preprocessing and mapping logic behind a single, clean orchestrator interface.



### `Constellation` (The Facade)

The central orchestrator for all biological data, graphs, and experiment states. It delegates specific data tasks to its internal specialized components.

```python
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
```

- **Responsibilities:** Loading YAML configurations, initializing data splits, and providing a unified data querying interface for the `Dapt` model during training.



### `Messenger`

The mapping utility class.

```python
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
```

- **Responsibilities:** Handles the translation between human-readable string names (e.g., gene symbol "A1BG") and internal numerical tensor indices required by PyTorch.



### `Sanctuary`

The state-storage class.

```python
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
```

- **Responsibilities:** Stores the experimental responses (post-perturbation expressions) and maintains the statistical fingerprints of the single-cell samples.



### `Eureka`

The graph topology constructor.

```python
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
```

- **Responsibilities:** Computes and retrieves the functional relationships and edges between genes. It builds the adjacency matrices for both the Gene Co-expression graph and the Gene Ontology (GO) perturbation graph.



---



## 3. The Data Loading Interface



This module handles the continuous delivery of large-scale single-cell data to the GPU.



### `Postman`

The data batching and delivery class.

```python
class Postman(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cond_mask, y_cond, y_ctrl = self.samples[idx]
        return cond_mask, y_cond, y_ctrl
```

- **Responsibilities:** Bridges the `Constellation` data storage and the model's training loop. It safely retrieves large batches of training and validation samples from `Sanctuary` and delivers them to the `Dapt` controller in optimized tensor formats.

