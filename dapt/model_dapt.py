import os
import torch
import numpy as np
import torch.nn as nn
import torch.functional as F
from copy import deepcopy
from utils_score import score
from torch_geometric.nn import SGConv
from utils_io import save_pkl, load_pkl
from utils_dapt import loss_fcn, predict


def predict_epoch(model, loader, device):
    model.eval()
    cond_masks_all, y_pred_all, y_ctrl_all, y_cond_all = [], [], [], []

    with torch.no_grad():
        for x, y_cond, y_ctrl in loader:
            x = x.to(device)
            y_cond = y_cond.to(device)
            y_ctrl = y_ctrl.to(device)

            # Unpack the tuple here!
            # The model returns: (prediction, reconstruction, target)
            output = model(x, y_ctrl)

            if isinstance(output, tuple):
                y_pred = output[0]  # Take only the first element (the gene expression prediction)
            else:
                y_pred = output

            cond_masks_all.append(x.cpu())
            y_pred_all.append(y_pred.cpu())
            y_ctrl_all.append(y_ctrl.cpu())
            y_cond_all.append(y_cond.cpu())

    # Concatenate and convert to numpy (Logic taken from utils_gears.predict)
    cond_masks_all = torch.cat(cond_masks_all, dim=0).numpy()
    y_pred_all = torch.cat(y_pred_all, dim=0).numpy()
    y_ctrl_all = torch.cat(y_ctrl_all, dim=0).numpy()
    y_cond_all = torch.cat(y_cond_all, dim=0).numpy()

    return cond_masks_all, y_pred_all, y_ctrl_all, y_cond_all

class Dapt:
    def __init__(self, **kwargs):
        self.model = None
        if kwargs.get('load_path'):
            self.load(kwargs['load_path'])
        else:
            self.model = DaptModel(**kwargs)

    def save(self, path):
        path_state = os.path.join(path, 'model.pth')
        path_config = os.path.join(path, 'config.pkl')

        torch.save(self.model.state_dict(), path_state)
        save_pkl(self.model.get_config(), path_config)

    def load(self, path):
        path_state = os.path.join(path, 'model.pth')
        path_config = os.path.join(path, 'config.pkl')

        model_config = load_pkl(path_config)
        self.model = DaptModel(**model_config)
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


class DaptModel(nn.Module):
    def __init__(self, **kwargs):
        super(DaptModel, self).__init__()

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
        self.eb_pert_gaph = nn.Embedding(self.n_perts, 12, max_norm=True)#64->12
        self.sg_pert_grah = SGConv(12, 12, 1)


        # --- Adapter config & descriptors (per-pert) ---
        self.use_adapter = bool(kwargs.get("use_adapter", True))
        self.fusion_mode = str(kwargs.get("fusion_mode", "hybrid"))
        self.descriptor_dim = int(kwargs.get("descriptor_dim", 0))

        pert_desc = kwargs.get("pert_descriptor_tensor", None)  # [n_perts, descriptor_dim]
        if self.use_adapter:
            assert pert_desc is not None and self.descriptor_dim > 0, "Adapter enabled but no per-pert descriptors provided"
            self.register_buffer("pert_descriptors", pert_desc)  # stays on device with model

        self.encoder = nn.Sequential(
            nn.Linear(self.descriptor_dim, 32), #single norman = 128
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 12) #single norman = 64
        )

        self.decoder = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, self.descriptor_dim)
        )

        def recon_loss(self, desc, desc_recon):
            return F.mse_loss(desc_recon, desc)


        self.pj_pert_fnal = ModuleMlp([12, 64, 64]) # correction: 64 -> 12

        self.bn_hybd = nn.BatchNorm1d(64)
        self.pj_hybd = ModuleMlp([64, 128, 64])

        self.indv_w1 = nn.Parameter(torch.rand(self.n_genes, 64))
        self.indv_b1 = nn.Parameter(torch.rand(1, self.n_genes))
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)

        self.pj_fnal = ModuleMlp([self.n_genes, 64, 64])

        self.indv_w2 = nn.Parameter(torch.rand(self.n_genes, 64+1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.n_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)


    def forward(self, x, y_ctrl):
        n_samples = x.size(0)

        fe_gene_indv = self.eb_gene_indv.weight.repeat(n_samples, 1)
        fe_gene_indv = self.bn_gene_indv(fe_gene_indv).relu()

        fe_gene_gaph = self.eb_gene_gaph.weight
        fe_gene_gaph = self.sg_gene_grah(fe_gene_gaph, self.edges_genes, self.edges_weights_genes)
        fe_gene_gaph = fe_gene_gaph.repeat(n_samples, 1)

        fe_gene_fnal = fe_gene_indv + 0.2 * fe_gene_gaph
        fe_gene_fnal = self.pj_gene_fnal(fe_gene_fnal)

        fe_pert_gaph = self.eb_pert_gaph.weight
        fe_pert_gaph = self.sg_pert_grah(fe_pert_gaph, self.edges_perts, self.edges_weights_perts)


        x_shifted = x + 1

        if self.oov_pert_mask is not None:
            # (1) No message passing / ID: zero the graph/ID bank rows for OOV perts
            fe_pert_gaph = fe_pert_gaph.clone()
            fe_pert_gaph[self.oov_pert_mask] = 0.0

        if self.use_adapter:
            fe_pert_bank = self.encoder(self.pert_descriptors)
            x_recon = self.decoder(fe_pert_bank)

            fe_pert_bank = fe_pert_gaph + fe_pert_bank




        zero_row = torch.zeros(1, fe_pert_bank.size(1), device=x.device)
        fe_pert_bank_extended = torch.cat([zero_row, fe_pert_bank], dim=0)
        fe_pert_cola = fe_pert_bank_extended[x_shifted[:, 0]]
        fe_pert_colb = fe_pert_bank_extended[x_shifted[:, 1]]
        fe_pert_fnal = fe_pert_cola + fe_pert_colb
        fe_pert_fnal = self.pj_pert_fnal(fe_pert_fnal)
        fe_pert_fnal = fe_pert_fnal.repeat_interleave(self.n_genes, dim=0)

        batch_pert_indices = x.long().view(-1)


        batch_x_recon = x_recon[batch_pert_indices]
        batch_x_target = self.pert_descriptors[batch_pert_indices]

        fe_hybd = fe_gene_fnal + fe_pert_fnal
        fe_hybd = self.bn_hybd(fe_hybd).relu()
        fe_hybd = self.pj_hybd(fe_hybd)

        fe_hybd = fe_hybd.reshape(n_samples, self.n_genes, -1)

        # replace with:
        N, S, D = fe_hybd.shape
        w1 = self.indv_w1.to(fe_hybd.dtype).to(fe_hybd.device).contiguous()  # [S, D]
        b1 = self.indv_b1
        b1 = b1.squeeze(0) if b1.dim() == 2 and b1.size(0) == 1 else b1  # [S]
        fe_indv = (fe_hybd.contiguous() * w1.unsqueeze(0)).sum(dim=2) + b1.unsqueeze(0)  # [N, S]
        fe_hybd = fe_indv

        fe_fnal_a = fe_hybd.unsqueeze(-1)
        fe_fnal_b = self.pj_fnal(fe_hybd)
        fe_fnal_b = fe_fnal_b.repeat(1, self.n_genes).reshape(n_samples, self.n_genes, -1)
        fe_fnal = torch.cat([fe_fnal_a, fe_fnal_b], dim=2)

        # replace with:
        N, S, D2 = fe_fnal.shape
        w2 = self.indv_w2.to(fe_fnal.dtype).to(fe_fnal.device).contiguous()  # [S, D2]
        b2 = self.indv_b2
        b2 = b2.squeeze(0) if b2.dim() == 2 and b2.size(0) == 1 else b2  # [S]
        pred = (fe_fnal.contiguous() * w2.unsqueeze(0)).sum(dim=2) + b2.unsqueeze(0)

        pred = pred + y_ctrl

        return pred, batch_x_recon, batch_x_target

    def get_config(self):
        return {
            # NOTE: The edges and weights are not saved, should be generated via the constellation.
            'n_genes': self.n_genes,
            'n_perts': self.n_perts
        }
