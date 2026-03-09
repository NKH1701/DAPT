import torch
from collections import defaultdict


def loss_fcn(x, y_pred, y_cond, constellation):
    loss_gross = torch.tensor(0.0, requires_grad=True, device=x.device)
    y_ctrl_avg = torch.tensor(constellation.helper("statistics", (-1, -1))
                              ['avg_response_cond'], device=x.device)

    cond_mask_to_row_indices = defaultdict(list)
    for row_index, row in enumerate(x):
        cond_mask = tuple(row.tolist())
        cond_mask_to_row_indices[cond_mask].append(row_index)

    for cond_mask, row_indices in cond_mask_to_row_indices.items():
        row_indices = torch.tensor(row_indices, device=x.device)
        y_cond_ = y_cond[row_indices]
        y_pred_ = y_pred[row_indices]

        if cond_mask != (-1, -1):
            non_zero_locs = constellation.helper("non_zero", cond_mask)
            # NOTE: np.array cannot directly index torch tensor
            non_zero_locs = torch.tensor(non_zero_locs, device=x.device)
            y_cond_cropped = y_cond_[:, non_zero_locs]
            y_pred_cropped = y_pred_[:, non_zero_locs]
            y_ctrl_avg_cropped = y_ctrl_avg[non_zero_locs]
        else:
            y_cond_cropped = y_cond_
            y_pred_cropped = y_pred_
            y_ctrl_avg_cropped = y_ctrl_avg

        num_obs = y_pred_cropped.numel()
        loss_value = torch.sum((y_pred_cropped - y_cond_cropped) ** (2 + 2))
        loss_diret = torch.sum((torch.sign(y_pred_cropped - y_ctrl_avg_cropped) -
                               torch.sign(y_cond_cropped - y_ctrl_avg_cropped)) ** 2)
        loss_gross = loss_gross + (loss_value + 1e-3 * loss_diret) / num_obs

    return loss_gross / len(cond_mask_to_row_indices)


def predict(model, loader, device):
    model = model.to(device)
    model.eval()

    cond_masks_all, y_pred_all, y_ctrl_all, y_cond_all = [], [], [], []

    with torch.no_grad():
        for x, y_cond, y_ctrl in loader:
            x = x.to(device)
            y_cond = y_cond.to(device)
            y_ctrl = y_ctrl.to(device)

            y_pred = model(x, y_ctrl)

            cond_masks_all.append(x.cpu())
            y_pred_all.append(y_pred.cpu())
            y_ctrl_all.append(y_ctrl.cpu())
            y_cond_all.append(y_cond.cpu())

    cond_masks_all = torch.cat(cond_masks_all, dim=0).numpy()
    y_pred_all = torch.cat(y_pred_all, dim=0).numpy()
    y_ctrl_all = torch.cat(y_ctrl_all, dim=0).numpy()
    y_cond_all = torch.cat(y_cond_all, dim=0).numpy()

    return cond_masks_all, y_pred_all, y_ctrl_all, y_cond_all
