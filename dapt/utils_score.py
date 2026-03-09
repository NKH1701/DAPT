import numpy as np


def calculate_precision_at_k(y_true, y_pred, k):
    """
    Calculates Precision@K by dynamically finding the top K genes
    (based on absolute magnitude of change) in both true and predicted arrays.
    """
    n_genes = y_true.shape[1]

    # If K is larger or equal to the total gene pool, precision is naturally 1.0
    if n_genes <= k:
        return np.ones(y_true.shape[0])

    # Get the INDICES of the top K largest absolute expression values
    # np.argsort sorts ascending, so we take the last 'k' items
    true_top_k = np.argsort(np.abs(y_true), axis=1)[:, -k:]
    pred_top_k = np.argsort(np.abs(y_pred), axis=1)[:, -k:]

    # Calculate the overlap (intersection) dynamically for each perturbation in the batch
    precisions = np.array([
        len(np.intersect1d(true_top_k[i], pred_top_k[i])) / k
        for i in range(y_true.shape[0])
    ])

    return precisions


def calculate_directionality(y_true, y_pred):
    """
    Calculates the proportion of correctly predicted signs (+/-),
    robustly masking out genes where the true change is exactly zero.
    """
    sign_true = np.sign(y_true)
    sign_pred = np.sign(y_pred)

    # Only evaluate directionality on genes that actually have a true non-zero change
    mask = (sign_true != 0)

    # Count how many valid (non-zero) genes exist per row
    valid_counts = mask.sum(axis=1)
    # Prevent division by zero for rows that might be entirely zeros
    valid_counts[valid_counts == 0] = 1

    # A match occurs if the signs are exactly the same AND it's a valid non-zero gene
    matches = (sign_pred == sign_true) & mask

    # Return the accuracy of directionality over valid genes
    return matches.sum(axis=1) / valid_counts

def score(cond_masks_all, y_pred_all, y_ctrl_all, y_cond_all, constellation, model_pick=False):

    cond_masks_unique, inverse = np.unique(cond_masks_all, axis=0, return_inverse=True)
    y_pred_all_ = np.zeros((len(cond_masks_unique), y_pred_all.shape[1]), dtype=float)
    y_ctrl_all_ = np.zeros((len(cond_masks_unique), y_ctrl_all.shape[1]), dtype=float)
    y_cond_all_ = np.zeros((len(cond_masks_unique), y_cond_all.shape[1]), dtype=float)
    for i in range(len(cond_masks_unique)):
        grouped_indices = np.where(inverse == i)[0]
        y_pred_all_[i] = np.mean(y_pred_all[grouped_indices], axis=0)
        y_ctrl_all_[i] = np.mean(y_ctrl_all[grouped_indices], axis=0)
        y_cond_all_[i] = np.mean(y_cond_all[grouped_indices], axis=0)
    cond_masks_all = cond_masks_unique
    y_pred_all = y_pred_all_
    y_ctrl_all = y_ctrl_all_
    y_cond_all = y_cond_all_

    # build np.ndarray to store:
    # the condition names (text)
    # the sorted gene expression values with columns according to the prime gene order
    num_samples = y_pred_all.shape[0]
    num_genes = y_pred_all.shape[1]
    cond_names = np.empty(num_samples, dtype=object)
    yp = np.empty((num_samples, num_genes), dtype=float)
    yt = np.empty((num_samples, num_genes), dtype=float)
    yp_res = np.empty((num_samples, num_genes), dtype=float)
    yt_res = np.empty((num_samples, num_genes), dtype=float)

    for i, (cond_mask, y_pred, y_ctrl, y_cond) in enumerate(zip(cond_masks_all, y_pred_all, y_ctrl_all, y_cond_all)):
        # get the condition name (text)
        cond_mask = tuple(cond_mask.tolist())
        cond_name = constellation.helper("condition", cond_mask)
        # get the prime gene rank
        gene_locs_ranked = constellation.helper("rank", cond_mask)
        # sort the gene expression values according to the prime gene order
        y_ctrl_ranked = y_ctrl[gene_locs_ranked]
        y_cond_ranked = y_cond[gene_locs_ranked]
        y_pred_ranked = y_pred[gene_locs_ranked]
        y_cond_ranked_res = y_cond_ranked - y_ctrl_ranked
        y_pred_ranked_res = y_pred_ranked - y_ctrl_ranked
        # save the results
        cond_names[i] = cond_name
        yp[i] = y_pred_ranked
        yt[i] = y_cond_ranked
        yp_res[i] = y_pred_ranked_res
        yt_res[i] = y_cond_ranked_res

    if model_pick:
        yp_ = yp[:, :20]
        yt_ = yt[:, :20]
        return np.mean(np.square(yp_ - yt_))

    # Save the original results
    # NOTE:
    # We only save the aggregated results here.
    # The results are sorted according to the prime gene order.
    evaluation_results = {
        'cond_names': cond_names,
        'y_pred': yp,
        'y_cond': yt,
    }

    # Calculate the evaluation metrics
    # NOTE: we save the per-condition results, not the mean results.
    for cover in [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, num_genes]:
        yp_ = yp[:, :cover]
        yt_ = yt[:, :cover]
        yp_res_ = yp_res[:, :cover]
        yt_res_ = yt_res[:, :cover]
        mse = np.mean((np.square(yp_ - yt_)), axis=1)
        mae = np.mean(np.abs(yp_ - yt_), axis=1)
        rmse = np.sqrt(mse)

        yp_res_mean = np.mean(yp_res_, axis=1, keepdims=True)
        yt_res_mean = np.mean(yt_res_, axis=1, keepdims=True)
        yp_res_centered = yp_res_ - yp_res_mean
        yt_res_centered = yt_res_ - yt_res_mean
        cov = np.sum(yp_res_centered * yt_res_centered, axis=1) / yp_res_.shape[1]
        std_yp_res = np.std(yp_res_, axis=1, ddof=0)
        std_yt_res = np.std(yt_res_, axis=1, ddof=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            pearson = cov / (std_yp_res * std_yt_res)
            pearson = np.nan_to_num(pearson, nan=0.0, posinf=1.0, neginf=-1.0)

        dot_product = np.sum(yp_res_ * yt_res_, axis=1)
        norm_yp_res = np.linalg.norm(yp_res_, axis=1)
        norm_yt_res = np.linalg.norm(yt_res_, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            cosine = dot_product / (norm_yp_res * norm_yt_res)
            cosine = np.nan_to_num(cosine, nan=0.0)

        # Precision @ K
        precision = calculate_precision_at_k(yt_res, yp_res, cover)

        # Directionality
        directionality = calculate_directionality(yt_res_, yp_res_)

        evaluation_results[f'mse_{cover}'] = mse
        evaluation_results[f'mae_{cover}'] = mae
        evaluation_results[f'rmse_{cover}'] = rmse
        evaluation_results[f'pearson_{cover}'] = pearson
        evaluation_results[f'cosine_{cover}'] = cosine
        # Save new keys
        evaluation_results[f'precision_{cover}'] = precision
        evaluation_results[f'directionality_{cover}'] = directionality

    return evaluation_results
