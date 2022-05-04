import numpy as np
import torch
from torch_scatter import scatter_max, scatter_mean


def project_to_2d(pred_pointwise, pred_pointwise_b, ij_batch, PCC, args):
    """
    We do all the computation to obtain
    pred_pl - [Bx4] prediction vector for the plot
    scores -  [(BxN)x2] probas_ground_nonground that a point belongs to stratum 1 or stratum 2
    """
    index_batches = []
    index_group = []
    batches_len = []
    restore_coords = []

    # we project 3D points to 2D plane
    # We use torch scatter to process
    for b in range(len(pred_pointwise_b)):
        ij = ij_batch[b]
        if PCC.is_cuda:
            ij = ij.cuda()
        unique, index = torch.unique(ij.T, dim=0, return_inverse=True)
        index_b = torch.full(torch.unique(index).size(), b)

        if PCC.is_cuda:
            index = index.cuda()
            index_b = index_b.cuda()
            unique = unique.cuda()
        index = index + np.asarray(batches_len).sum()
        index_batches.append(index.type(torch.LongTensor))
        index_group.append(index_b.type(torch.LongTensor))
        batches_len.append(torch.unique(index).size(0))
        unique_coord = torch.cat((index_b.reshape(-1, 1), unique), 1)
        restore_coords.append(unique_coord)
    index_batches = torch.cat(index_batches)
    index_group = torch.cat(index_group)
    restore_coords_batch = torch.cat(restore_coords)
    pred_rasters = torch.full((len(pred_pointwise_b), args.diam_pix, args.diam_pix, args.nb_stratum), -1, dtype=torch.float)

    if PCC.is_cuda:
        index_batches = index_batches.cuda()
        index_group = index_group.cuda()
        pred_rasters = pred_rasters.cuda()

    # We get rid of stem class for projection and we combine the predictions
    # pred_pointwise = torch.cat((pred_pointwise[:, 1:3], pred_pointwise[:, 3:-1].sum(1).reshape(-1, 1)), 1)
    pred_pointwise = torch.cat((pred_pointwise[:, 1:3], pred_pointwise[:, 3:-1].max(1)[0].reshape(-1, 1)), 1)
    pred_pointwise[pred_pointwise > 1] = 1


    pixel_max = scatter_max(pred_pointwise.T, index_batches)[0].T

    pred_rasters[restore_coords_batch[:, 0], restore_coords_batch[:, 1], restore_coords_batch[:, 2]] = pixel_max

    pred_rasters_reshaped = pred_rasters.permute(0, 3, 1, 2).reshape(-1, args.diam_pix, args.diam_pix)

    # pred_pl = scatter_mean(pixel_max.T, index_group).T

    return pixel_max, pred_rasters_reshaped

