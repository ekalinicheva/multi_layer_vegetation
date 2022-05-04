import torch
import numpy as np
from scipy.stats import gamma


EPS = 0.0001


# Negative loglikelihood loss
def loss_loglikelihood(pred_pointwise, cloud, params, PCC, args):
    fit_alpha_g, fit_loc_g, fit_beta_g = params["a_g"], params["loc_g"], params["scale_g"]
    fit_alpha_v, fit_loc_v, fit_beta_v = params["a_v"], params["loc_v"], params["scale_v"]

    # We extract heights of every point
    z_all = np.empty((0))
    for current_cloud in cloud:
        # z = current_cloud[2] * args.z_max  # we go back from normalized data
        z = current_cloud[2] * args.plot_radius  # we go back from normalized data
        z_all = np.append(z_all, z)
    z_all = np.asarray(z_all).reshape(-1)

    # we compute the z-likelihood for each point of the cloud that the point belongs to strate1 (ground) or strate2 (medium and high vegetation)
    if fit_loc_g == 0:
        pdf_ground = gamma.pdf(z_all + 1e-2, a=fit_alpha_g, loc=fit_loc_g,
                            scale=fit_beta_g)  # probability for ground points
        pdf_nonground = gamma.pdf(z_all + 1e-2, a=fit_alpha_v, loc=fit_loc_v,
                            scale=fit_beta_v)  # probability for medium and high vegetation points
    else:
        pdf_ground = gamma.pdf(z_all, a=fit_alpha_g, loc=fit_loc_g,
                            scale=fit_beta_g)  # probability for ground points
        pdf_nonground = gamma.pdf(z_all, a=fit_alpha_v, loc=fit_loc_v,
                            scale=fit_beta_v)  # probability for medium and high vegetation points

    p_all_pdf = np.concatenate((pdf_ground.reshape(-1, 1), pdf_nonground.reshape(-1, 1)), 1)
    p_all_pdf = torch.tensor(p_all_pdf)

    p_ground, p_nonground = pred_pointwise[:, :2].sum(1), pred_pointwise[:, 2:].sum(1)

    # print(PCC.is_cuda)
    if PCC.is_cuda:
        p_all_pdf = p_all_pdf.cuda()
        p_ground = p_ground.cuda()
        p_nonground = p_nonground.cuda()

    p_ground_nonground = torch.cat((p_ground.view(-1, 1), p_nonground.view(-1, 1)), 1)
    likelihood = torch.mul(p_ground_nonground, p_all_pdf)
    return - torch.log(likelihood.sum(1)).mean(), likelihood
    

bce_loss = torch.nn.BCELoss(reduction='mean')
bce_loss_none = torch.nn.BCELoss(reduction='none')
def loss_bce(pred_pixels, gt, args, level_loss=False):
    """
    level_loss: wheather we want to obtain losses for different vegetation levels separately
    """
    has_values = ((gt != -1).cpu().numpy() & (pred_pixels!=-1).cpu().numpy())
    return bce_loss(pred_pixels.flatten()[has_values.flatten()], gt.float().flatten()[has_values.flatten()])




# cross_loss = torch.nn.CrossEntropyLoss(reduction='mean', weight=torch.Tensor([0.2, 1, 1, 0.2, 0.5, 1]).cuda(), ignore_index=-1)
# cross_loss = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
def loss_cross_entropy(pred_pointwise, gt_points, args):
    # cross_loss = torch.nn.CrossEntropyLoss(reduction='mean', weight=torch.Tensor([0.05, 0.5, 1, 0.2, 0.7, 0.7]).cuda(), ignore_index=-1)
    if args.n_class == 6:
        cross_loss = torch.nn.CrossEntropyLoss(reduction='mean', weight=torch.Tensor([0.2, 1, 1, 0.2, 1, 1]).cuda(),
                                               ignore_index=-1)
    else:
        cross_loss = torch.nn.CrossEntropyLoss(reduction='mean', weight=torch.Tensor([0.2, 1, 1, 0.2, 1, 1, 1]).cuda(),
                                               ignore_index=-1)
    return cross_loss(pred_pointwise, gt_points)



def loss_entropy(pred_pixels):
    pred_pixels_has_value = pred_pixels != -1
    pred_pixels_selected = pred_pixels.flatten()[pred_pixels_has_value.flatten()]
    return -(pred_pixels_selected * torch.log(pred_pixels_selected + EPS) + (1 - pred_pixels_selected) * torch.log(1 - pred_pixels_selected + EPS)).mean()