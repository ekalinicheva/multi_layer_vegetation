import numpy as np
import torch
import sys
import torch_points3d.core.data_transform as cT
np.set_printoptions(threshold=sys.maxsize)


import matplotlib.pyplot as plt
fig = plt.figure()


def augment(cloud_data):
    """augmentation function
    Does random rotation around z axis and adds Gaussian noise to all the features, except z and return number
    """
    # random rotation around the Z axis
    angle_degrees = np.random.choice(360, 1)[0]
    c, s = np.cos(np.radians(angle_degrees)), np.sin(np.radians(angle_degrees))
    M = np.array(((c, -s), (s, c))) #rotation matrix around axis z with angle "angle", counterclockwise
    cloud_data[:2] = torch.mm(torch.Tensor(M).double(), cloud_data[:2])   #perform the rotation efficiently

    #random gaussian noise to intensity
    sigma, clip = 0.01, 0.03
    # cloud_data[:2] = cloud_data[:2] + np.clip(sigma*np.random.randn(cloud_data[:2].shape[0], cloud_data[:2].shape[1]), a_min=-clip, a_max=clip).astype(np.float32)
    # # # cloud_data[3] = cloud_data[3] + np.clip(sigma*np.random.randn(cloud_data[3].shape[0], cloud_data[3].shape[1]), a_min=-clip, a_max=clip).astype(np.float32)
    cloud_data[3] = cloud_data[3] + torch.Tensor(np.clip(sigma*np.random.randn(cloud_data[3].shape[0]), a_min=-clip, a_max=clip)).double()
    return cloud_data


def cloud_loader(plot_id, dataset, gt_raster, min_coords, train, index_dict, args):
    """
    load a plot and returns points features (normalized xyz + features) and
    ground truth
    INPUT:
    tile_name = string, name of the tile
    train = int, train = 1 iff in the train set
    OUTPUT
    cloud_data, [n x 4] float Tensor containing points coordinates and intensity
    labels, [n] long int Tensor, containing the points semantic labels
    """
    cloud_data = dataset[plot_id].clone()
    xy_min_cylinder = np.asarray(min_coords[plot_id].copy())
    if not args.inference:
        gt = gt_raster[plot_id].copy()
        gt = torch.from_numpy(gt)
    else:
        gt = torch.Tensor([-1])

    xymean = xy_min_cylinder + args.plot_radius

    #normalizing data
    # Z data was already partially normalized during loading
    cloud_data[:, 0] = (cloud_data[:, 0] - xymean[0])/args.plot_radius #x
    cloud_data[:, 1] = (cloud_data[:, 1] - xymean[1])/args.plot_radius #y
    cloud_data[:, 2] = cloud_data[:, 2] / args.plot_radius  # y
    int_max = 57425
    cloud_data[:, 3] = cloud_data[:, 3] / int_max
    # cloud_data[:, 4] = (cloud_data[:, 4] - 1) / (7-1)
    # cloud_data[:, 5] = (cloud_data[:, 5] - 1) / (7 - 1)
    cloud_data[:, 4] = cloud_data[:, 4] / 7
    cloud_data[:, 5] = cloud_data[:, 5] / 7

    if "d" in args.input_feats:
        cloud_data[:, 6] = cloud_data[:, 6] / args.dist_max
    gt_points = cloud_data[:, args.n_input_feats].long()
    cloud_data = cloud_data.T

    if plot_id not in index_dict:
        xy = cloud_data[:2] * args.plot_radius + xymean.reshape(2, 1) + args.mean_dataset.reshape(2, 1)
        xy_min_cylinder_true_coord = xy_min_cylinder.reshape(2, 1) + args.mean_dataset.reshape(2, 1)
        xy_round = torch.floor(xy * (1 / args.pixel_size)) / (1 / args.pixel_size)
        new_xy = ((xy_round - xy_min_cylinder_true_coord) / args.pixel_size)  # no matter what, we always clip by whole coordinates
        ij = new_xy[[1, 0], :].int()  # we swap x and y tp be able to pass to ij discrete coords
        ij[0] = args.diam_pix - 1 - ij[0]
        # index_dict[plot_id] = ij
    else:
        ij = index_dict[plot_id].clone()


    if train and args.data_augmentation and not args.inference:
        cloud_data = augment(cloud_data)
    # print(gt)
    # print(gt_points)

    # return regular_cylinders_norm, gt_tensor
    return cloud_data, gt, gt_points, ij, xy_min_cylinder


def cloud_collate(batch):
    """ Collates a list of dataset samples into a batch list for clouds
    and a single array for labels
    This function is necessary to implement because the clouds have different sizes (unlike for images)
    """
    clouds, labels, labels_points, ij, xy_min_cyl = list(zip(*batch))
    labels = torch.cat(labels, 0)
    labels_points = torch.cat(labels_points, 0)
    return clouds, labels, labels_points, ij, xy_min_cyl
