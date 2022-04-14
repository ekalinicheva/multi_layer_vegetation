import numpy as np
import torch
import sys
from torch_geometric.data import Batch, Data
from sklearn.neighbors import KDTree
import torch_points3d.core.data_transform as cT
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)


def create_grids(dataset, raster_gt, args, train=None):
    cylinders_dataset_by_plot = {}
    cylinder_rasters_gt_by_plot = {}


    if train:
        step = args.sample_grid_size
    elif train is False:
        step = args.regular_grid_size



    origin_counter = 0
    for pl_id, point_cloud in dataset.items():
        # We open the GT raster
        raster_set = raster_gt[pl_id]
        raster, H, W, geo, proj, bands_nb = raster_set
        if len(raster)==4:
            raster = raster[1:]
        x_min_plot, pixel_size, _, y_max_plot, _, _ = geo


        pos = torch.from_numpy(point_cloud[:, :3])
        y = torch.Tensor(point_cloud[:, -1]).type(torch.LongTensor)
        point_cloud_data = Data(pos=pos, y=y)
        point_cloud_data.intensity = torch.from_numpy(point_cloud[:, 3])
        point_cloud_data.return_nb = torch.Tensor(point_cloud[:, 4]).type(torch.FloatTensor)
        if "d" in args.input_feats:
            point_cloud_data.dist_water = torch.Tensor(point_cloud[:, 5]).type(torch.FloatTensor)
        # we create origin indices, so it will be easier to reconstruct parcels from cylinders
        point_cloud_data.origins = torch.Tensor(range(origin_counter, len(pos)+origin_counter)).type(torch.LongTensor)
        origin_counter += len(pos)

        point_cloud_data.pl_id = pl_id
        tree = KDTree(np.asarray(point_cloud_data.pos[:, :-1]), leaf_size=50)
        setattr(point_cloud_data, cT.CylinderSampling.KDTREE_KEY, tree)
        regular_cylinders_new = []
        cropped_rasters = []


        if train is None:
            if pl_id in args.train_pl:
                step = args.sample_grid_size
            else:
                step = args.regular_grid_size

        for y in np.arange(y_max_plot - args.mean_dataset[1] - args.plot_radius, y_max_plot - args.mean_dataset[1] - int(H * args.pixel_size) - step + args.plot_radius, -step):
            for x in np.arange(x_min_plot - args.mean_dataset[0] + args.plot_radius, x_min_plot - args.mean_dataset[0] + int(W * args.pixel_size) - args.plot_radius + step, step):
                cylinder_sampler = cT.CylinderSampling(args.plot_radius - 0.00001, np.asarray([x, y, 0]), align_origin=False)
                cylinder = cylinder_sampler(point_cloud_data)
                # x_min_cylinder1, y_min_cylinder1 = torch.floor(torch.min(cylinder.pos[:, :2], 0)[0] + args.mean_dataset)
                # x_max_cylinder1, y_max_cylinder1 = torch.ceil(torch.max(cylinder.pos[:, :2], 0)[0] + args.mean_dataset)
                # print(x_min_cylinder1, y_min_cylinder1, x_max_cylinder1, y_max_cylinder1)
                # regular_cylinders.append(cylinder)

                nb_pts = len(cylinder.pos)
                if nb_pts > args.min_pts_cylinder:
                    print(nb_pts)
                    # Those are not the coords from the point clouds, but the cropping limits
                    x_min_cylinder, x_max_cylinder = x - args.plot_radius, x + args.plot_radius
                    y_min_cylinder, y_max_cylinder = y - args.plot_radius, y + args.plot_radius
                    if "d" in args.input_feats:
                        new_cylinder = torch.cat((cylinder.pos, cylinder.intensity.reshape(-1, 1), cylinder.return_nb.reshape(-1, 1), cylinder.dist_water.reshape(-1, 1), cylinder.y.reshape(-1, 1),
                                                  torch.full((nb_pts, 1), x_min_cylinder), torch.full((nb_pts, 1), y_min_cylinder),
                                                  torch.full((nb_pts, 1), cylinder.pl_id), cylinder.origins.reshape(-1, 1)), 1)
                    else:
                        new_cylinder = torch.cat((cylinder.pos, cylinder.intensity.reshape(-1, 1), cylinder.return_nb.reshape(-1, 1), cylinder.y.reshape(-1, 1),
                                                  torch.full((nb_pts, 1), x_min_cylinder), torch.full((nb_pts, 1), y_min_cylinder),
                                                  torch.full((nb_pts, 1), cylinder.pl_id), cylinder.origins.reshape(-1, 1)), 1)

                    assert(x_max_cylinder-x_min_cylinder <= args.plot_radius * 2 and y_max_cylinder-y_min_cylinder <= args.plot_radius * 2)
                    x_offset = int(np.floor((x_min_cylinder-(x_min_plot - args.mean_dataset[0]))/args.pixel_size))
                    y_offset = int(np.floor((y_max_plot - args.mean_dataset[1] - y_max_cylinder)/args.pixel_size))
                    # print(y_offset, x_offset)
                    cropped_raster = raster[:, y_offset:y_offset+(int(args.plot_radius*2/args.pixel_size)), x_offset:(x_offset+int(args.plot_radius*2/args.pixel_size))]
                    if cropped_raster.shape[1] == cropped_raster.shape[2] and cropped_raster.shape[1] == args.plot_radius*2/args.pixel_size:
                        regular_cylinders_new.append(new_cylinder)
                        # cylinders_dataset.append(new_cylinder)
                        cropped_rasters.append(cropped_raster)
                        # cylinder_rasters_gt.append(cropped_raster)

        cylinders_dataset_by_plot[pl_id] = regular_cylinders_new
        cylinder_rasters_gt_by_plot[pl_id] = cropped_rasters
        cylinders_dataset = sum(cylinders_dataset_by_plot.values(), [])
        cylinder_rasters_gt = sum(cylinder_rasters_gt_by_plot.values(), [])

    # return cylinders_dataset_by_plot, cylinders_dataset, cylinder_rasters_gt_by_plot, cylinder_rasters_gt
    return cylinders_dataset_by_plot, cylinder_rasters_gt_by_plot