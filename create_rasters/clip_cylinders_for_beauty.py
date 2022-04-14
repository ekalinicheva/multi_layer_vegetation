import numpy as np
import torch
import sys, re
from osgeo import gdal
from torch_geometric.data import Data
from sklearn.neighbors import KDTree
import torch_points3d.core.data_transform as cT
from utils.useful_functions import open_tiff, open_ply, create_tiff, create_ply
from plyfile import PlyData, PlyElement

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)


path = "/home/ign.fr_ekalinicheva/DATASETS/Processed_GT/"
path_result = path + "RESULTS_4_stratum/2022-02-27_110951_ours/"
path_gt = path + "stratum_coverage_no_clip_extended_height/"
pl_id = 15
epoch = 100

# pred_raster, H, W, geo, _, _ = open_tiff(path_result, "Pl_" + str(pl_id) + "_predicted_coverage_ep_" + str(epoch))
pred_raster, H, W, geo, _, _ = open_tiff(path_result, "Pl_" + str(pl_id) + "_predicted_coverage_ep_" + str(epoch))

data_ply_trees_placette, col_full = open_ply(path_result + "Pl_" + str(pl_id) + "_predicted_coverage_ep_" + str(epoch) + ".ply")
point_cloud, _ = open_ply(path_result + "Pl_" + str(pl_id) + "_predicted_coverage_ep_" + str(epoch) + ".ply")

# gt_height_raster, _, _, geo_gt, _, _ = open_tiff(path_gt + 'Placette_' + str(pl_id) + "/", "Pl_" + str(pl_id) + "_Coverage_height_05")

print(col_full)

raster, H, W, geo, proj, bands_nb = open_tiff(path_result, "Pl_" + str(pl_id) + "_predicted_coverage_ep_" + str(epoch))
x_min_plot, pixel_size, _, y_max_plot, _, _ = geo


pred_points = data_ply_trees_placette[:, np.where(col_full == 'class')[0]]


pos = torch.from_numpy(data_ply_trees_placette[:, :3])
y = torch.Tensor(data_ply_trees_placette[:, np.where(col_full == 'class')[0]]).type(torch.LongTensor)
red = torch.Tensor(data_ply_trees_placette[:, np.where(col_full == 'red')[0]]).type(torch.LongTensor)
green = torch.Tensor(data_ply_trees_placette[:, np.where(col_full == 'green')[0]]).type(torch.LongTensor)
blue = torch.Tensor(data_ply_trees_placette[:, np.where(col_full == 'blue')[0]]).type(torch.LongTensor)
point_cloud_data = Data(pos=pos, y=y, red=red, green=green, blue=blue)
# we create origin indices, so it will be easier to reconstruct parcels from cylinders

tree = KDTree(np.asarray(point_cloud_data.pos[:, :-1]), leaf_size=50)
setattr(point_cloud_data, cT.CylinderSampling.KDTREE_KEY, tree)
regular_cylinders_new = []
cropped_rasters = []

plot_radius = 5
pixel_size = 0.5

step = 5
i=1
for y in np.arange(y_max_plot - plot_radius, y_max_plot - int(H * pixel_size) - step + plot_radius, -step):
    for x in np.arange(x_min_plot + plot_radius, x_min_plot + int(W * pixel_size) - plot_radius + step, step):
        cylinder_sampler = cT.CylinderSampling(plot_radius - 0.00001, np.asarray([x, y, 0]), align_origin=False)
        cylinder = cylinder_sampler(point_cloud_data)

        nb_pts = len(cylinder.pos)

        # Those are not the coords from the point clouds, but the cropping limits
        x_min_cylinder, x_max_cylinder = x - plot_radius, x + plot_radius
        y_min_cylinder, y_max_cylinder = y - plot_radius, y + plot_radius
        new_cylinder = torch.cat((cylinder.pos, cylinder.y.reshape(-1, 1), cylinder.red.reshape(-1, 1), cylinder.green.reshape(-1, 1), cylinder.blue.reshape(-1, 1)), 1)

        assert(x_max_cylinder-x_min_cylinder <= plot_radius * 2 and y_max_cylinder-y_min_cylinder <= plot_radius * 2)
        x_offset = int(np.floor((x_min_cylinder-(x_min_plot))/pixel_size))
        y_offset = int(np.floor((y_max_plot  - y_max_cylinder)/pixel_size))
        # print(y_offset, x_offset)
        cropped_raster = raster[:, y_offset:y_offset+(int(plot_radius*2/pixel_size)), x_offset:(x_offset+int(plot_radius*2/pixel_size))]
        pixel_size_string = re.sub('[.,]', '', str(pixel_size))
        geo1 = [x_min_cylinder, pixel_size, 0, y_max_cylinder, 0, -pixel_size]


        create_tiff(nb_channels=4,
                    new_tiff_name=path_result+ "cylinders/" + "Pl_" + str(
                        pl_id) + "_Coverage_height_" + pixel_size_string + "_pred_cylinder_"+str(i)+".tif", width=int(plot_radius*2/pixel_size),
                    height=int(plot_radius*2/pixel_size), datatype=gdal.GDT_Float32, data_array=cropped_raster, geotransformation=geo1, nodata=None)


        ply_array = np.ones(
            cylinder.pos.shape[0], dtype=[("x", "f8"), ("y", "f8"), ("z", "f4"), ("class", "u1"),
                                 ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        )


        ply_array["x"] = cylinder.pos[:, 0]
        ply_array["y"] = cylinder.pos[:, 1]
        ply_array["z"] = cylinder.pos[:, 2]
        ply_array["class"] = cylinder.y.reshape(-1)
        ply_array["red"] = cylinder.red.reshape(-1)
        ply_array["green"] = cylinder.green.reshape(-1)
        ply_array["blue"] = cylinder.blue.reshape(-1)

        ply_file = PlyData([PlyElement.describe(ply_array, 'vertex')], text=True)
        ply_file.write(path_result+ "cylinders/" + "Pl_" + str(pl_id) + "_predicted_coverage_" + str(i) + ".ply")

        i += 1
