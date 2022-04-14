'''
    Create final rasters and the mesh.
'''

import numpy as np
import pandas as pd
import re, os, sys
from osgeo import gdal
from sklearn.metrics import mean_squared_error, mean_absolute_error
from plyfile import PlyData, PlyElement
from utils.useful_functions import create_tiff, create_dir, open_ply, open_tiff
from utils.confusion_matrix import ConfusionMatrix
from skimage.morphology import remove_small_holes, remove_small_objects
np.set_printoptions(threshold=sys.maxsize)

name_dem = ["shrub", "understory", "base_canopy", "canopy"]


shrub_mre = np.empty((0))
understory_mre = np.empty((0))
canopy_base_mre = np.empty((0))
canopy_top_mre = np.empty((0))
shrub_height = np.empty((0))
understory_height = np.empty((0))
canopy_base_height = np.empty((0))
canopy_top_height = np.empty((0))
list_mre = [shrub_mre, understory_mre, canopy_base_mre, canopy_top_mre]
list_height = [shrub_height, understory_height, canopy_base_height, canopy_top_height]
list_height_true = list_height.copy()

def create_dsm(pos_points, pred_points, pred_raster, gt_height_raster, gt_points, geo, path_strata_coverage_pred, pl_id):
    full_placette_df = pd.DataFrame(pos_points, columns=["x", "y", "z"], dtype=np.float64)
    full_placette_df["pred"] = pred_points

    x_min_plot, pixel_size, _, y_max_plot, _, _ = geo
    H, W = pred_raster[0].shape

    pixel_size_string = re.sub('[.,]', '', str(pixel_size))

    matrix_plot_height_canopy = np.full((H, W), 0, dtype=np.float32)
    matrix_plot_crown_base_canopy = np.full((H, W), 0, dtype=np.float32)
    matrix_plot_height_understory = np.full((H, W), 0, dtype=np.float32)
    matrix_plot_height_shrub = np.full((H, W), 0, dtype=np.float32)
    matrix_nodata = np.full((H, W), 0, dtype=np.float32)

    full_placette_df['x_round'] = np.floor(full_placette_df['x'] * (1 / pixel_size)) / (1 / pixel_size)
    full_placette_df['y_round'] = np.floor(full_placette_df['y'] * (1 / pixel_size)) / (1 / pixel_size)

    i = H - 1
    for y in np.arange(y_max_plot - H * pixel_size, y_max_plot, pixel_size):
        j = 0
        for x in np.arange(x_min_plot, x_min_plot + W * pixel_size, pixel_size):
            result = full_placette_df[(full_placette_df['x_round'] == x) & (full_placette_df['y_round'] == y)]
            points_shrub = result[result['pred'] == 1] # shrub
            points_understory = result[result['pred'] == 2] # understory
            points_canopy = result[(result['pred'] == 3) | (result['pred'] == 4)] # canopy # change if aulne is added

            if len(points_shrub)>0:
                matrix_plot_height_shrub[i][j] = np.max(points_shrub['z'])
            if len(points_understory)>0:
                matrix_plot_height_understory[i][j] = np.max(points_understory['z'])
            if len(points_canopy)>0:
                matrix_plot_crown_base_canopy[i][j] = np.min(points_canopy['z'])
                matrix_plot_height_canopy[i][j] = np.max(points_canopy['z'])

            if len(result)>0:
                matrix_nodata[i][j]=1

            j += 1
        i -= 1

    dem = np.concatenate(([matrix_plot_height_shrub], [matrix_plot_height_understory], [matrix_plot_crown_base_canopy],  [matrix_plot_height_canopy]), 0)
    layer_thickness = np.concatenate(([matrix_plot_height_shrub], [matrix_plot_height_understory], [matrix_plot_height_canopy - matrix_plot_crown_base_canopy]), 0)


    for d in range(len(dem)-2):

        binary = np.zeros_like((dem[d]), dtype=int)
        binary[dem[d]>0] = 1
        binary = binary.astype(bool)
        binary_after = remove_small_objects(binary, 2, connectivity=1)
        binary_after2 = remove_small_holes(binary_after, 2, connectivity=2)
        dem[d][(binary_after2 == False)&(binary == True)] = 0
        to_change = np.where((binary == False) & (binary_after2 == True))
        to_change = np.concatenate(([to_change[0]], [to_change[1]]), 0).transpose()
        # print(to_change)
        for c in to_change:
            new_value = dem[d,c[0]-1:c[0]+2, c[1]-1:c[1]+2]
            dem[d,c[0], c[1]] = np.mean(new_value[new_value>0])

    dem[np.concatenate(([[matrix_nodata], [matrix_nodata], [matrix_nodata], [matrix_nodata]]),0)==0] = -1
    layer_thickness[np.concatenate(([[matrix_nodata], [matrix_nodata], [matrix_nodata]]),0)==0] = -1

    print("Only vegetation-filled pixels total")
    for d in range(len(dem)):

        height_mean = np.round(np.mean(dem[d][dem[d]>0]), 1)
        height_std = np.round(np.std(dem[d][dem[d]>0]), 1)
        height_max = np.round(np.max(dem[d][dem[d]>0]), 1)
        height_min = np.round(np.min(dem[d][dem[d]>0]), 1)

        print("Height mean " + name_dem[d] + " " + str(height_mean), "m")
        print("Height std " + name_dem[d] + " " + str(height_std), "m")
        print("Height max " + name_dem[d] + " " + str(height_max), "m")
        print("Height min " + name_dem[d] + " " + str(height_min), "m")


        if d==2:
            thickness_mean = np.round(np.mean(layer_thickness[d][layer_thickness[d] > 0]), 1)
            thickness_std = np.round(np.std(layer_thickness[d][layer_thickness[d] > 0]), 1)
            thickness_max = np.round(np.max(layer_thickness[d][layer_thickness[d] > 0]), 1)
            thickness_min = np.round(np.min(layer_thickness[d][layer_thickness[d] > 0]), 1)

            print("Thickness mean " + name_dem[d] + " " + str(thickness_mean), "m")
            print("Thickness std " + name_dem[d] + " " + str(thickness_std), "m")
            print("Thickness max " + name_dem[d] + " " + str(thickness_max), "m")
            print("Thickness min " + name_dem[d] + " " + str(thickness_min), "m")

        if d<3:
            vegetation_coverage_perc = np.round(np.count_nonzero(dem[d])/len(dem[d][dem[d] >= 0])*100, 1)
            holes_coverage_perc = 100 - vegetation_coverage_perc
            print("Vegetation coverage " + name_dem[d] + " " + str(vegetation_coverage_perc), "%")
            print("Holes coverage " + name_dem[d] + " " + str(holes_coverage_perc), "%")




    create_tiff(nb_channels=4,
                new_tiff_name=path_strata_coverage_pred + "Pl_" + str(pl_id) + "_Coverage_height_" + pixel_size_string + "_pred.tif", width=W,
                height=H, datatype=gdal.GDT_Float32, data_array=dem, geotransformation=geo, nodata=None)

    coverage_binary = np.zeros_like(dem[:3], dtype=int)
    coverage_binary[dem[:3]>0] = 1
    # coverage_binary[np.concatenate(([[matrix_nodata], [matrix_nodata], [matrix_nodata]]),0)==0] = -1


    create_tiff(nb_channels=3,
                new_tiff_name=path_strata_coverage_pred + "Pl_" + str(pl_id) + "_Coverage_binary_" + pixel_size_string + "_pred.tif", width=W,
                height=H, datatype=gdal.GDT_Int16, data_array=coverage_binary, geotransformation=geo, nodata=None)

    absolute_holes = np.sum(coverage_binary, axis=0)
    absolute_holes[absolute_holes > 0] = 1
    absolute_holes[absolute_holes < 0] = -1

    absolute_holes_coverage_perc = np.round(np.count_nonzero(absolute_holes==0) / len(absolute_holes[absolute_holes >= 0]) * 100, 1)
    print("Absolute holes " + " " + str(absolute_holes_coverage_perc), "%")

    return dem, matrix_nodata


path = os.path.expanduser("~/DATASETS/Processed_GT/")
path_result = path + "RESULTS_4_stratum/2022-02-27_110951_ours/"
#2022-02-26_215820_ss8192
#2022-02-28_082049_m05
#2022-03-03_192430_noel_m0
#2022-02-17_225300_no_2D
#2022-02-28_005116_pix025
#2022-02-27_110951_ours
#2022-02-25_220612_ours
#2022-02-26_032038_pix1
# 2022-03-01_165840_r10
# 2022-03-01_201229_no2D
# 2022-03-02_114246_R2
# 2022-02-28_182033_ours
path_gt = path + "stratum_coverage_no_clip_extended_height/"
pl_id_list = [4, 15, 20]
epoch = 100


for pl_id in pl_id_list:
    print("Plot", str(pl_id))
    pred_raster, H, W, geo, _, _ = open_tiff(path_result, "Pl_" + str(pl_id) + "_predicted_coverage_ep_" + str(epoch))
    data_ply_trees_placette, col_full = open_ply(path_result + "Pl_" + str(pl_id) + "_predicted_coverage_ep_" + str(epoch) + ".ply")


    pos_points = data_ply_trees_placette[:, :3]
    gt_points = data_ply_trees_placette[:, 4]
    pred_points = data_ply_trees_placette[:, np.where(col_full == 'class')[0]]


    dem, matrix_no_data = create_dsm(pos_points, pred_points, pred_raster, geo, path_result, pl_id)

