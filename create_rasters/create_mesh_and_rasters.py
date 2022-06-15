'''
    Create final rasters and the mesh. And computes some accuracy statistics if GT is provided.
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

def create_dsm(pos_points, pred_points, pred_raster, geo, path_strata_coverage_pred, pl_id, gt_height_raster=None, gt_binary=None):
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
            points_canopy = result[(result['pred'] == 3) | (result['pred'] == 4)] # canopy

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

    len_previous = 0
    ply_array_mesh_all = None
    triangle_mesh_ply_all = None
    for d in range(len(dem)):
        triangle_mesh = []
        point_cloud = []
        point_cloud_mesh = []
        color_face = []
        dem_flattened = dem[d].flatten()
        y = y_max_plot
        for i in range(H):
            x = x_min_plot
            for j in range(W):
                index = j + W * i
                z = dem_flattened[index]
                point_cloud_mesh.append([x, y, z])
                if z != -1:
                    point_cloud.append([x, y, z])
                    if j != W-1 and i != H-1:
                        index2 = index + 1
                        index3 = index + W
                        index4 = index + W + 1

                        z2 = dem_flattened[index2]
                        z3 = dem_flattened[index3]
                        z4 = dem_flattened[index4]

                        # if (z != 0 and z2 != 0 and z3 != 0 if d in [2,3] else True) and z2 != -1 and z3 != -1:
                        # if z != 0 and z2 != 0 and z3 != 0 and z2 != -1 and z3 != -1:
                        if True:
                            triangle = [index, index2, index3]
                            triangle_mesh.append(triangle)
                            color_face.append(np.mean([z, z2, z3]))
                        # if (z2 != 0 and z3 != 0 and z4 != 0 if d in [2,3] else True) and z2 != -1 and z3 != -1 and z4 != -1:
                        # if z2 != 0 and z3 != 0 and z4 != 0 and z2 != -1 and z3 != -1 and z4 != -1:
                        if True:
                            triangle = [index2, index3, index4]
                            triangle_mesh.append(triangle)
                            color_face.append(np.mean([z4, z2, z3]))
                x += pixel_size
            y -= pixel_size

        point_cloud_mesh = np.asarray(point_cloud_mesh)
        color_face = np.asarray(color_face)

        triangle_mesh = np.asarray(triangle_mesh)

        triangle_mesh_ply = np.ones(len(triangle_mesh),
                              dtype=[('vertex_indices', 'i4', (3,)), ('vertex_colors', 'f4')])
        triangle_mesh_ply_to_add = np.ones(len(triangle_mesh),
                              dtype=[('vertex_indices', 'i4', (3,)), ('vertex_colors', 'f4')])
        triangle_mesh_ply['vertex_indices'] = triangle_mesh
        triangle_mesh_ply['vertex_colors'] = color_face




        ply_array_mesh = np.ones(len(point_cloud_mesh), dtype=[("x", "f8"), ("y", "f8"), ("z", "f4"), ("c", "f4")])
        # print(len(ply_array_mesh))
        ply_array_mesh["x"] = point_cloud_mesh[:, 0]
        ply_array_mesh["y"] = point_cloud_mesh[:, 1]
        ply_array_mesh["z"] = point_cloud_mesh[:, 2]
        ply_array_mesh["c"] = point_cloud_mesh[:, 2]

        if d>1:
            triangle_mesh_ply_to_add['vertex_indices'] = triangle_mesh + len_previous
            triangle_mesh_ply_to_add['vertex_colors'] = color_face

            if ply_array_mesh_all is None:
                ply_array_mesh_all = ply_array_mesh.copy()
                triangle_mesh_ply_all = triangle_mesh_ply.copy()
            else:
                ply_array_mesh_all = np.concatenate((ply_array_mesh_all, ply_array_mesh), 0)
                triangle_mesh_ply_all = np.concatenate((triangle_mesh_ply_all, triangle_mesh_ply_to_add), 0)
            len_previous += len(ply_array_mesh)


        mesh_ply_file = PlyData([PlyElement.describe(ply_array_mesh, 'vertex'), PlyElement.describe(triangle_mesh_ply, 'faces')], text=True)
        mesh_ply_file.write(path_strata_coverage_pred + "Pl_" + str(pl_id) + "_Coverage_height_"+name_dem[d]+"_el_" + pixel_size_string + "_pred_mesh.ply")


        if d==3:
            # Link canopy to the base
            triangle_mesh_connection = []
            color_face_connection = []
            y = y_max_plot
            for i in range(H):
                x = x_min_plot
                for j in range(W):
                    index = j + W * i
                    z = dem_flattened[index]
                    index3 = None
                    index4 = None
                    index5 = None
                    index6 = None
                    if z != -1 and z != 0:
                        indices = [index - W, index + 1, index + W, index - 1]
                        # if 0 in dem_flattened[indices] or -1 in dem_flattened[indices]:
                        if j != W - 1:

                            if dem_flattened[index + 1] not in [-1, 0]:
                                if i == 0 or i==H-1:
                                    index3 = index + 1
                                    index4 = index + 1 - len(ply_array_mesh)

                                elif dem_flattened[indices[0]] in [-1, 0] or dem_flattened[indices[2]] in [-1, 0]:
                                    index3 = index + 1
                                    index4 = index + 1 - len(ply_array_mesh)


                        if i != H - 1:

                            if dem_flattened[index + W] not in [-1, 0]:
                                if j == 0 or j == W - 1:
                                    index5 = index + W
                                    index6 = index + W - len(ply_array_mesh)
                                elif dem_flattened[indices[1]] in [-1, 0] or dem_flattened[indices[3]] in [-1, 0]:
                                    index5 = index + W
                                    index6 = index + W - len(ply_array_mesh)
                    if index3 is not None:
                        # print("happiness")
                        index2 = index - len(ply_array_mesh)

                        triangle = [index, index2, index3]
                        triangle_mesh_connection.append(triangle)
                        color_face_connection.append(np.mean(dem_flattened[triangle]))

                        triangle = [index3, index4, index2]
                        triangle_mesh_connection.append(triangle)
                        color_face_connection.append(np.mean(dem_flattened[triangle]))
                    if index5 is not None:
                        # print("happiness")
                        index2 = index - len(ply_array_mesh)

                        triangle = [index, index2, index5]
                        triangle_mesh_connection.append(triangle)
                        color_face_connection.append(np.mean(dem_flattened[triangle]))

                        triangle = [index5, index6, index2]
                        triangle_mesh_connection.append(triangle)
                        color_face_connection.append(np.mean(dem_flattened[triangle]))
                    x += pixel_size
                y -= pixel_size
            color_face_connection = np.asarray(color_face_connection)
            triangle_mesh_connection = np.asarray(triangle_mesh_connection)

            triangle_mesh_ply_to_add = np.ones(len(triangle_mesh_connection),
                                               dtype=[('vertex_indices', 'i4', (3,)), ('vertex_colors', 'f4')])
            triangle_mesh_ply_to_add['vertex_indices'] = triangle_mesh_connection + len_previous
            triangle_mesh_ply_to_add['vertex_colors'] = color_face_connection

            ply_array_mesh_all = np.concatenate((ply_array_mesh_all, ply_array_mesh), 0)
            triangle_mesh_ply_all = np.concatenate((triangle_mesh_ply_all, triangle_mesh_ply_to_add), 0)

        # len_previous += len(ply_array_mesh)

    mesh_ply_file = PlyData(
        [PlyElement.describe(ply_array_mesh_all, 'vertex'), PlyElement.describe(triangle_mesh_ply_all, 'faces')], text=True)
    mesh_ply_file.write(path_strata_coverage_pred + "Pl_" + str(pl_id) + "_Coverage_height_all_el_" + pixel_size_string + "_pred_mesh.ply")


    create_tiff(nb_channels=4,
                new_tiff_name=path_strata_coverage_pred + "Pl_" + str(pl_id) + "_Coverage_height_" + pixel_size_string + "_pred.tif", width=W,
                height=H, datatype=gdal.GDT_Float32, data_array=dem, geotransformation=geo, nodata=None)

    coverage_binary = np.zeros_like(dem[:3], dtype=int)
    coverage_binary[dem[:3]>0] = 1
    coverage_binary[np.concatenate(([[matrix_nodata], [matrix_nodata], [matrix_nodata]]),0)==0] = -1

    coverage_binary_error = coverage_binary.copy()

    coverage_binary_error[(coverage_binary != gt_binary[[0,1,3]])&(gt_binary[[0,1,3]]!=-1)&(coverage_binary!=-1)]=2
    # coverage_binary_error[np.where(gt_binary==-1)] = coverage_binary[np.where(gt_binary==-1)]


    create_tiff(nb_channels=3,
                new_tiff_name=path_strata_coverage_pred + "Pl_" + str(pl_id) + "_Coverage_binary_" + pixel_size_string + "_pred.tif", width=W,
                height=H, datatype=gdal.GDT_Int16, data_array=coverage_binary, geotransformation=geo, nodata=None)

    create_tiff(nb_channels=3,
                new_tiff_name=path_strata_coverage_pred + "Pl_" + str(pl_id) + "_Coverage_binary_error_" + pixel_size_string + "_pred.tif", width=W,
                height=H, datatype=gdal.GDT_Int16, data_array=coverage_binary_error, geotransformation=geo, nodata=None)


    print("Only vegetation-filled pixels")
    for d in range(len(dem)):
        if len(gt_height_raster[d].flatten()[(gt_height_raster[d].flatten()>0.1)&(dem[d].flatten()>0.1)])==0:
            mae = 0
        else:
            mae = mean_absolute_error(gt_height_raster[d].flatten()[(gt_height_raster[d].flatten()>0.1)&(dem[d].flatten()>0.1)], dem[d].flatten()[(gt_height_raster[d].flatten()>0.1)&(dem[d].flatten()>0.1)])
        mre = np.abs(gt_height_raster[d].flatten() - dem[d].flatten())[(gt_height_raster[d].flatten()>0.1)&(dem[d].flatten()>0.1)]/(gt_height_raster[d].flatten()[(gt_height_raster[d].flatten()>0.1)&(dem[d].flatten()>0.1)])
        # print(np.max(mre))
        list_mre[d] = np.concatenate((list_mre[d], mre), 0)
        list_height[d] = np.concatenate((list_height[d], dem[d].flatten()[(gt_height_raster[d].flatten()>0.1)&(dem[d].flatten()>0.1)]), 0)
        list_height_true[d] = np.concatenate((list_height_true[d], gt_height_raster[d].flatten()[(gt_height_raster[d].flatten()>0.1)&(dem[d].flatten()>0.1)]), 0)
        print(name_dem[d])
        print("mean height true", np.round(np.mean(gt_height_raster[d].flatten()[gt_height_raster[d].flatten() > 0.1]), 2), "m")
        print("mean height pred", np.round(np.mean(dem[d].flatten()[dem[d].flatten() > 0]), 2), "m")
        print("MAE " + name_dem[d] + " " + str(np.round(mae, 2)), "m")
        print("MRE " + name_dem[d] + " " + str(np.round(np.mean(mre)*100, 2)), "%")
        print("\n")

    return dem, matrix_nodata


path = os.path.expanduser("~/DATASETS/Processed_GT/")
path_result = path + "RESULTS_4_stratum/2022-02-27_110951_ours/"
path_gt = path + "stratum_coverage_no_clip_extended_height/"
pl_id_list = [4, 15, 20]
epoch = 100
inference = False

print(path_result)

class_names = ["ground", "shrub", "understory", "leaves", "pines", "stems"]
class_names_2d = ["nothing", "vegetation"]
cm = ConfusionMatrix(class_names)
cm_2d_shrub = ConfusionMatrix(class_names_2d)
cm_2d_understory = ConfusionMatrix(class_names_2d)
cm_2d_canopy = ConfusionMatrix(class_names_2d)
all_dem = np.empty((4, 0))

for pl_id in pl_id_list:
    print("Plot", str(pl_id))
    pred_raster, H, W, geo, _, _ = open_tiff(path_result, "Pl_" + str(pl_id) + "_predicted_coverage_ep_" + str(epoch))
    data_ply_trees_placette, col_full = open_ply(path_result + "Pl_" + str(pl_id) + "_predicted_coverage_ep_" + str(epoch) + ".ply")
    pos_points = data_ply_trees_placette[:, :3]
    pred_points = data_ply_trees_placette[:, np.where(col_full == 'class')[0]]
    if inference is False:
        gt_height_raster, _, _, geo_gt, _, _ = open_tiff(path_gt + 'Placette_' + str(pl_id) + "/", "Pl_" + str(pl_id) + "_Coverage_height_05")
        gt_binary = np.zeros_like(gt_height_raster[:, :H, :W], dtype=int)
        gt_binary[gt_height_raster[:, :H, :W]>0] = 1
        gt_binary[gt_height_raster[:, :H, :W]==-1] = -1
        dem, matrix_no_data = create_dsm(pos_points, pred_points, pred_raster, geo, path_result, pl_id,
                                         gt_height_raster[:, :H, :W], gt_binary)
        gt_points = data_ply_trees_placette[:, 4]


    else:
        dem, matrix_no_data = create_dsm(pos_points, pred_points, pred_raster, geo, path_result, pl_id)


    dem_binary = np.zeros_like(dem, dtype=int)
    dem_binary[dem>0] = 1

    if inference is False:
        cm.add(gt_points, pred_points)
        cm_2d_shrub.add(gt_binary[0].flatten()[(matrix_no_data.flatten()==1) & (gt_binary[0].flatten()!=-1)], dem_binary[0].flatten()[(matrix_no_data.flatten()==1) & (gt_binary[0].flatten()!=-1)])
        cm_2d_understory.add(gt_binary[1].flatten()[(matrix_no_data.flatten()==1) & (gt_binary[1].flatten()!=-1)], dem_binary[1].flatten()[(matrix_no_data.flatten()==1) & (gt_binary[1].flatten()!=-1)])
        cm_2d_canopy.add(gt_binary[3].flatten()[(matrix_no_data.flatten()==1) & (gt_binary[3].flatten()!=-1)], dem_binary[3].flatten()[(matrix_no_data.flatten()==1) & (gt_binary[3].flatten()!=-1)])


print("Only vegetation-filled pixels total")
for d in range(len(dem)):
    # print(list_mre[d])
    mre = np.mean(list_mre[d])
    height_mean = np.mean(list_height_true[d])
    height_std = np.std(list_height_true[d])
    print("MAE " + name_dem[d] + " " + str(np.round(mean_absolute_error(list_height_true[d], list_height[d]), 2)), "m")
    print("MRE " + name_dem[d] + " " + str(np.round(mre*100, 1)), "%")
    print("Height mean " + name_dem[d] + " " + str(np.round(height_mean, 1)), "m")
    print("Height std " + name_dem[d] + " " + str(np.round(height_std, 1)), "m")


cm.class_IoU()
cm.overall_accuracy()

# print(np.round(cm.CM, 0).astype(int))
cm.plot_matrix(path_result)


print("shrub")
cm_2d_shrub.class_IoU()
cm_2d_shrub.overall_accuracy()


print("understory")
cm_2d_understory.class_IoU()
cm_2d_understory.overall_accuracy()


print("canopy")
cm_2d_canopy.class_IoU()
cm_2d_canopy.overall_accuracy()