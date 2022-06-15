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
from skimage.morphology import remove_small_holes, remove_small_objects
np.set_printoptions(threshold=sys.maxsize)

name_dem = ["shrub", "understory", "base_canopy", "canopy"]
shapefile_folder = os.path.expanduser("~/DATASETS/sites_placettes/")


def create_dsm(pos_points, pred_points, H, W, geo, path_strata_coverage_pred, pl_id, site_mask):
    str_pl_id = str(pl_id)
    if len(str_pl_id)==3:
        str_pl_id = "S" + str_pl_id[:2] + "P" + str_pl_id[2]
    else:
        str_pl_id = "S0" + str_pl_id[0] + "P" + str_pl_id[1]

    full_placette_df = pd.DataFrame(pos_points, columns=["x", "y", "z"], dtype=np.float64)
    full_placette_df["pred"] = pred_points

    x_min_plot, pixel_size, _, y_max_plot, _, _ = geo

    pixel_size_string = re.sub('[.,]', '', str(pixel_size))

    matrix_plot_height_canopy = np.full((H, W), 0, dtype=np.float32)
    matrix_plot_crown_base_canopy = np.full((H, W), 0, dtype=np.float32)
    matrix_plot_height_understory = np.full((H, W), 0, dtype=np.float32)
    matrix_plot_height_shrub = np.full((H, W), 0, dtype=np.float32)
    matrix_plot_feu = np.full((H, W), 0, dtype=np.int8)
    matrix_plot_conif = np.full((H, W), 0, dtype=np.int8)
    matrix_nodata = np.full((H, W), 0, dtype=np.float32)

    full_placette_df['x_round'] = np.floor(full_placette_df['x'] * (1 / pixel_size)) / (1 / pixel_size)
    full_placette_df['y_round'] = np.floor(full_placette_df['y'] * (1 / pixel_size)) / (1 / pixel_size)

    i = H - 1
    for y in np.arange(y_max_plot - H * pixel_size, y_max_plot, pixel_size):
        j = 0
        for x in np.arange(x_min_plot, x_min_plot + W * pixel_size, pixel_size):
            result = full_placette_df[(full_placette_df['x_round'] == x) & (full_placette_df['y_round'] == y)]
            points_shrub = result[result['pred'] == 1]  # shrub
            points_understory = result[result['pred'] == 2]     # understory
            points_canopy = result[(result['pred'] == 3) | (result['pred'] == 4)]   # canopy # change if aulne is added
            points_feu = result[(result['pred'] == 3) | (result['pred'] == 5)]      # feuillis (chene + aulne(if this class exists))
            points_conif = result[result['pred'] == 4]    # coniferous


            if len(points_shrub) > 1:
                matrix_plot_height_shrub[i][j] = np.max(points_shrub['z'])
            if len(points_understory) > 1:
                matrix_plot_height_understory[i][j] = np.max(points_understory['z'])
            if len(points_canopy) > 1:
                matrix_plot_crown_base_canopy[i][j] = np.min(points_canopy['z'])
                matrix_plot_height_canopy[i][j] = np.max(points_canopy['z'])
            if len(points_feu) > 1:
                matrix_plot_feu[i][j] = 1
            if len(points_conif) > 1:
                matrix_plot_conif[i][j] = 1

            if len(result) > 1:
                matrix_nodata[i][j] = 1

            j += 1
        i -= 1

    dem = np.concatenate(([matrix_plot_height_shrub], [matrix_plot_height_understory], [matrix_plot_crown_base_canopy],  [matrix_plot_height_canopy]), 0)
    layer_thickness = np.concatenate(([matrix_plot_height_shrub], [matrix_plot_height_understory], [matrix_plot_height_canopy - matrix_plot_crown_base_canopy]), 0)
    coverage_feu_conif = np.concatenate(([matrix_plot_feu], [matrix_plot_conif]), 0)

    for d in range(len(dem)-2):

        binary = np.zeros_like((dem[d]), dtype=int)
        binary[dem[d]>0] = 1
        binary = binary.astype(bool)
        binary_after = remove_small_objects(binary, 2, connectivity=1)
        binary_after2 = remove_small_holes(binary_after, 2, connectivity=2)
        dem[d][(binary_after2 == False) & (binary == True)] = 0
        to_change = np.where((binary == False) & (binary_after2 == True))
        to_change = np.concatenate(([to_change[0]], [to_change[1]]), 0).transpose()
        # print(to_change)
        for c in to_change:
            new_value = dem[d, c[0]-1:c[0]+2, c[1]-1:c[1]+2]
            dem[d, c[0], c[1]] = np.mean(new_value[new_value>0])

    print(site_mask.shape)
    print(matrix_nodata.shape)
    matrix_nodata[site_mask != pl_id] = 0
    dem[np.concatenate(([[matrix_nodata], [matrix_nodata], [matrix_nodata], [matrix_nodata]]), 0) == 0] = -1
    layer_thickness[np.concatenate(([[matrix_nodata], [matrix_nodata], [matrix_nodata]]), 0) == 0] = -1
    coverage_feu_conif[np.concatenate(([[matrix_nodata], [matrix_nodata]]), 0) == 0] = -1


    vegetation_coverage = []

    print("Only vegetation-filled pixels total")
    for d in range(len(dem)):
        try:
            height_min = np.round(np.min(dem[d][dem[d] > 0]), 1)
            height_max = np.round(np.max(dem[d][dem[d]>0]), 1)
            height_mean = np.round(np.mean(dem[d][dem[d]>0]), 1)
            height_std = np.round(np.std(dem[d][dem[d]>0]), 1)
        except:
            height_min = 0
            height_max = 0
            height_mean = 0
            height_std = 0

        if d != 3:
            df_list[d] = df_list[d].append({'id': str_pl_id, 'height_min': height_min, 'height_max': height_max, 'height_mean': height_mean, 'height_std': height_std}, ignore_index=True)
        else:
            df_list[d] = df_list[d].append({'id': str_pl_id, 'height_min': height_min, 'height_max': height_max, 'height_mean': height_mean, 'height_std': height_std, "thickness_min": thickness_min, "thickness_max": thickness_max, "thickness_mean": thickness_mean, "thickness_std": thickness_std}, ignore_index=True)
        print(df_list[d])

        print("Height min " + name_dem[d] + " " + str(height_min), "m")
        print("Height max " + name_dem[d] + " " + str(height_max), "m")
        print("Height mean " + name_dem[d] + " " + str(height_mean), "m")
        print("Height std " + name_dem[d] + " " + str(height_std), "m")


        if d==2:
            thickness_mean = np.round(np.mean(layer_thickness[d][layer_thickness[d] > 0]), 1)
            thickness_std = np.round(np.std(layer_thickness[d][layer_thickness[d] > 0]), 1)
            thickness_max = np.round(np.max(layer_thickness[d][layer_thickness[d] > 0]), 1)
            thickness_min = np.round(np.min(layer_thickness[d][layer_thickness[d] > 0.1]), 1)

            print("Thickness mean " + name_dem[d] + " " + str(thickness_mean), "m")
            print("Thickness std " + name_dem[d] + " " + str(thickness_std), "m")
            print("Thickness max " + name_dem[d] + " " + str(thickness_max), "m")
            print("Thickness min " + name_dem[d] + " " + str(thickness_min), "m")

        if d < 3:
            vegetation_coverage_perc = np.round(len(dem[d][dem[d] > 0])/len(dem[d][dem[d] >= 0])*100, 1)
            holes_coverage_perc = np.round(100 - vegetation_coverage_perc, 1)
            vegetation_coverage.append(vegetation_coverage_perc)
            print("Vegetation coverage " + name_dem[d] + " " + str(vegetation_coverage_perc), "%")
            print("Holes coverage " + name_dem[d] + " " + str(holes_coverage_perc), "%")



    coverage_binary = np.zeros_like(dem[:3], dtype=int)
    coverage_binary[dem[:3]>0] = 1
    # coverage_binary[np.concatenate(([[matrix_nodata], [matrix_nodata], [matrix_nodata]]),0)==0] = -1


    absolute_holes = np.sum(coverage_binary, axis=0)
    absolute_holes[absolute_holes > 0] = 1
    absolute_holes[absolute_holes < 0] = -1

    absolute_holes_coverage_perc = np.round(len(absolute_holes[absolute_holes == 0]) / len(absolute_holes[absolute_holes >= 0]) * 100, 1)
    print("Absolute holes " + " " + str(absolute_holes_coverage_perc), "%")

    cov_feu = np.round(len(coverage_feu_conif[0][coverage_feu_conif[0] == 1]) / len(coverage_feu_conif[0][coverage_feu_conif[0] >= 0]) * 100, 1)
    cov_conif = np.round(len(coverage_feu_conif[1][coverage_feu_conif[1] == 1]) / len(coverage_feu_conif[1][coverage_feu_conif[1] >= 0]) * 100, 1)

    surface = np.count_nonzero(site_mask == pl_id) * pixel_size * pixel_size


    df_list[-1] = df_list[-1].append(
        {"id": str_pl_id, 'Veg_Cov_GV': vegetation_coverage[0], 'Veg_Cov_U': vegetation_coverage[1], 'Veg_Cov_O': vegetation_coverage[2],
         'Abs_Holes': absolute_holes_coverage_perc,
         "Cov_Feu": cov_feu, "Cov_Conif": cov_conif, "Surface": surface}, ignore_index=True)

    create_tiff(nb_channels=4,
                new_tiff_name=path_strata_coverage_pred + "final_results/" + "Pl_" + str_pl_id + "_Coverage_height.tif", width=W,
                height=H, datatype=gdal.GDT_Float32, data_array=dem, geotransformation=geo, nodata=-1)

    create_tiff(nb_channels=3,
                new_tiff_name=path_strata_coverage_pred + "final_results/" + "Pl_" + str_pl_id + "_Coverage_thickness.tif", width=W,
                height=H, datatype=gdal.GDT_Float32, data_array=layer_thickness, geotransformation=geo, nodata=-1)

    create_tiff(nb_channels=3,
                new_tiff_name=path_strata_coverage_pred + "final_results/" + "Pl_" + str_pl_id + "_Coverage_binary.tif", width=W,
                height=H, datatype=gdal.GDT_Int16, data_array=coverage_binary, geotransformation=geo, nodata=-1)

    create_tiff(nb_channels=2,
                new_tiff_name=path_strata_coverage_pred + "final_results/" + "Pl_" + str_pl_id + "_Coverage_b1_feu_b2_conif.tif", width=W,
                height=H, datatype=gdal.GDT_Int16, data_array=coverage_feu_conif, geotransformation=geo, nodata=-1)



path = os.path.expanduser("~/DATASETS/Processed_GT/")
tif_name = ["placettes", "sites100m_EPSG2154", "nvx_sites_100m_EPSG2154"]
rasterized_shapes, H_rs, W_rs, geo_rs, proj, _ = open_tiff(shapefile_folder, tif_name[1])
raster_x_min, pixel_size_rs, _, raster_y_max, _, _ = geo_rs

path_result = path + "RESULTS_4_stratum/2022-06-01_155016/"
pl_id_list = np.arange(2, 31)
# pl_id_list = np.asarray([[i*10+1, i*10+2, i*10+3] for i in pl_id_list]).flatten()
print(pl_id_list)

df_GV = pd.DataFrame(columns=["id", "height_min", "height_max", "height_mean", "height_std"])
df_understory = pd.DataFrame(columns=["id", "height_min", "height_max", "height_mean", "height_std"])
df_overstory_bottom = pd.DataFrame(columns=["id", "height_min", "height_max", "height_mean", "height_std"])
df_overstory_top = pd.DataFrame(columns=["id", "height_min", "height_max", "height_mean", "height_std", "thickness_min", "thickness_max", "thickness_mean", "thickness_std"])
df_veg_coverage = pd.DataFrame(columns=["id", "Veg_Cov_GV", "Veg_Cov_U", "Veg_Cov_O", "Abs_Holes", "Cov_Feu", "Cov_Conif", "Surface"])
df_list = [df_GV, df_understory, df_overstory_bottom, df_overstory_top, df_veg_coverage]


for pl_id in pl_id_list:
    print("Plot", str(pl_id))
    if os.path.isfile(path_result + "Pl_" + str(pl_id) + "_predicted_coverage.ply"):
        # pred_raster, H, W, geo, _, _ = open_tiff(path_result, "Pl_" + str(pl_id) + "_predicted_coverage")

        # data_ply_trees_placette1, col_full = open_ply(path_result + "Pl_131_predicted_coverage.ply")
        # data_ply_trees_placette2, col_full = open_ply(path_result + "Pl_132_predicted_coverage.ply")
        # data_ply_trees_placette1 = data_ply_trees_placette1[data_ply_trees_placette1[:, 0]<452232]
        # data_ply_trees_placette2 = data_ply_trees_placette2[data_ply_trees_placette2[:, 0]>=452232]
        #
        # data_ply_trees_placette = np.concatenate((data_ply_trees_placette1, data_ply_trees_placette2), 0)
        # ply_array = np.asarray([tuple(row) for row in data_ply_trees_placette],
        #                        dtype=[("x", "f8"), ("y", "f8"), ("z", "f4"), ("class", "u1"), ("gt", "u1"), ("error", "u1"),
        #                          ("ground", "f4"), ("shrub", "f4"), ("understory", "f4"), ("canopy", "f4"),
        #                          ("red", "u1"), ("green", "u1"), ("blue", "u1")])
        # ply_file = PlyData([PlyElement.describe(ply_array, 'vertex')], text=True)
        # ply_file.write(path_result + "Pl_13_predicted_coverage.ply")
        # exit()

        data_ply_trees_placette, col_full = open_ply(path_result + "Pl_" + str(pl_id) + "_predicted_coverage.ply")
        i, j = np.where(rasterized_shapes == pl_id)
        i_min, i_max, j_min, j_max = np.min(i), np.max(i), np.min(j), np.max(j)
        x_min, x_max, y_min, y_max = raster_x_min + j_min * pixel_size_rs, raster_x_min + (j_max + 1) * pixel_size_rs, \
                                     raster_y_max - (i_max + 1) * pixel_size_rs, raster_y_max - i_min * pixel_size_rs
        clipped_shape = rasterized_shapes[i_min:i_max+1, j_min:j_max+1]

        data_ply_trees_placette = data_ply_trees_placette[(data_ply_trees_placette[:, 0] >= x_min) &
                                                          (data_ply_trees_placette[:, 0] < x_max) &
                                                          (data_ply_trees_placette[:, 1] >= y_min) &
                                                          (data_ply_trees_placette[:, 1] < y_max)]

        pos_points = data_ply_trees_placette[:, :3]
        pred_points = data_ply_trees_placette[:, np.where(col_full == 'class')[0]]

        geo_new = [x_min, pixel_size_rs, 0, y_max, 0, -pixel_size_rs]
        H, W = int((y_max - y_min) / pixel_size_rs), int((x_max - x_min) / pixel_size_rs)

        create_dir(path_result + "final_results/")
        create_dsm(pos_points, pred_points, H, W, geo_new, path_result, pl_id, clipped_shape)


# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter(path_result + "final_results/" + tif_name[0] + ".xlsx", engine='xlsxwriter')

# Write each dataframe to a different worksheet.
df_GV.to_excel(writer, sheet_name='Ground_veg')
df_understory.to_excel(writer, sheet_name='Understory')
df_overstory_bottom.to_excel(writer, sheet_name='Overstory_bottom')
df_overstory_top.to_excel(writer, sheet_name='Overstory_top')
df_veg_coverage.to_excel(writer, sheet_name='Veg_Coverage')


# Write each dataframe to a different worksheet.
df_list[0].to_excel(writer, sheet_name='Ground_veg', index=False)
df_list[1].to_excel(writer, sheet_name='Understory', index=False)
df_list[2].to_excel(writer, sheet_name='Overstory_bottom', index=False)
df_list[3].to_excel(writer, sheet_name='Overstory_top', index=False)
df_list[4].to_excel(writer, sheet_name='Veg_Coverage', index=False)


# Close the Pandas Excel writer and output the Excel file.
writer.save()
