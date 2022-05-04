import os, re
import numpy as np
import pandas as pd
from osgeo import gdal, osr
import geopandas as gpd
from multiprocessing import Pool
from utils.fix_stem_height import fix_height
from utils.useful_functions import create_tiff, create_dir, open_ply


from scipy.ndimage.morphology import binary_fill_holes

from plyfile import PlyData, PlyElement


path = os.path.expanduser("~/DATASETS/Processed_GT/")
path_final_legs = os.path.expanduser("~/DATASETS/Processed_GT/final_data_legs/")
selected_placette_folders_final = os.listdir(path_final_legs)
# path_strata_coverage = path + "stratum_coverage_no_clip_extended_double_layer/"
path_strata_coverage = path + "stratum_coverage_no_clip_extended_height/"

ground_height = 0.5
shrub_height = 1.5
understory_height = 5


pixel_size = 0.5


# If we devide stratum by points height and not by the overall height of an entity, set this one to True, else False
double_layer = False


def create_database(s):
# for s in selected_placette_folders_final:
    pl_id = (re.search("Placette_([0-9]*)", s)).group(1)
    print(pl_id)
    # ply_trees_placette = path_final_legs + s + "/Pl_" + str(pl_id) + "_final_trees_with_none_legs_clipped.ply"
    ply_trees_placette = path_final_legs + s + "/Pl_" + str(pl_id) + "_final_trees_with_none_legs.ply"
    trees_csv = path_final_legs + s + "/Pl_" + str(pl_id) + "_final_trees_cat.csv"


    if ply_trees_placette and os.path.exists(trees_csv) and int(pl_id)==6:
        data_ply_trees_placette, col_full = open_ply(ply_trees_placette)

        path_placette = path + "arbres_par_placette/selected_data/Selected_data_for_placette_" + str(pl_id) + "/"
        path_placette_res_final = path_final_legs + "Placette_" + str(pl_id) + "/"
        create_dir(path_placette_res_final)

        arbres_placette_gdf = gpd.read_file(path_placette + "arbres_placette_" + pl_id + ".shp")


        full_placette_df = pd.DataFrame(data_ply_trees_placette, columns=col_full, dtype=np.float64)
        trees_ids = np.sort(full_placette_df['tree_id'].astype(int).unique())[
                    1:-1]  # We get trees ids, except for 0 - nodata and -1 ground points


        x_min_plot, y_min_plot = np.floor(np.min(data_ply_trees_placette[:, :2], axis=0)).astype(int)
        x_max_plot, y_max_plot = np.ceil(np.max(data_ply_trees_placette[:, :2], axis=0)).astype(int)

        H, W = int(np.ceil((y_max_plot - y_min_plot)/pixel_size)), int(np.ceil((x_max_plot - x_min_plot)/pixel_size))   #we change datatype to avoid problems with gdal later


        full_placette_df['x_round'] = np.floor(full_placette_df['x'] * (1 / pixel_size)) / (1 / pixel_size)
        full_placette_df['y_round'] = np.floor(full_placette_df['y'] * (1 / pixel_size)) / (1 / pixel_size)

        nodata_points = full_placette_df[(full_placette_df["tree_id"] == 0) | (full_placette_df["tree_id"] == trees_ids[-1])]  # points that weren't assigned to any tree, but are located in the plot


        # Coverage based on annotated trees
        matrix_plot_binary_canopy = np.zeros((H, W), dtype=int)
        matrix_plot_binary_understory = np.zeros_like(matrix_plot_binary_canopy)
        matrix_plot_binary_shrub = np.zeros_like(matrix_plot_binary_canopy)

        matrix_plot_classes_canopy = np.full((H, W), -1, dtype=int)
        matrix_plot_classes_canopy_associated_height = np.full((H, W), -1, dtype=np.float32)    #we create this raster just to be able to produce correct canopy occupation gt


        matrix_plot_height_canopy = np.full((H, W), -1, dtype=np.float32)
        matrix_plot_crown_base_canopy = np.full((H, W), -1, dtype=np.float32)
        matrix_plot_height_understory = np.full((H, W), -1, dtype=np.float32)
        matrix_plot_height_shrub = np.full((H, W), -1, dtype=np.float32)

        geo = [x_min_plot, pixel_size, 0, y_max_plot, 0, -pixel_size]

        # Overall plot coverage
        matrix_binary_pl_ground_sure = np.full((H, W), -1)
        matrix_binary_pl_shrub_sure = np.full((H, W), -1)
        matrix_binary_pl_understory_sure = np.full((H, W), -1)
        matrix_binary_pl_canopy_sure = np.full((H, W), -1)

        i = int((y_max_plot - y_min_plot) / pixel_size) - 1
        for y in np.arange(y_min_plot, y_max_plot, pixel_size).round(1):
            j = 0
            for x in np.arange(x_min_plot, x_max_plot, pixel_size).round(1):
                # result = full_placette_df[(full_placette_df['x_round'] == x) & (full_placette_df['y_round'] == y)]
                result = nodata_points[(nodata_points['x_round'] == x) & (nodata_points['y_round'] == y)]
                result_full = full_placette_df[(full_placette_df['x_round'] == x) & (full_placette_df['y_round'] == y)]

                # print(result, result_full)
                if len(result)>0 or len(result_full)>0:
                    if np.min(result['z']) < 0.1:
                        matrix_binary_pl_ground_sure[i][j] = 1
                    if len(result) > 0:
                        if np.max(result['z']) <= shrub_height:

                            matrix_binary_pl_understory_sure[i][j] = 0
                            matrix_binary_pl_canopy_sure[i][j] = 0
                            if np.max(result['z']) >= ground_height:
                                matrix_binary_pl_shrub_sure[i][j] = 1
                                matrix_plot_height_shrub[i][j] = np.max(result['z'])
                            else:
                                matrix_binary_pl_shrub_sure[i][j] = 0
                                matrix_binary_pl_ground_sure[i][j] = 1


                        elif np.max(result['z']) <= understory_height :
                            matrix_binary_pl_understory_sure[i][j] = 1
                            matrix_binary_pl_canopy_sure[i][j] = 0
                            matrix_plot_height_understory[i][j] = np.max(result['z'])

                        elif np.max(result['z']) > understory_height :
                            matrix_binary_pl_canopy_sure[i][j] = 1
                            matrix_plot_height_canopy[i][j] = np.max(result['z'])

                    elif len(result_full)>0:
                        if np.max(result_full['z'])<= understory_height:
                            matrix_binary_pl_understory_sure[i][j] = 1
                            matrix_binary_pl_canopy_sure[i][j] = 0
                            matrix_plot_height_understory[i][j] = np.max(result_full['z'])

                        elif np.max(result_full['z']) > understory_height :
                            matrix_binary_pl_canopy_sure[i][j] = 1
                            matrix_plot_height_canopy[i][j] = np.max(result_full['z'])
                j += 1
            i -= 1





        for tree_id in trees_ids:
            print("Tree ", tree_id)

            tree_info = arbres_placette_gdf[arbres_placette_gdf["id_tree"] == tree_id].iloc[0]
            subset_tree = full_placette_df[full_placette_df['tree_id'] == tree_id]


            radius = tree_info['radius'].round(3)  # trunc radius
            houppier = tree_info['houppier'].round(3)  # height of base of tree crown
            diameter = tree_info['diameter_h']
            category = tree_info['cat']

            data_ply_tree = data_ply_trees_placette[data_ply_trees_placette[:, -1] == tree_id]
            tree_coord = data_ply_tree[:, :3]
            z = tree_coord[:, 2]
            height_real = np.max(z)
            if height_real>understory_height and not np.isnan(houppier):
                houppier_calc = fix_height(subset_tree.to_numpy(), houppier, radius)
                houppier = houppier_calc

            if np.isnan(houppier):
                houppier = 0.1

            if houppier == 0:
                houppier = 0.1


            x_min, x_max = np.min(subset_tree['x_round']).round(1), np.max(subset_tree['x_round']).round(1) + pixel_size
            y_min, y_max = np.min(subset_tree['y_round']).round(1), np.max(subset_tree['y_round']).round(1) + pixel_size

            matrix_binary_tree = np.zeros((H, W), dtype=int)
            matrix_binary_tree_bottom = np.zeros((H, W), dtype=int)
            # print(np.arange(x_min, x_max, pixel_size, dtype=np.float64))
            # print(np.arange(y_min, y_max, pixel_size, dtype=np.float64))
            i = int((y_max_plot - y_min)/pixel_size) - 1
            for y in np.arange(y_min, y_max, pixel_size).round(1):
                j = int((x_min - x_min_plot)/pixel_size)
                for x in np.arange(x_min, x_max, pixel_size).round(1):
                    result = subset_tree[(subset_tree['x_round'] == x) & (subset_tree['y_round'] == y) & (subset_tree['z'] >= houppier)]  #TODO: add crown or trunk
                    result_no_data = nodata_points[(nodata_points['x_round'] == x) & (nodata_points['y_round'] == y)]
                    if len(result) > 0:
                        matrix_binary_tree[i][j] = 1
                        if height_real>understory_height: #HERE
                            no_data = [result_no_data[result_no_data['z']>understory_height]] if len(result_no_data[result_no_data['z']>understory_height])>0 else [1000]
                            # print(no_data)
                            matrix_plot_height_canopy[i][j] = max(np.max(result['z']), matrix_plot_height_canopy[i][j]) #HERE
                            # print(np.max(result['z']))
                            # print(max(matrix_plot_classes_canopy_associated_height[i][j], np.max(result_no_data['z'])))
                            matrix_plot_classes_canopy_associated_height[i][j] = np.max(result['z']) if np.max(result['z'])> max(matrix_plot_classes_canopy_associated_height[i][j], np.max(result_no_data['z'])) else max(matrix_plot_classes_canopy_associated_height[i][j], np.max(result_no_data['z']))# we create this raster just to be able to produce correct canopy occupation gt
                            matrix_plot_classes_canopy[i][j] = category if np.max(result['z'])==matrix_plot_classes_canopy_associated_height[i][j] else matrix_plot_classes_canopy[i][j]


                            if matrix_plot_crown_base_canopy[i][j] != -1:
                                if houppier < 1 and np.min(result['z']) < 4:
                                    matrix_plot_crown_base_canopy[i][j] = np.max([-1,
                                                                                  matrix_plot_crown_base_canopy[i][j]])
                                else:
                                    matrix_plot_crown_base_canopy[i][j] = np.min([np.min(result['z']),
                                                                     matrix_plot_crown_base_canopy[i][j], np.min(no_data)])
                            else:
                                if houppier < 1 and np.min(result['z']) < 4:
                                    matrix_plot_crown_base_canopy[i][j] =np.max([-1,
                                                                                  matrix_plot_crown_base_canopy[i][j]])
                                else:
                                    matrix_plot_crown_base_canopy[i][j] = min([np.min(result['z']), np.min(no_data)]) #HERE
                        elif height_real <= understory_height and height_real > shrub_height:
                            matrix_plot_height_understory[i][j] = max(np.max(result['z']),
                                                                     matrix_plot_height_understory[i][j])
                        elif height_real <= shrub_height:
                            matrix_plot_height_shrub[i][j] = max(np.max(result['z']),
                                                                         matrix_plot_height_shrub[i][j])
                    # If we divide stratum by points height and not by the overall height of an entity, use this one
                    if double_layer:
                        if np.min(result['z'])<=understory_height:
                            matrix_binary_tree_bottom[i][j] = 1
                    j += 1
                i -= 1

            # matrix_binary_tree = binary_fill_holes(matrix_binary_tree).astype(int)

            if height_real > understory_height:
                matrix_plot_binary_canopy += matrix_binary_tree
                if double_layer:
                    matrix_plot_binary_understory += matrix_binary_tree_bottom
            elif height_real <= understory_height and height_real > shrub_height:
                matrix_plot_binary_understory += matrix_binary_tree
            else:
                matrix_plot_binary_shrub += matrix_binary_tree


        path_strata_coverage_pl = path_strata_coverage + "Placette_" + pl_id + "/"
        create_dir(path_strata_coverage_pl)


        matrix_binary_pl_canopy_sure[matrix_plot_binary_canopy >= 1] = 1
        matrix_binary_pl_understory_sure[matrix_plot_binary_understory >= 1] = 1
        matrix_binary_pl_shrub_sure[matrix_plot_binary_shrub >= 1] = 1

        matrix_plot_height_canopy[matrix_binary_pl_canopy_sure==0]=0
        matrix_plot_crown_base_canopy[matrix_binary_pl_canopy_sure==0]=0
        matrix_plot_height_understory[matrix_binary_pl_understory_sure==0]=0
        matrix_plot_height_shrub[matrix_binary_pl_shrub_sure==0]=0


        matrix_plot_classes_canopy[matrix_plot_classes_canopy == 23] = 4  # pine class
        # matrix_plot_classes_canopy[(matrix_plot_classes_canopy == 3) | (matrix_plot_classes_canopy == 7)] = 5    # aulne and charme class
        # matrix_plot_classes_canopy[(matrix_plot_classes_canopy != 4) & (matrix_plot_classes_canopy != 5) & (matrix_plot_classes_canopy != 7) & (
        #             matrix_plot_classes_canopy != 23) & (matrix_plot_classes_canopy != -1)] = 3    # chene + others
        matrix_plot_classes_canopy[(matrix_plot_classes_canopy != 4) & (matrix_plot_classes_canopy != -1)] = 3    # chene + others
        matrix_plot_classes_canopy[matrix_binary_pl_canopy_sure == 0] = 0  # hole in the canopy

        matrix_plot_classes_canopy[(matrix_plot_classes_canopy == 0) & (matrix_binary_pl_understory_sure==1)] = 2  # understory class
        matrix_plot_classes_canopy[(matrix_plot_classes_canopy == 0) & (matrix_binary_pl_shrub_sure==1)] = 1  # ground vegetation class



        pixel_size_string = re.sub('[.,]', '', str(pixel_size))

        create_tiff(nb_channels=4,
                    new_tiff_name=path_strata_coverage_pl + "Pl_" + str(pl_id) + "_Coverage_sure_" + pixel_size_string + ".tif", width=W,
                    height=H, datatype=gdal.GDT_Int16, data_array=np.concatenate(([matrix_binary_pl_ground_sure],
                                                                                 [matrix_binary_pl_shrub_sure],
                                                                                 [matrix_binary_pl_understory_sure],
                                                                                 [matrix_binary_pl_canopy_sure]),
                                                                                0), geotransformation=geo, nodata= None)
        create_tiff(nb_channels=4,
                    new_tiff_name=path_strata_coverage_pl + "Pl_" + str(pl_id) + "_Coverage_height_" + pixel_size_string + ".tif", width=W,
                    height=H, datatype=gdal.GDT_Float32, data_array=np.concatenate(([matrix_plot_height_shrub],
                                                                                 [matrix_plot_height_understory],
                                                                                 [matrix_plot_crown_base_canopy],
                                                                                 [matrix_plot_height_canopy]),
                                                                                0), geotransformation=geo, nodata= None)
        # geo = [x_min_plot, pixel_size/2, 0, y_max_plot, 0, -pixel_size/2]
        # create_tiff(nb_channels=1,
        #             new_tiff_name=path_strata_coverage_pl + "Pl_" + str(pl_id) + "_Coverage_canopy_6_classes_" + re.sub('[.,]', '', str(pixel_size/2)) + ".tif", width=W*2,
        #             height=H*2, datatype=gdal.GDT_Int16, data_array=matrix_plot_classes_canopy.repeat(2, axis=0).repeat(2, axis=1), geotransformation=geo, nodata= None)

        # # Different pixel size
        # pixel_size_1 = 1
        #
        # sure_rasters=np.concatenate(([matrix_binary_pl_ground_sure], [matrix_binary_pl_shrub_sure],
        #                                                                                  [matrix_binary_pl_understory_sure],
        #                                                                                  [matrix_binary_pl_canopy_sure]),
        #                                                                                 0)
        # sure_height_rasters = np.concatenate(([matrix_plot_height_shrub],
        #                              [matrix_plot_height_understory],
        #                              [matrix_plot_crown_base_canopy],
        #                              [matrix_plot_height_canopy]),
        #                             0)
        #
        # new_sure_raster_1 = np.full((4, int(np.ceil(H/2)), int(np.ceil(W/2))), -1, dtype=int)
        # new_height_raster_1 = np.full((4, int(np.ceil(H/2)), int(np.ceil(W/2))), -1, dtype=float)
        #
        # for r in range(len(sure_rasters)):
        #     for i in range(0,H,2):
        #         for j in range(0,W,2):
        #             binary_small = sure_rasters[r, i:i+2, j:j+2]
        #             height_small = sure_height_rasters[r, i:i + 2, j:j + 2]
        #             height_small_new = height_small[height_small>0]
        #             # if len(height_small_new)>0:
        #             #     new_height_raster_1[r, int(i / 2), int(j / 2)] = height_small[height_small > 0].mean()
        #             # else:
        #             #     new_height_raster_1[r, int(i / 2), int(j / 2)] = height_small[height_small > 0].max()
        #             new_sure_raster_1[r, int(i/2), int(j/2)] = binary_small.max()
        #             new_height_raster_1[r, int(i / 2), int(j / 2)] = height_small.max()
        #
        # pixel_size_string = re.sub('[.,]', '', str(pixel_size_1))
        #
        # geo = [x_min_plot, pixel_size_1, 0, y_max_plot, 0, -pixel_size_1]
        #
        # create_tiff(nb_channels=4,
        #             new_tiff_name=path_strata_coverage_pl + "Pl_" + str(pl_id) + "_Coverage_sure_" + pixel_size_string + ".tif", width=int(np.ceil(W/2)),
        #             height=int(np.ceil(H/2)), datatype=gdal.GDT_Int16, data_array=new_sure_raster_1, geotransformation=geo, nodata= None)
        # create_tiff(nb_channels=4,
        #             new_tiff_name=path_strata_coverage_pl + "Pl_" + str(pl_id) + "_Coverage_height_" + pixel_size_string + ".tif", width=int(np.ceil(W/2)),
        #             height=int(np.ceil(H/2)), datatype=gdal.GDT_Float32, data_array=new_height_raster_1, geotransformation=geo, nodata= None)
        #
        #
        # # Pixel size 2
        # pixel_size_2 = 2
        #
        #
        # new_sure_raster_2 = np.full((4, int(np.ceil(H / 4)), int(np.ceil(W / 4))), -1, dtype=int)
        # new_height_raster_2 = np.full((4, int(np.ceil(H / 4)), int(np.ceil(W / 4))), -1, dtype=float)
        #
        # for r in range(len(sure_rasters)):
        #     for i in range(0, H, 4):
        #         for j in range(0, W, 4):
        #             binary_small = sure_rasters[r, i:i + 4, j:j + 4]
        #             height_small = sure_height_rasters[r, i:i + 4, j:j + 4]
        #             # height_small_new = height_small[height_small > 0]
        #             # if len(height_small_new)>0:
        #             #     new_height_raster_1[r, int(i / 2), int(j / 2)] = height_small[height_small > 0].mean()
        #             # else:
        #             #     new_height_raster_1[r, int(i / 2), int(j / 2)] = height_small[height_small > 0].max()
        #             new_sure_raster_2[r, int(i / 4), int(j / 4)] = binary_small.max()
        #             new_height_raster_2[r, int(i / 4), int(j / 4)] = height_small.max()
        #
        # pixel_size_string = re.sub('[.,]', '', str(pixel_size_2))
        #
        # geo = [x_min_plot, pixel_size_2, 0, y_max_plot, 0, -pixel_size_2]
        #
        # create_tiff(nb_channels=4,
        #             new_tiff_name=path_strata_coverage_pl + "Pl_" + str(
        #                 pl_id) + "_Coverage_sure_" + pixel_size_string + ".tif", width=int(np.ceil(W / 4)),
        #             height=int(np.ceil(H / 4)), datatype=gdal.GDT_Int16, data_array=new_sure_raster_2, geotransformation=geo,
        #             nodata=None)
        # create_tiff(nb_channels=4,
        #             new_tiff_name=path_strata_coverage_pl + "Pl_" + str(
        #                 pl_id) + "_Coverage_height_" + pixel_size_string + ".tif", width=int(np.ceil(W / 4)),
        #             height=int(np.ceil(H / 4)), datatype=gdal.GDT_Float32, data_array=new_height_raster_2,
        #             geotransformation=geo, nodata=None)



if __name__ == '__main__':
    # create_database()
    with Pool(10) as p:
        p.map(create_database, selected_placette_folders_final)