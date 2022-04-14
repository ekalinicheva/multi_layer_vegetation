import os, re
import numpy as np
import pandas as pd
from osgeo import gdal, osr
import geopandas as gpd
from multiprocessing import Pool



from scipy.ndimage.morphology import binary_fill_holes

from plyfile import PlyData, PlyElement


path = os.path.expanduser("~/DATASETS/Processed_GT/")
path_final_legs = os.path.expanduser("~/DATASETS/Processed_GT/final_data_legs/")
selected_placette_folders_final = os.listdir(path_final_legs)
# path_strata_coverage = path + "stratum_coverage_no_clip_extended_double_layer/"
path_strata_coverage = path + "stratum_coverage_no_clip_extended/"

ground_height = 0.5
shrub_height = 1.5
understory_height = 5


pixel_size = 0.25


# If we devide stratum by points height and not by the overall height of an entity, set this one to True, else False
double_layer = False


# Create directory if does not exit, otherwise delete it with all corresponding data and create an empty one
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    # else:
    #     shutil.rmtree(dir_name)
    #     os.makedirs(dir_name)


def open_ply(ply_name):
    assert (os.path.isfile(ply_name))
    with open(ply_name, 'rb') as f:
        plydata = PlyData.read(f)
        col_names = plydata['vertex'].data.dtype.names
        nbr_points = plydata['vertex'].count
        data = np.zeros(shape=[nbr_points, len(col_names)], dtype=np.float64)
        for c in range(len(col_names)):
            data[:, c] = plydata['vertex'].data[col_names[c]]
    return data, col_names


# We create a tiff file with 2 or 3 stratum
def create_tiff(nb_channels, new_tiff_name, width, height, datatype, data_array, geotransformation, nodata=None):
    # We set Lambert 93 projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2154)
    proj = srs.ExportToWkt()
    # We create a datasource
    driver_tiff = gdal.GetDriverByName("GTiff")
    dst_ds = driver_tiff.Create(new_tiff_name, width, height, nb_channels, datatype)
    if nb_channels == 1:
        dst_ds.GetRasterBand(1).WriteArray(data_array)
        if nodata is not None:
            dst_ds.GetRasterBand(1).SetNoDataValue(nodata)
    else:
        for ch in range(nb_channels):
            dst_ds.GetRasterBand(ch + 1).WriteArray(data_array[ch])
            if nodata is not None:
                dst_ds.GetRasterBand(ch + 1).SetNoDataValue(nodata)
    dst_ds.SetGeoTransform(geotransformation)
    dst_ds.SetProjection(proj)
    return dst_ds



def create_database(s):
    # for s in selected_placette_folders_final:
    pl_id = (re.search("Placette_([0-9]*)", s)).group(1)
    print(s)
    ply_trees_placette = path_final_legs + s + "/Pl_" + str(pl_id) + "_final_trees_with_none_legs_clipped.ply"
    ply_trees_placette = path_final_legs + s + "/Pl_" + str(pl_id) + "_final_trees_with_none_legs.ply"
    trees_csv = path_final_legs + s + "/Pl_" + str(pl_id) + "_final_trees_cat.csv"




    if os.path.exists(ply_trees_placette) and os.path.exists(trees_csv) and int(pl_id)==9:
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

        matrix_plot_height_canopy = np.zeros((H, W), dtype=np.float32)
        matrix_plot_crown_base_canopy = np.zeros_like(matrix_plot_height_canopy)
        matrix_plot_height_understory = np.zeros_like(matrix_plot_height_canopy)
        matrix_plot_crown_base_understory = np.zeros_like(matrix_plot_height_canopy)

        geo = [x_min_plot, pixel_size, 0, y_max_plot, 0, -pixel_size]

        # Overall plot coverage
        matrix_binary_pl_ground_sure = np.full((H, W), -1)
        matrix_binary_pl_shrub_sure = np.full((H, W), -1)
        matrix_binary_pl_understory_sure = np.full((H, W), -1)
        matrix_binary_pl_canopy_sure = np.full((H, W), -1)

        i = int((y_max_plot - y_min_plot) / pixel_size) - 1
        for y in np.arange(y_min_plot, y_max_plot, pixel_size):
            j = 0
            for x in np.arange(x_min_plot, x_max_plot, pixel_size):
                # result = full_placette_df[(full_placette_df['x_round'] == x) & (full_placette_df['y_round'] == y)]
                result = nodata_points[(nodata_points['x_round'] == x) & (nodata_points['y_round'] == y)]

                if np.min(result['z']) < 0.1:
                    matrix_binary_pl_ground_sure[i][j] = 1


                if np.max(result['z']) <= shrub_height:

                    matrix_binary_pl_understory_sure[i][j] = 0
                    matrix_binary_pl_canopy_sure[i][j] = 0
                    if np.max(result['z']) >= ground_height:
                        matrix_binary_pl_shrub_sure[i][j] = 1
                    else:
                        matrix_binary_pl_shrub_sure[i][j] = 0
                        matrix_binary_pl_ground_sure[i][j] = 1

                elif np.max(result['z']) <= understory_height:
                    # print(np.max(result['z']))
                    matrix_binary_pl_understory_sure[i][j] = 1
                    matrix_binary_pl_canopy_sure[i][j] = 0
                elif np.max(result['z']) > understory_height:
                    matrix_binary_pl_canopy_sure[i][j] = 1
                j += 1
            i -= 1


        for tree_id in trees_ids:
            print("Tree ", tree_id)

            tree_info = arbres_placette_gdf[arbres_placette_gdf["id_tree"] == tree_id].iloc[0]
            subset_tree = full_placette_df[full_placette_df['tree_id'] == tree_id]


            radius = tree_info['radius'].round(3)  # trunc radius
            houppier = tree_info['houppier'].round(3)  # height of base of tree crown
            diameter = tree_info['diameter_h']

            data_ply_tree = data_ply_trees_placette[data_ply_trees_placette[:, -1] == tree_id]
            tree_coord = data_ply_tree[:, :3]
            xy = tree_coord[:, :2]
            z = tree_coord[:, 2]

            height_real = np.max(z)


            x_min, x_max = np.min(subset_tree['x_round']), np.max(subset_tree['x_round']) + pixel_size
            y_min, y_max = np.min(subset_tree['y_round']), np.max(subset_tree['y_round']) + pixel_size



            if np.isnan(houppier):
                houppier = 0.1
            if houppier == 0:
                houppier = 0.1


            matrix_binary_tree = np.zeros((H, W), dtype=int)
            matrix_binary_tree_bottom = np.zeros((H, W), dtype=int)

            i = int((y_max_plot - y_min)/pixel_size) - 1
            for y in np.arange(y_min, y_max, pixel_size):
                j = int((x_min - x_min_plot)/pixel_size)
                for x in np.arange(x_min, x_max, pixel_size):
                    # print(y, x)
                    result = subset_tree[(subset_tree['x_round'] == x) & (subset_tree['y_round'] == y) & (subset_tree['z'] >= houppier)]  #TODO: add crown or trunk
                    if len(result) > 0:
                        matrix_binary_tree[i][j] = 1
                    # If we devide stratum by points height and not by the overall height of an entity, use this one
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
                matrix_plot_height_canopy[np.where((matrix_binary_tree == 1) & (matrix_plot_height_canopy < height_real))] = height_real
                matrix_plot_crown_base_canopy[np.where((matrix_binary_tree == 1) & ((matrix_plot_crown_base_canopy > houppier) | (matrix_plot_crown_base_canopy == 0)))] = houppier
            elif height_real <= understory_height and height_real > shrub_height:
                matrix_plot_binary_understory += matrix_binary_tree
                matrix_plot_height_understory[np.where((matrix_binary_tree == 1) & (matrix_plot_height_understory < height_real))] = height_real
                matrix_plot_crown_base_understory[np.where((matrix_binary_tree == 1) & ((matrix_plot_crown_base_understory > houppier) | (matrix_plot_crown_base_understory ==0)))] = houppier
            else:
                matrix_plot_binary_shrub += matrix_binary_tree


        path_strata_coverage_pl = path_strata_coverage + "Placette_" + pl_id + "/"
        create_dir(path_strata_coverage_pl)


        matrix_binary_pl_canopy_sure[matrix_plot_binary_canopy >= 1] = 1
        matrix_binary_pl_understory_sure[matrix_plot_binary_understory >= 1] = 1
        matrix_binary_pl_shrub_sure[matrix_plot_binary_shrub >= 1] = 1



        pixel_size_string = re.sub('[.,]', '', str(pixel_size))

        create_tiff(nb_channels=4,
                    new_tiff_name=path_strata_coverage_pl + "Pl_" + str(pl_id) + "_Coverage_sure_" + pixel_size_string + ".tif", width=W,
                    height=H, datatype=gdal.GDT_Int16, data_array=np.concatenate(([matrix_binary_pl_ground_sure],
                                                                                 [matrix_binary_pl_shrub_sure],
                                                                                 [matrix_binary_pl_understory_sure],
                                                                                 [matrix_binary_pl_canopy_sure]),
                                                                                0), geotransformation=geo, nodata= None)

        create_tiff(nb_channels=1,
                    new_tiff_name=path_strata_coverage_pl + "Pl_" + str(pl_id) + "_Coverage_canopy_" + pixel_size_string + ".tif",
                    width=W,
                    height=H, datatype=gdal.GDT_Int16, data_array=matrix_plot_binary_canopy, geotransformation=geo)
        create_tiff(nb_channels=1,
                    new_tiff_name=path_strata_coverage_pl + "Pl_" + str(pl_id) + "_Coverage_understory_" + pixel_size_string + ".tif",
                    width=W,
                    height=H, datatype=gdal.GDT_Int16, data_array=matrix_plot_binary_understory,
                    geotransformation=geo)

        create_tiff(nb_channels=2,
                    new_tiff_name=path_strata_coverage_pl + "Pl_" + str(pl_id) + "_Height_canopy_" + pixel_size_string + ".tif", width=W,
                    height=H, datatype=gdal.GDT_Float32,
                    data_array=np.concatenate(([matrix_plot_crown_base_canopy], [matrix_plot_height_canopy]), 0),
                    geotransformation=geo)
        create_tiff(nb_channels=2,
                    new_tiff_name=path_strata_coverage_pl + "Pl_" + str(pl_id) + "_Height_understory_" + pixel_size_string + ".tif",
                    width=W,
                    height=H, datatype=gdal.GDT_Float32, data_array=np.concatenate(
                ([matrix_plot_crown_base_understory], [matrix_plot_height_understory]), 0), geotransformation=geo)



if __name__ == '__main__':
    # create_database()


    with Pool(10) as p:
        p.map(create_database, selected_placette_folders_final)