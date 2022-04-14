import os, re
import numpy as np
import pandas as pd
from osgeo import gdal, osr
import geopandas as gpd
from utils.confusion_matrix import ConfusionMatrix
from multiprocessing import Pool
from utils.fix_stem_height import fix_height
from utils.useful_functions import create_tiff, create_dir, open_ply, open_tiff
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error


name_dem = ["shrub", "understory", "base_canopy", "canopy"]


path = os.path.expanduser("~/DATASETS/Processed_GT/")
path_final_legs = os.path.expanduser("~/DATASETS/Processed_GT/final_data_legs/")
selected_placette_folders_final = os.listdir(path_final_legs)
# path_strata_coverage = path + "stratum_coverage_no_clip_extended_double_layer/"
path_gt = path + "stratum_coverage_no_clip_extended_height/"



class_names = ["ground", "shrub", "understory", "leaves", "pines", "stems"]
class_names_2d = ["nothing", "vegetation"]
cm = ConfusionMatrix(class_names)
cm_2d_shrub = ConfusionMatrix(class_names_2d)
cm_2d_understory = ConfusionMatrix(class_names_2d)
cm_2d_canopy = ConfusionMatrix(class_names_2d)
all_dem = np.empty((4, 0))


# ground_height = 0.5
# shrub_height = 1.5
# understory_height = 5


pixel_size = 0.5

# regression_logistic_shrub = LogisticRegression()
# regression_logistic_understory = LogisticRegression()
# regression_logistic_canopy = LogisticRegression()


regression_logistic_shrub = RandomForestClassifier(max_depth=4, max_features="auto", n_jobs=4)
regression_logistic_understory = RandomForestClassifier(max_depth=4, max_features="auto", n_jobs=4)
regression_logistic_canopy = RandomForestClassifier(max_depth=4, max_features="auto", n_jobs=4)


# regression_linear_shrub = LinearRegression()
# regression_linear_understory = LinearRegression()
# regression_linear_canopy_base = LinearRegression()
# regression_linear_canopy_top = LinearRegression()



regression_linear_shrub = RandomForestRegressor(max_depth=4, max_features="auto", n_jobs=4)
regression_linear_understory = RandomForestRegressor(max_depth=4, max_features="auto", n_jobs=4)
regression_linear_canopy_base = RandomForestRegressor(max_depth=4, max_features="auto", n_jobs=4)
regression_linear_canopy_top = RandomForestRegressor(max_depth=4, max_features="auto", n_jobs=4)

regression_features = np.empty((0, 19), dtype=float)
regression_gt_height = np.empty((0, 4), dtype=float)
regression_gt_binary = np.empty((0, 3), dtype=int)




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


# def create_database(s):
for pl_id in [1, 10, 12, 13, 14, 16, 17, 18, 19, 2, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 5, 6, 7, 8, 9, 92, 4, 15, 20]:

# for pl_id in [1, 2, 4]:

    s = "Placette_"+str(pl_id)

    # pl_id = (re.search("Placette_([0-9]*)", s)).group(1)
    print(s)
    # ply_trees_placette = path_final_legs + s + "/Pl_" + str(pl_id) + "_final_trees_with_none_legs_clipped.ply"
    ply_trees_placette = path_final_legs + s + "/Pl_" + str(pl_id) + "_final_trees_with_none_legs.ply"
    trees_csv = path_final_legs + s + "/Pl_" + str(pl_id) + "_final_trees_cat.csv"



    regression_features_test = np.empty((0, 19), dtype=float)
    regression_gt_height_test = np.empty((0, 4), dtype=float)
    regression_gt_binary_test = np.empty((0, 3), dtype=int)

    regression_features_test = np.empty((0, 19), dtype=float)
    regression_gt_height_test = np.empty((0, 4), dtype=float)
    regression_gt_binary_test = np.empty((0, 3), dtype=int)

    data_ply_trees_placette, col_full = open_ply(ply_trees_placette)

    path_placette = path + "arbres_par_placette/selected_data/Selected_data_for_placette_" + str(pl_id) + "/"
    path_placette_res_final = path_final_legs + "Placette_" + str(pl_id) + "/"
    create_dir(path_placette_res_final)

    arbres_placette_gdf = gpd.read_file(path_placette + "arbres_placette_" + str(pl_id) + ".shp")


    full_placette_df = pd.DataFrame(data_ply_trees_placette, columns=col_full, dtype=np.float64)
    trees_ids = np.sort(full_placette_df['tree_id'].astype(int).unique())[
                1:-1]  # We get trees ids, except for 0 - nodata and -1 ground points

    gt_height_raster, H, W, geo, _, _ = open_tiff(path_gt + 'Placette_' + str(pl_id) + "/", "Pl_" + str(pl_id) + "_Coverage_height_05")
    gt_binary_raster, _, _, _, _, _ = open_tiff(path_gt + 'Placette_' + str(pl_id) + "/", "Pl_" + str(pl_id) + "_Coverage_sure_05")


    x_min_plot, y_min_plot = np.floor(np.min(data_ply_trees_placette[:, :2], axis=0)).astype(int)
    x_max_plot, y_max_plot = np.ceil(np.max(data_ply_trees_placette[:, :2], axis=0)).astype(int)

    # H, W = int(np.ceil((y_max_plot - y_min_plot)/pixel_size)), int(np.ceil((x_max_plot - x_min_plot)/pixel_size))   #we change datatype to avoid problems with gdal later

    full_placette_df['x_round'] = np.floor(full_placette_df['x'] * (1 / pixel_size)) / (1 / pixel_size)
    full_placette_df['y_round'] = np.floor(full_placette_df['y'] * (1 / pixel_size)) / (1 / pixel_size)


    # Coverage based on annotated trees


    # geo = [x_min_plot, pixel_size, 0, y_max_plot, 0, -pixel_size]


    matrix_no_data = np.zeros((H, W), dtype=int)
    i = int((y_max_plot - y_min_plot) / pixel_size) - 1
    for y in np.arange(y_min_plot, y_max_plot, pixel_size):
        j = 0
        for x in np.arange(x_min_plot, x_max_plot, pixel_size):
            result_full = full_placette_df[(full_placette_df['x_round'] == x) & (full_placette_df['y_round'] == y)]

            # print(result, result_full)
            if len(result_full)>0:
                # z_mean, z_min, z_max, z_std, intensity_mean, return_nb_mean, nbr_pts
                # print(result_full.keys())
                z_mean = result_full['z'].mean()
                z_min = result_full['z'].min()
                z_max = result_full['z'].max()
                z_std = np.std(result_full['z'])
                intensity_mean = result_full['intensity'].mean()
                return_nb_mean = result_full['nrb_returns'].mean()  #wrong name in dataset
                hist = np.histogram(result_full['z'], bins=[0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10, 15, 20, 30])[0]
                # print([z_mean, z_min, z_max, z_std, intensity_mean, return_nb_mean])
                regr = np.concatenate((np.asarray([z_mean, z_min, z_max, z_std, intensity_mean, return_nb_mean]), hist), 0)
                gt_binary_pix = gt_binary_raster[1:, i, j]
                gt_height_pix = gt_height_raster[:, i, j]
                matrix_no_data[i][j] = 1

            else:
                regr = np.zeros((19))
                gt_binary_pix = gt_binary_raster[1:, i, j]
                gt_height_pix = gt_height_raster[:, i, j]
            # print(regr)
            if pl_id not in [4, 15, 20]:
                regression_features = np.concatenate((regression_features, [regr]), 0)


                regression_gt_binary = np.concatenate((regression_gt_binary, [gt_binary_pix]), 0)
                regression_gt_height = np.concatenate((regression_gt_height, [gt_height_pix]), 0)
            else:
                regression_features_test = np.concatenate((regression_features_test, [regr]), 0)

                regression_gt_binary_test = np.concatenate((regression_gt_binary_test, [gt_binary_pix]), 0)
                regression_gt_height_test = np.concatenate((regression_gt_height_test, [gt_height_pix]), 0)

            j += 1
        i -= 1

    if pl_id in [4, 15, 20]:
        if pl_id==4:
            regression_logistic_shrub = RandomForestClassifier(max_depth=4, max_features="auto", n_jobs=4).fit(regression_features[regression_gt_binary[:, 0] != -1], regression_gt_binary[:, 0][regression_gt_binary[:, 0] != -1])
            regression_logistic_understory = RandomForestClassifier(max_depth=4, max_features="auto", n_jobs=4).fit(regression_features[regression_gt_binary[:, 1] != -1], regression_gt_binary[:, 1][regression_gt_binary[:, 1] != -1])
            regression_logistic_canopy = RandomForestClassifier(max_depth=4, max_features="auto", n_jobs=4).fit(regression_features[regression_gt_binary[:, 2] != -1], regression_gt_binary[:, 2][regression_gt_binary[:, 2] != -1])

            regression_linear_shrub = RandomForestRegressor(max_depth=4, max_features="auto", n_jobs=4).fit(regression_features[regression_gt_height[:, 0] != -1], regression_gt_height[:, 0][regression_gt_height[:, 0] != -1])
            regression_linear_understory = RandomForestRegressor(max_depth=4, max_features="auto", n_jobs=4).fit(regression_features[regression_gt_height[:, 1] != -1], regression_gt_height[:, 1][regression_gt_height[:, 1] != -1])
            regression_linear_canopy_base = RandomForestRegressor(max_depth=4, max_features="auto", n_jobs=4).fit(regression_features[regression_gt_height[:, 2] != -1], regression_gt_height[:, 2][regression_gt_height[:, 2] != -1])
            regression_linear_canopy_top = RandomForestRegressor(max_depth=4, max_features="auto", n_jobs=4).fit(regression_features[regression_gt_height[:, 3] != -1], regression_gt_height[:, 3][regression_gt_height[:, 3] != -1])



            # regression_logistic_shrub = LogisticRegression().fit(regression_features[regression_gt_binary[:, 0] != -1], regression_gt_binary[:, 0][regression_gt_binary[:, 0] != -1])
            # regression_logistic_understory = LogisticRegression().fit(regression_features[regression_gt_binary[:, 1] != -1], regression_gt_binary[:, 1][regression_gt_binary[:, 1] != -1])
            # regression_logistic_canopy = LogisticRegression().fit(regression_features[regression_gt_binary[:, 2] != -1], regression_gt_binary[:, 2][regression_gt_binary[:, 2] != -1])
            #
            # regression_linear_shrub = LinearRegression().fit(regression_features[regression_gt_height[:, 0] != -1], regression_gt_height[:, 0][regression_gt_height[:, 0] != -1])
            # regression_linear_understory = LinearRegression().fit(regression_features[regression_gt_height[:, 1] != -1], regression_gt_height[:, 1][regression_gt_height[:, 1] != -1])
            # regression_linear_canopy_base = LinearRegression().fit(regression_features[regression_gt_height[:, 2] != -1], regression_gt_height[:, 2][regression_gt_height[:, 2] != -1])
            # regression_linear_canopy_top = LinearRegression().fit(regression_features[regression_gt_height[:, 3] != -1], regression_gt_height[:, 3][regression_gt_height[:, 3] != -1])
            #

        pred_logistic_shrub = regression_logistic_shrub.predict(regression_features_test)
        pred_logistic_understory = regression_logistic_understory.predict(regression_features_test)
        pred_logistic_canopy = regression_logistic_canopy.predict(regression_features_test)

        pred_linear_shrub = regression_linear_shrub.predict(regression_features_test)
        pred_linear_understory = regression_linear_understory.predict(regression_features_test)
        pred_linear_canopy_base = regression_linear_canopy_base.predict(regression_features_test)
        pred_linear_canopy_top = regression_linear_canopy_top.predict(regression_features_test)

        # pred_logistic = np.concatenate(([pred_logistic_shrub], [pred_logistic_understory], [pred_logistic_canopy]), 0)
        pred_linear = np.concatenate(([pred_linear_shrub], [pred_linear_understory], [pred_linear_canopy_base], [pred_linear_canopy_top]), 0)
        pred_binary = np.concatenate(([pred_logistic_shrub], [pred_logistic_understory], [pred_logistic_canopy]), 0)


        #
        #
        # pred_linear_binary = np.zeros_like(pred_linear)
        # pred_linear_binary[pred_linear > 0] = 1


        print(pred_binary[1])
        cm_2d_shrub.add(
            regression_gt_binary_test[:, 0][
                (matrix_no_data.flatten() == 1) & (regression_gt_binary_test[:, 0] != -1)],
            pred_binary[0][
                (matrix_no_data.flatten() == 1) & (regression_gt_binary_test[:, 0] != -1)])
        cm_2d_understory.add(
            regression_gt_binary_test[:, 1][
                (matrix_no_data.flatten() == 1) & (regression_gt_binary_test[:, 1] != -1)],
            pred_binary[1][
                (matrix_no_data.flatten() == 1) & (regression_gt_binary_test[:, 1] != -1)])
        cm_2d_canopy.add(
            regression_gt_binary_test[:, 2].flatten()[
                (matrix_no_data.flatten() == 1) & (regression_gt_binary_test[:, 2] != -1)],
            pred_binary[2][
                (matrix_no_data.flatten() == 1) & (regression_gt_binary_test[:, 2] != -1)])

        print("Only vegetation-filled pixels")
        for d in range(len(pred_linear)):
            if len(gt_height_raster[d].flatten()[(gt_height_raster[d].flatten()>0.1)&(pred_linear[d]>0.1)])==0:
                mae = 0
            else:
                mae = mean_absolute_error(gt_height_raster[d].flatten()[(gt_height_raster[d].flatten()>0.1)&(pred_linear[d]>0.1)], pred_linear[d][(gt_height_raster[d].flatten()>0.1)&(pred_linear[d]>0.1)])
            mre = np.abs(gt_height_raster[d].flatten() - pred_linear[d])[(gt_height_raster[d].flatten()>0.1)&(pred_linear[d]>0.1)]/(gt_height_raster[d].flatten()[(gt_height_raster[d].flatten()>0.1)&(pred_linear[d]>0.1)])
            # print(np.max(mre))
            list_mre[d] = np.concatenate((list_mre[d], mre), 0)
            list_height[d] = np.concatenate((list_height[d], pred_linear[d][(gt_height_raster[d].flatten()>0.1)&(pred_linear[d]>0.1)]), 0)
            list_height_true[d] = np.concatenate((list_height_true[d], gt_height_raster[d].flatten()[(gt_height_raster[d].flatten()>0.1)&(pred_linear[d]>0.1)]), 0)
            print(name_dem[d])
            print("mean height true", np.round(np.mean(gt_height_raster[d].flatten()[gt_height_raster[d].flatten() > 0.1]), 2), "m")
            print("mean height pred", np.round(np.mean(pred_linear[d][pred_linear[d] > 0]), 2), "m")
            print("MAE " + name_dem[d] + " " + str(np.round(mae, 2)), "m")
            print("MRE " + name_dem[d] + " " + str(np.round(np.mean(mre)*100, 2)), "%")
            print("\n")



print("Only vegetation-filled pixels total")
for d in range(len(pred_linear)):
    # print(list_mre[d])
    mre = np.mean(list_mre[d])
    height_mean = np.mean(list_height[d])
    height_std = np.std(list_height[d])
    print("MAE " + name_dem[d] + " " + str(np.round(mean_absolute_error(list_height_true[d], list_height[d]), 2)), "m")
    print("MRE " + name_dem[d] + " " + str(np.round(mre*100, 1)), "%")
    print("Height mean " + name_dem[d] + " " + str(np.round(height_mean, 1)), "m")
    print("Height std " + name_dem[d] + " " + str(np.round(height_std, 1)), "m")




print("shrub")
cm_2d_shrub.class_IoU()
cm_2d_shrub.overall_accuracy()


print("understory")
cm_2d_understory.class_IoU()
cm_2d_understory.overall_accuracy()


print("canopy")
cm_2d_canopy.class_IoU()
cm_2d_canopy.overall_accuracy()