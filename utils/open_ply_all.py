import os, re
import numpy as np
import pandas as pd
from osgeo import gdal, osr
from utils.useful_functions import open_tiff, open_ply
import warnings
from utils.fix_stem_height import fix_height
warnings.simplefilter(action='ignore')




def open_ply_all(args, selected_placettes=None):
    print("open ply")
    annotated_points_count = 0
    points_count = 0

    path_clouds = args.path + args.folder_clouds
    path_strata_coverage = args.path + args.folder_gt_rasters

    selected_placette_folders_final = os.listdir(path_clouds)


    if "d" in args.input_feats:
        path_water = args.path + args.folder_water + "/water_clipped.tif"

    dataset = {}
    gt_rasters_dataset = {}

    if selected_placettes is not None:
        placette_id_list = selected_placettes
    else:
        placette_id_list = args.pl_id_list

    all_points = None
    for s in selected_placette_folders_final:
        pl_id = int((re.search("Placette_([0-9]*)", s)).group(1))
        print(s)
        ply_trees_placette = path_clouds + s + "/Pl_" + str(pl_id) + "_final_data_xyzinr.ply"
        if args.pixel_size%1!=0:
            pixel_size_string = re.sub('[.,]', '', str(args.pixel_size))
        else:
            pixel_size_string = str(int(args.pixel_size))

        if pl_id in placette_id_list:
            data_ply_trees_placette, col_full = open_ply(ply_trees_placette)


            path_strata_coverage_pl = path_strata_coverage + "Placette_" + str(pl_id) + "/"
            name_strata_coverage_pl = "Pl_" + str(pl_id) + "_Coverage_sure_" + pixel_size_string
            trees_csv = path_clouds + s + "/Pl_" + str(pl_id) + "_trees_params.csv"
            trees_data_csv = pd.read_csv(trees_csv, index_col=False)


            gt_raster = open_tiff(path_strata_coverage_pl, name_strata_coverage_pl)
            gt_rasters_dataset[pl_id] = gt_raster
            H, W, geo = gt_raster[1:4]
            trees_placette_df = pd.DataFrame(data_ply_trees_placette, columns=col_full)

            data_final_df = trees_placette_df.join(trees_data_csv.set_index('tree_id'), on='tree_id')
            data_final_df = data_final_df.fillna(value=-1)
            data_final_df["tree_height_class"] = -1
            annotated_points_count += len(data_final_df[data_final_df['tree_id'] != 0])

            points_count += len(data_final_df)

            data_final_df.loc[data_final_df['z'] == 0, 'tree_height_class'] = 0
            data_final_df.loc[data_final_df['height'] > 5, 'tree_height_class'] = 3
            data_final_df.loc[(data_final_df['height'] > 1.5) & (data_final_df['height'] <= 5), 'tree_height_class'] = 2
            data_final_df.loc[(data_final_df['height'] >-1) & (data_final_df['height'] <= 1.5), 'tree_height_class'] = 1

            if args.n_class == 6:
                data_final_df.loc[(data_final_df['height'] > 5) & (data_final_df['cat'] == 23), 'tree_height_class'] = 4    # pine class

            if args.n_class == 7:
                data_final_df.loc[(data_final_df['height'] > 5) & (data_final_df['cat'] == 23), 'tree_height_class'] = 4    # pine class
                data_final_df.loc[(data_final_df['height'] > 5) & ((data_final_df['cat'] == 3) | (data_final_df['cat'] == 7)), 'tree_height_class'] = 5     # aulne and charme class

            # We add stem class
            for tree_id in np.unique(data_final_df["tree_id"])[2:]:
                houppier = data_final_df[data_final_df["tree_id"] == tree_id].iloc[0]['crown_h']
                tree_height = data_final_df[data_final_df["tree_id"] == tree_id].iloc[0]['height']
                if houppier > 0 and tree_height > 5:
                    radius = data_final_df[data_final_df["tree_id"] == tree_id].iloc[0]['trunc_r']
                    subset_tree = data_final_df[data_final_df["tree_id"] == tree_id].to_numpy()
                    # we fix the height of the base of the tree crown, as the in-situ measure were often approximate
                    houppier_calc = fix_height(subset_tree, houppier, radius)

                    data_final_df.loc[
                        (data_final_df['tree_id'] == tree_id) & (data_final_df['z'] <= houppier_calc), 'tree_height_class'] = args.n_class - 1

            if "d" in args.input_feats:

                ulx, uly, lrx, lry = geo[0], geo[3], geo[0] + W * args.pixel_size, geo[3] - H * args.pixel_size

                driver_mem = gdal.GetDriverByName('MEM')
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(2154)    # Lambert-93 projection
                proj = srs.ExportToWkt()


                # output = "/home/ign.fr_ekalinicheva/DATASETS/Processed_GT/search_for_water/water_dist_reshaped_placette.tif"
                to_add_margin_meters = 2000
                to_add_margin_pix = int(to_add_margin_meters / args.pixel_size) * 2
                ds_water_clipped = gdal.Translate('', path_water, xRes=args.pixel_size, yRes=args.pixel_size, resampleAlg="bilinear",
                                              projWin=[ulx-to_add_margin_meters, uly+to_add_margin_meters, lrx+to_add_margin_meters, lry-to_add_margin_meters], outputType=gdal.GDT_Byte, format='vrt')


                geo_water_clipped = [geo[0] - to_add_margin_meters, args.pixel_size, 0, geo[3] + to_add_margin_meters, 0, -args.pixel_size]
                ds_water_clipped.SetGeoTransform(geo_water_clipped)
                ds_water_clipped.SetProjection(proj)



                ds_dist_water = driver_mem.Create("", W+to_add_margin_pix, H+to_add_margin_pix, 1,
                                    gdal.GDT_Float32)

                ds_dist_water.SetGeoTransform(geo_water_clipped)

                ds_dist_water.SetProjection(proj)
                gdal.ComputeProximity(ds_water_clipped.GetRasterBand(1), ds_dist_water.GetRasterBand(1), ["VALUES=1", "DISTUNITS=GEO"])


                ds_dist_water_placette = gdal.Translate('', ds_dist_water,
                                              projWin=[ulx, uly, lrx, lry], outputType=gdal.GDT_Float32, format='vrt')

                water_dist_array = np.round(ds_dist_water_placette.GetRasterBand(1).ReadAsArray(), 1)

                ds_dist_water = None
                ds_water_clipped = None
                ds_dist_water_placette = None

                data_final_df[['x_round', 'y_round']] = np.floor(data_final_df[['x', 'y']] * (1 / args.pixel_size)) / (1 / args.pixel_size)
                data_final_df['j'] = ((data_final_df['x_round'] - ulx) / args.pixel_size).astype('int16')
                data_final_df['i'] = ((data_final_df['y_round'] - lry) / args.pixel_size).astype('int16')
                data_final_df['i'] = H - 1 - data_final_df['i']


                data_final_df['new_index'] = data_final_df['j'] + data_final_df['i'] * W

                data_final_df['d'] = water_dist_array.flatten()[data_final_df['new_index']]
                data_final_df['d'].round(1)
                # we put colums in right order, so features are at the beginning and then tree_id and class_by_height
                inversed_cols = np.concatenate((col_full, ['d'], ['tree_height_class']),0)[np.concatenate(([range(len(col_full)-1), [-2, -3, -1]]), 0)]
                data_ply_trees_placette_new = data_final_df[
                    inversed_cols].to_numpy()

            else:
                # we extract useful columns
                data_ply_trees_placette_new = data_final_df[
                    np.concatenate((col_full, ['tree_height_class']), 0)].to_numpy()


            data_ply_trees_placette_new = data_ply_trees_placette_new[data_ply_trees_placette_new[:, 2]>=0] #we delete strange points
            dataset[pl_id] = data_ply_trees_placette_new



            if all_points is None:
                all_points = np.empty((0, data_ply_trees_placette_new.shape[1]))
            all_points = np.append(all_points, data_ply_trees_placette_new, axis=0)


    print("Number of annotated points", annotated_points_count)
    print("Number of points", points_count)

    mean_dataset = np.ceil(np.mean(all_points[:, :2], axis=0))

    if "d" in args.input_feats:
        max_dist = np.max(all_points[:, -3], axis=0)
    else:
        max_dist = None

    if "d" in args.input_feats:
        ds_water = None

    for pl_id in dataset.keys():
        dataset[pl_id][:, :2] = dataset[pl_id][:, :2] - mean_dataset


    all_points[:, :2] = all_points[:, :2] - mean_dataset


    return all_points, dataset, mean_dataset, col_full, gt_rasters_dataset, max_dist




def open_ply_inference(args):
    print("open ply")
    path_clouds = args.path_inference
    selected_placette_folders_final = os.listdir(path_clouds)

    if "d" in args.input_feats:
        path_water = args.path + args.folder_water + "/water_clipped.tif"

    dataset = {}

    all_points = None
    for s in selected_placette_folders_final:
        pl_id = int((re.search("Placette_([0-9]*)", s)).group(1))

        ply_trees_placette = path_clouds + s + "/Pl_" + str(pl_id) + "_inference_data_clipped_xyzinr.ply"

        if pl_id in args.inference_pl:
            print(s)
            data_ply_trees_placette, col_full = open_ply(ply_trees_placette)

            if "d" in args.input_feats:

                geo = [np.floor(data_ply_trees_placette[:, 0].min()), args.pixel_size, 0, np.ceil(data_ply_trees_placette[:, 1].max()), 0, -args.pixel_size]
                H, W = int(np.ceil(data_ply_trees_placette[:, 1].max()) - np.floor(data_ply_trees_placette[:, 1].min())), int(np.ceil(data_ply_trees_placette[:, 0].max()) - np.floor(data_ply_trees_placette[:, 0].min()))

                ulx, uly, lrx, lry = geo[0], geo[3], geo[0] + W * args.pixel_size, geo[3] - H * args.pixel_size

                driver_mem = gdal.GetDriverByName('MEM')
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(2154)    # Lambert-93 projection
                proj = srs.ExportToWkt()


                # output = "/home/ign.fr_ekalinicheva/DATASETS/Processed_GT/search_for_water/water_dist_reshaped_placette.tif"
                to_add_margin_meters = 2000
                to_add_margin_pix = int(to_add_margin_meters / args.pixel_size) * 2
                ds_water_clipped = gdal.Translate('', path_water, xRes=args.pixel_size, yRes=args.pixel_size, resampleAlg="bilinear",
                                              projWin=[ulx-to_add_margin_meters, uly+to_add_margin_meters, lrx+to_add_margin_meters, lry-to_add_margin_meters], outputType=gdal.GDT_Byte, format='vrt')


                geo_water_clipped = [geo[0] - to_add_margin_meters, args.pixel_size, 0, geo[3] + to_add_margin_meters, 0, -args.pixel_size]
                ds_water_clipped.SetGeoTransform(geo_water_clipped)
                ds_water_clipped.SetProjection(proj)



                ds_dist_water = driver_mem.Create("", W+to_add_margin_pix, H+to_add_margin_pix, 1,
                                    gdal.GDT_Float32)

                ds_dist_water.SetGeoTransform(geo_water_clipped)

                ds_dist_water.SetProjection(proj)
                gdal.ComputeProximity(ds_water_clipped.GetRasterBand(1), ds_dist_water.GetRasterBand(1), ["VALUES=1", "DISTUNITS=GEO"])


                ds_dist_water_placette = gdal.Translate('', ds_dist_water,
                                              projWin=[ulx, uly, lrx, lry], outputType=gdal.GDT_Float32, format='vrt')

                water_dist_array = np.round(ds_dist_water_placette.GetRasterBand(1).ReadAsArray(), 1)

                ds_dist_water = None
                ds_water_clipped = None
                ds_dist_water_placette = None

                data_final_df = pd.DataFrame(data_ply_trees_placette, columns=col_full)

                data_final_df[['x_round', 'y_round']] = np.floor(data_final_df[['x', 'y']] * (1 / args.pixel_size)) / (1 / args.pixel_size)
                data_final_df['j'] = ((data_final_df['x_round'] - ulx) / args.pixel_size).astype('int16')
                data_final_df['i'] = ((data_final_df['y_round'] - lry) / args.pixel_size).astype('int16')
                data_final_df['i'] = H - 1 - data_final_df['i']


                data_final_df['new_index'] = data_final_df['j'] + data_final_df['i'] * W

                data_final_df['d'] = water_dist_array.flatten()[data_final_df['new_index']]
                data_final_df['d'].round(1)
                # we put colums in right order, so features are at the beginning and then tree_id and class_by_height
                inversed_cols = np.concatenate((col_full, ['d'], ['tree_height_class']),0)[np.concatenate(([range(len(col_full)-1), [-2, -3, -1]]), 0)]
                data_ply_trees_placette_new = data_final_df[
                    inversed_cols].to_numpy()

            else:
                data_ply_trees_placette_new = data_ply_trees_placette



            data_ply_trees_placette_new = data_ply_trees_placette_new[data_ply_trees_placette_new[:, 2]>=0] #we delete strange points
            dataset[pl_id] = data_ply_trees_placette_new



            if all_points is None:
                all_points = np.empty((0, data_ply_trees_placette_new.shape[1]))
            all_points = np.append(all_points, data_ply_trees_placette_new, axis=0)

    mean_dataset = np.ceil(np.mean(all_points[:, :2], axis=0))

    if "d" in args.input_feats:
        ds_water = None


    for pl_id in dataset.keys():
        dataset[pl_id][:, :2] = dataset[pl_id][:, :2] - mean_dataset


    all_points[:, :2] = all_points[:, :2] - mean_dataset


    return all_points, dataset, mean_dataset, col_full

