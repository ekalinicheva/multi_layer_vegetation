import os
from plyfile import PlyData, PlyElement
from osgeo import gdal, gdal_array, ogr, osr
import numpy as np


driver_tiff = gdal.GetDriverByName("GTiff")
driver_shp = ogr.GetDriverByName("ESRI Shapefile")


# Print stats to file
def print_stats(stats_file, text, print_to_console=True):
    with open(stats_file, 'a') as f:
        if isinstance(text, list):
            for t in text:
                f.write(t + "\n")
                if print_to_console:
                    print(t)
        else:
            f.write(text + "\n")
            if print_to_console:
                print(text)
    f.close()


# Function to create a new folder if does not exists
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def open_ply(ply_name):
    assert (os.path.isfile(ply_name))
    with open(ply_name, 'rb') as f:
        plydata = PlyData.read(f)
        col_names = plydata['vertex'].data.dtype.names
        nbr_points = plydata['vertex'].count
        data = np.zeros(shape=[nbr_points, len(col_names)], dtype=np.float64)
        for c in range(len(col_names)):
            data[:, c] = plydata['vertex'].data[col_names[c]]
    return data, np.asarray(col_names)


# Open GeoTIFF as an array
def open_tiff(path, name):
    ds = gdal.Open(path+"/"+name+".tif")
    geo = ds.GetGeoTransform()
    proj = ds.GetProjection()
    bands_nb = ds.RasterCount
    W = ds.RasterXSize
    H = ds.RasterYSize
    try:
        image_array = gdal_array.LoadFile(path + "/" + name+".tif")
    except:
        image_array = gdal_array.LoadFile(path + name+".tif")
    ds = None
    return np.asarray(image_array), H, W, geo, proj, bands_nb


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



def create_ply(pos, label_pred, score_pred, gt_label, args, file):

    if args.n_class==7:
        OBJECT_COLOR_4 = np.asarray(
            [
                [81, 109, 114],  # 'ground'   ->  grey #516D72
                [233, 229, 107],  # 'shrub' .-> .yellow
                [95, 156, 196],  # 'understory' .-> . blue #5F9CC4
                [108, 135, 75],  # 'tree'   ->  dark green #6C874B
                [1, 1, 1],  # 'pine'   ->  black
                [143, 243, 36],  # 'pine'   ->  apple green #8FF324
                [255, 127, 0],  # 'stem'   ->  orange #FF7F00
            ]
        )
    else:
        OBJECT_COLOR_4 = np.asarray(
            [
                [81, 109, 114],  # 'ground'   ->  grey #516D72
                [233, 229, 107],  # 'shrub' .-> .yellow
                [95, 156, 196],  # 'understory' .-> . blue #5F9CC4
                [108, 135, 75],  # 'tree'   ->  dark green #6C874B
                [1, 1, 1],  # 'pine'   ->  black
                [255, 127, 0],  # 'stem'   ->  orange #FF7F00
            ]
        )


    colors = OBJECT_COLOR_4[np.asarray(label_pred)]
    ply_array = np.ones(
        pos.shape[0], dtype=[("x", "f8"), ("y", "f8"), ("z", "f4"), ("class", "u1"), ("gt", "u1"), ("error", "u1"),
                             ("ground", "f4"), ("shrub", "f4"), ("understory", "f4"), ("canopy", "f4"),
                             ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    )
    ply_array["x"] = pos[:, 0]
    ply_array["y"] = pos[:, 1]
    ply_array["z"] = pos[:, 2]
    ply_array["class"] = label_pred
    ply_array["gt"] = gt_label

    error = gt_label.copy()
    error[gt_label==-1] = 0
    if args.nb_stratum == 3:
        error[np.where((gt_label==1) & (label_pred!=1))] = 3
        error[np.where((gt_label==2) & (label_pred!=2))] = 4

    elif args.n_class == 5:
        error[np.where((gt_label==2) & (label_pred!=2))] = 5
        error[np.where((gt_label!=2) & (label_pred==2))] = 6
        error[np.where((gt_label==3) & (label_pred!=3))] = 7
        error[np.where((gt_label!=3) & (label_pred==3))] = 8
        error[np.where((gt_label==4) & (label_pred!=4))] = 9
        error[np.where((gt_label!=4) & (label_pred==4))] = 10
        if args.stem:
            error[np.where((gt_label == 5) & (label_pred != 5))] = 11
            error[np.where((gt_label != 5) & (label_pred == 5))] = 12
    else:
        error[np.where((gt_label==2) & (label_pred!=2))] = 4
        error[np.where((gt_label!=2) & (label_pred==2))] = 5
        error[np.where((gt_label==3) & (label_pred!=3))] = 6
        error[np.where((gt_label!=3) & (label_pred==3))] = 7

    # print(np.unique(error, return_counts=True))

    ply_array["error"] = error



    ply_array["ground"] = score_pred[:, 0]
    ply_array["shrub"] = score_pred[:, 1]
    ply_array["understory"] = score_pred[:, -2]
    ply_array["canopy"] = score_pred[:, -1]
    ply_array["red"] = colors[:, 0]
    ply_array["green"] = colors[:, 1]
    ply_array["blue"] = colors[:, 2]

    ply_file = PlyData([PlyElement.describe(ply_array, 'vertex')], text=True)
    ply_file.write(file)
