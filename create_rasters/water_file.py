from osgeo import gdal, ogr
import numpy as np

shape_file = "/home/ign.fr_ekalinicheva/DATASETS/Processed_GT/search_for_water/water_clipped.shp"
shape_file_traj = "/home/ign.fr_ekalinicheva/DATASETS/04-Zones/02-Zone_acquisition//Acquisition_FRISBEE_Ciron_EPSG2154.shp"
shape_file = "/home/ign.fr/ekalinicheva/Downloads/TronconHydrographique_FXX-shp/TronconHydrographique_FXX.shp"


output_raster = "/home/ign.fr_ekalinicheva/DATASETS/Processed_GT/search_for_water/water_clipped.tif"
output_raster2 = "/home/ign.fr_ekalinicheva/DATASETS/Processed_GT/search_for_water/traj_clipped.tif"
output_raster3 = "/home/ign.fr_ekalinicheva/DATASETS/Processed_GT/search_for_water/water_dist_clipped.tif"


input_shp = ogr.Open(shape_file)
traj_shp = ogr.Open(shape_file_traj)
shp_layer = input_shp.GetLayer()
shp_layer_traj = traj_shp.GetLayer()

pixel_size = 1
xmin, xmax, ymin, ymax = shp_layer_traj.GetExtent()
xmin, xmax, ymin, ymax = np.floor(xmin), np.ceil(xmax), np.floor(ymin), np.ceil(ymax)


print(xmin, xmax, ymin, ymax)

print(gdal.GDALRasterizeOptions)
ds = gdal.Rasterize(output_raster, shape_file, xRes=pixel_size, yRes=pixel_size,
                    burnValues=1, outputBounds=[xmin, ymin, xmax, ymax],
                    outputType=gdal.GDT_Byte)

ds2 = gdal.Rasterize(output_raster2, shape_file_traj, xRes=pixel_size, yRes=pixel_size,
                    burnValues=1, outputBounds=[xmin, ymin, xmax, ymax],
                    outputType=gdal.GDT_Byte)

drv = gdal.GetDriverByName('GTiff')
dst_ds = drv.Create(output_raster3,
                    ds.RasterXSize, ds.RasterYSize, 1,
                    gdal.GetDataTypeByName('Float32'))
print(ds.RasterXSize)

dst_ds.SetGeoTransform(ds.GetGeoTransform())
dst_ds.SetProjection(ds.GetProjectionRef())


gdal.ComputeProximity(ds.GetRasterBand(1), dst_ds.GetRasterBand(1), ["VALUES=1", "DISTUNITS=GEO"])

ds = None
dst_ds = None
