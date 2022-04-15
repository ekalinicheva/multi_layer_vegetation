# Multi-Layer Modeling of Dense Vegetation from Aerial LiDAR Scans

This repository contrains code and dataset for the article "Multi-Layer Modeling of Dense Vegetation from Aerial LiDAR Scans", CVPR 2022, Earth Vision Workshop.

## Dataset
The **WildForest3D** dataset contains 29 plots of dense forest, with 7 million 3D points and 2.1 million individual labels.
The study area is located in the heavily forested French region of Aquitaine, and was scanned using a LiDAR installed on unmanned aerial vehicle with an average of 60 pulses per m². Each point is associated with its coordinates in Lambert-93 projection, the intensity value of returned laser signal, and the echo return number. The elevation of the points is given using a digital elevation model, such that the ground points are always at z=0m.

The dataset is located in _WildForest3D_ folder. The dataset structure is the following:
* data_point_clouds - folder with ply point clouds
* stratum_coverage_2D - folder with GeoTIFF ground truth rasters generated from 3D point clouds (see the article)


## Model

## Postprocessing
