# Multi-Layer Modeling of Dense Vegetation from Aerial LiDAR Scans

This repository contrains code and dataset for the article _"Multi-Layer Modeling of Dense Vegetation from Aerial LiDAR Scans", CVPR 2022, Earth Vision Workshop_.

## Dataset
The **WildForest3D** dataset contains 29 plots of dense forest, with 7 million 3D points and 2.1 million individual labels.
The study area is located in the heavily forested French region of Aquitaine, and was scanned using a LiDAR installed on unmanned aerial vehicle with an average of 60 pulses per m². Each point is associated with its coordinates in Lambert-93 projection, the intensity value of returned laser signal, and the echo return number. The elevation of the points is given using a digital elevation model, such that the ground points are always at z=0m.

The dataset is located in _WildForest3D_ folder. The dataset structure is the following:
* _data_point_clouds_ - contains 29 folders _Placette_XX_ organized by plot ID, each of these folders contains *.ply point clouds and corresponding *.csv files with the parameters of each individual tree/bush instance.
* _stratum_coverage_2D_ - folder with GeoTIFF ground truth rasters generated from 3D point clouds (see the article)


## Model

## Postprocessing
