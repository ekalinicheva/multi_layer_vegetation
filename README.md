# Multi-Layer Modeling of Dense Vegetation from Aerial LiDAR Scans

This repository contrains the WildForest3D dataset and the code for pipeline presented in the article:

_Ekaterina Kalinicheva, Loic Landrieu, Clément Mallet, Nesrine Chehata "Multi-Layer Modeling of Dense Vegetation from Aerial LiDAR Scans", CVPR 2022, Earth Vision Workshop._ [\[arXiv\]](https://arxiv.org/abs/2204.11620)


<img src="examples_images/gradient-4.png" width="500" />

**THE DESCRIPTION IS NOT FINALIZED YET!**

## Dataset
The **WildForest3D** dataset contains 29 plots of dense forest, with 7 million 3D points and 2.1 million individual labels.
The study area is located in the heavily forested French region of Aquitaine, and was scanned using a LiDAR installed on unmanned aerial vehicle with an average of 60 pulses per m². Each point is associated with its coordinates in Lambert-93 projection, the intensity value of returned laser signal, and the echo return number. The elevation of the points is given using a digital elevation model, such that the ground points are always at z=0m.

The dataset is located in _WildForest3D_ folder. 
If you want the code to work properly, your folder structure should be the following. arg.smth is the parameter you have to precise in the configuration file `config.py`:

├─your_main_data_folder (args.path)  
&emsp;├─data_point_clouds (args.folder_clouds)  
&emsp;│&emsp;├─Placette_1  
&emsp;│&emsp;├─Placette_2  
&emsp;│&emsp;└─Placette_XX  
&emsp;│&emsp;&emsp;├─Pl_XX_final_data_xyzinr.ply  
&emsp;│&emsp;&emsp;├─Pl_XX_trees_params.csv  
&emsp;│&emsp;&emsp;└─Pl_XX_trees_bb.csv  
&emsp;├─gt_rasters (args.folder_gt_rasters)  
&emsp;│&emsp;├─Placette_1  
&emsp;│&emsp;├─Placette_2  
&emsp;│&emsp;└─Placette_XX  
&emsp;│&emsp;&emsp;├─Pl_XX_Coverage_sure_05.tif      
&emsp;│&emsp;&emsp;└─Pl_XX_Coverage_height_05.tif     
&emsp;├─water_raster (args.folder_water)  
&emsp;│&emsp;└─water_clipped.tif                  
&emsp;└─results (args.folder_results)


* `data_point_clouds/` - contains 29 folders `Placette_XX/` organized by plot ID, each of those folders contains: 
  * `Pl_XX_final_data_xyzinr.ply` that contains clipped plots with annotated point clouds, the points are annotated in an instance-wise way, so that each point contains the following information : XYZ coordinates, intensity value, number of returns, return number, and the ID of the instance the point belongs to. ID=0 corresponds to a non-annotated point. 
  * `Pl_XX_trees_params.csv` file with the parameters of each individual tree/bush instance. It contains information about each instance (its class name, class category, height, crown base height, etc). Using the script utils/open_ply_all.ply we can generate the 6 classes dataset that was used in the article. 
  * `Pl_XX_trees_bb.csv` - coorfinates of axis-oriented bounding boxes (BB) for each annotated tree (not used in the article, but might be useful to someone in this world). Each BB is discribed by its XY center, the extent in X and Y axes, and the BB height. Note that we consider that the bottom Y coordinate is always 0.     
* `gt_rasters/` - contains 29 folders `Placette_XX/` organized by plot ID, each of these folders contains: two GeoTIFF ground truth rasters generated from 3D point clouds (see the article for the details):
  * `Pl_XX_Coverage_sure_05.tif` (05 stands for the pixel size - 0.5m - though other pixel size rasters can be generated with the code `utils/generate_dataset.py`) - GT with binary occupancy maps for 3 vegetation layers : ground vegetation, understory, overstory. 1 - vegetataion, 0 - no vegetation, -1 - nodata. Nodata pixels are present, because we only have partial 3D annotation, so its projection on the rasters may create the ambiguity.
  * `Pl_XX_Coverage_height_05.tif` - the vegetation height of the vegetation-filled pixels by layer : ground vegetation (GV), understory, bottom of overstory and top of overstory. Note that by default, bottoms of GV and understory are 0.
* `water_raster/` - folder with GeoTIFF raster `water_clipped.tif` that respresent the distance to the closest water source (rivers). Pixels size 1m. It helps to distinguish the alder class which likes being close to the water. When we generate dataset, we have to precise in the configuration if we want to use this feature. If yes, it is added to each 3D point of the dataset. This is not precised in the article, as we only distinguish coniferous from decidious trees in the initial research. The water feature was added after the article submission.


## Model

<img src="examples_images/algo_scheme.png" width="750" />



The full pipeline description can be found in the article. The PointNet++ model can be found in `model/model_pointnet2.py` and the correponding model parameters in the `config.py`:
* _subsample_size_ - the number of points we sample per cylinder (default 16384);
* _smart_sampling_ - whether we sample those points in a smart way or not (added after submitting the article, so not in there). If smart sampling, all the points with height 0<z<=5m are always chosen, for other points we assign the proability: for z=0m and z>15m the probability is p=0.5, for 5<z<=15m the probability lies between 0.99 and 0.5 (linear);
* _r_num_pts_ - the number of groups of point sets for each MLP set abstraction level (default \[12228, 4098, 1024])
* _rr_ - the ball radius of the neighbourhood search of point sets for each MLP set abstraction level.
* _ratio_ - the ratio of reatained neightbourhood points of the ball neighbourhood.
* _drop_ - the probability value of the DropOut layer.


## Results

Our code saves the model and the results computed for test set every 5 epochs (parameter _args.n_epoch_test_ that can be modified). All the results are saved in the folder `results/`. Each model is saved in the subfolder `YYYY-MM-DD_HHMMSS/` which is created automatically once the code is launched. The following output is obtained at the end of each 5th (YY) epoch:
* `Pl_XX_predicted_coverage_ep_YY.ply` - classified 3D point cloud (hard class assigment) at YY epoch for XX plot.
* `Pl_XX_predicted_coverage_ep_YY.tif` - vegetation layers' occupancy prediction (soft assigment, obtained directly from output logits). Hard classification results are produced in the postprocessing step from 3D predictions.
* `epoch_YY.pt` - 

## Postprocessing


## Citation
If you are using our code or dataset, please, cite us:

@misc{https://doi.org/10.48550/arxiv.2204.11620,  
&emsp;  doi = {10.48550/ARXIV.2204.11620},  
&emsp;  url = {https://arxiv.org/abs/2204.11620},  
&emsp;  author = {Kalinicheva, Ekaterina and Landrieu, Loic and Mallet, Clément and Chehata, Nesrine},  
&emsp;  title = {Multi-Layer Modeling of Dense Vegetation from Aerial LiDAR Scans},  
&emsp;  publisher = {arXiv},  
&emsp;  year = {2022}  
}
