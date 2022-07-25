import argparse
import os
import numpy as np


parser = argparse.ArgumentParser(description='model')

# System Parameters
parser.add_argument('--path', default="DATASETS/Processed_GT/", type=str,
                    help="Main folder directory")
parser.add_argument('--folder_clouds', default="final_data_legs/", type=str,
                    help="Folder with point clouds. Contains subfolders, named as Placette_ID  where ID is the placette id.")
parser.add_argument('--folder_gt_rasters', default="stratum_coverage_no_clip_extended_height/", type=str,
                    help="Folder with GT rasters. Contains subfolders, named as Placette_ID  where ID is the placette id.")
parser.add_argument('--folder_water', default="search_for_water/", type=str,
                    help="Folder with distance to water raster and river shp.")
parser.add_argument('--folder_results', default="RESULTS_4_stratum/", type=str,
                    help="Folder with all the final results.")


parser.add_argument('--pl_id_list', default=[1, 10, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 4, 5, 6, 7, 8, 9, 92],
                    type=int, help="The IDs of plots that we use (train + test). By default, there are all of them.")
parser.add_argument('--test_pl', default=[20, 15, 4],
                    type=int, help="The IDs of plots that we use for test.")
parser.add_argument('--inference_pl', default=[1, 2, 3],
                    type=int, help="The IDs of plots that we use for inference (in case we do it).")

parser.add_argument('--cuda', default=1, type=int, help="Whether we use cuda (1) or not (0)")
parser.add_argument('--folds', default=1, type=int, help="Number of folds for cross validation model training")

# Model Parameters
parser.add_argument('--n_class', default=6, type=int,
                    help="[5,6,7] Size of the model output vector. In our case 6 - different vegetation coverage types: ground, ground vegetation, understory, decidious, coniferous, stem. "
                         "If 7 -  ground, ground vegetation, understory, decidious (mostly oaks), coniferous, aulne, stem")
parser.add_argument('--nb_stratum', default=3, type=int,
                    help="Number of vegetation stratum that we compute 3 - ground vegetation + understory + overstory")
parser.add_argument('--input_feats', default='xyzinr', type=str,
                    help="Point features that we keep. To choose between xyzinrd. Please, do not permute. "
                         "xyz - coordinates, i - intensity, n - number of returns, r - return number, d - distance to river.")


parser.add_argument('--logl', default=True, type=bool, help="Whether we add loglikelihood loss or not")
parser.add_argument('--m', default=0.1, type=float,
                    help="Loss regularization. The weight of the negative loglikelihood loss in the total loss")

parser.add_argument('--ent', default=False, type=bool, help="Whether we add entropy loss or not")
parser.add_argument('--e', default=0.1, type=float,
                    help="Loss regularization for entropy. The weight of the entropy loss in the total loss")
parser.add_argument('--r', default=1, type=float,
                    help="Loss regularization for BCE raster loss. The weight of the BCE loss in the total loss")

# parser.add_argument('--ECM_ite_max', default=5, type=int, help='Max number of EVM iteration')
# parser.add_argument('--NR_ite_max', default=10, type=int, help='Max number of Netwon-Rachson iteration')

# Network Parameters
parser.add_argument('--ratio', default=[0.33, 0.5, 0.5], type=int, help="Ratio of centroid of PointNet2 layers")
# parser.add_argument('--ratio', default=[0.5, 0.5, 0.5], type=int, help="Ratio of centroid of PointNet2 layers")

parser.add_argument('--rr', default=[0.1, 0.25, 0.5], type=float, nargs=3, help="Radius of PointNet2 layers samplings")

parser.add_argument('--drop', default=0.25, type=float, help="Probability value of the DropOut layer of the model")
# parser.add_argument("--log_embeddings", default=False, action="store_true",
#                     help="False to avoid logging embeddings")

# Optimization Parameters
parser.add_argument('--wd', default=0.001, type=float, help="Weight decay for the optimizer")
parser.add_argument('--lr', default=0.0005, type=float, help="Learning rate")
parser.add_argument('--step_size', default=20, type=int,
                    help="After this number of steps we decrease learning rate. (Period of learning rate decay)")
parser.add_argument('--lr_decay', default=0.5, type=float,
                    help="We multiply learning rate by this value after certain number of steps (see --step_size). (Multiplicative factor of learning rate decay)")
parser.add_argument('--n_epoch', default=150, type=int, help="Number of training epochs")
parser.add_argument('--n_epoch_test', default=5, type=int, help="We evaluate every -th epoch")
parser.add_argument('--batch_size', default=5, type=int, help="Size of the training batch")


# Cylinder sampling parameters
parser.add_argument('--subsample_size', default=4096 * 4, type=int, help="Subsample cloud size")
parser.add_argument('--smart_sampling', default=True, type=bool,
                    help="Whether we sample points depending on their height or not")
parser.add_argument('--plot_radius', default=5, type=int, help="Size of sampled cylinder")
parser.add_argument('--pixel_size', default=0.5, type=float, help="Size of occupancy maps pixels")

parser.add_argument('--regular_grid_size', default=5, type=int, help="Regular grid size")
parser.add_argument('--sample_grid_size', default=1, type=int, help="Sample grid size")

parser.add_argument('--nbr_training_samples', default=1000, type=int, help="How many samples per training step")
parser.add_argument('--min_pts_cylinder', default=250, type=int,
                    help="The min number of points in sampled cylinder, otherwise it is deleted")


parser.add_argument('--data_augmentation', default=True, type=bool,
                    help="Whether we do data augmentation or not.")


parser.add_argument('--inference', default=False, type=bool,
                    help="Whether we train model or produce the results with the trained one.")
parser.add_argument('--path_inference', default="DATASETS/Processed_GT/inference_data/sites100m_EPSG2154_full/", type=str,
                    help="Main folder directory")
parser.add_argument('--trained_ep', default=100, type=int,
                    help="The epoch we load from the pretrained model.")
parser.add_argument('--path_model', default="/home/ign.fr/ekalinicheva/DATASETS/Processed_GT/RESULTS_4_stratum/2022-05-09_120514/", type=str,
                    help="Path to the pretrained model in case we use it.")


parser.add_argument('--train_model', default=True, type=bool,
                    help="Whether we train model or use the pretrained one to continue the training."
                         "If False, we use pretrained model at --path_model with --trained_ep epoch.")

args = parser.parse_args()

assert (args.nb_stratum in [3, 4]), "Number of stratum should be 3 or 4!"
assert (args.lr_decay < 1), "Learning rate decrease should be smaller than 1, as learning rate should decrease"

args.path = os.path.expanduser("~/" + args.path)
args.path_inference = os.path.expanduser("~/" + args.path_inference)


args.inference_pl = np.arange(1, 31)

# args.pl_id_list = np.sort(
#     [1, 10, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 4, 5, 6, 7, 8, 9, 92])
# args.test_pl = np.asarray([20, 15, 4])
# args.val_pl = np.asarray([])
# args.train_pl = np.setdiff1d(np.setdiff1d(args.pl_id_list, args.val_pl, assume_unique=True), args.test_pl,
#                              assume_unique=True)

args.pl_id_list = np.sort(args.pl_id_list)
args.test_pl = np.sort(args.test_pl)
args.train_pl = np.setdiff1d(args.pl_id_list, args.test_pl, assume_unique=True)
