import warnings
warnings.simplefilter(action='ignore')
import functools


# We import from other files
from train_full import *
from data_loader.loader import *
from data_loader.create_grids import *
from utils.open_ply_all import *
from model.loss_functions import *
from model.accuracy import *
from config import args


print(torch.cuda.is_available())
np.random.seed(42)
torch.cuda.empty_cache()

def main():
    # We keep track of time and stats
    start_time = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(start_time)))
    run_name = str(time.strftime("%Y-%m-%d_%H%M%S"))
    args.run_name = run_name

    stats_path = os.path.join(args.results_path, run_name) + "/"
    args.stats_path = stats_path
    create_dir(stats_path)
    print("Results folder: ", stats_path)

    stats_file = os.path.join(stats_path, "stats.txt")
    args.stats_file = stats_file


    args.pl_id_list = np.sort(
        [1, 10, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 4, 5, 6, 7, 8, 9, 92])
    args.test_pl = np.asarray([20, 15, 4])

    args.val_pl = np.asarray([])
    args.train_pl = np.setdiff1d(np.setdiff1d(args.pl_id_list, args.val_pl, assume_unique=True), args.test_pl, assume_unique=True)

    args.inference_pl = np.asarray([])

    # all_points, dataset, mean_dataset, col_full, gt_rasters_dataset, dist_max = open_ply_all(args)



    folder_name_params = "rg_" + str(args.regular_grid_size) + "_sg_" + str(args.sample_grid_size) + '_plot_r_' + str(args.plot_radius) + '_pixel_size_' + str(args.pixel_size) + '_min_pts_cyl_' + str(args.min_pts_cylinder) + "_n_plots_" + str(len(args.pl_id_list))
    path_cylinders = 'saved_data/' + folder_name_params + "/"

    print_stats(stats_file, str(args), print_to_console=True)  # save all the args parameters

    # We open las files and create a dataset
    print("Loading data in memory")
    args.gt_rasters_dataset = None
    if os.path.exists(path_cylinders) and len(os.listdir(path_cylinders)) != 0:
        z_all = np.load(path_cylinders + "/all_zzzz.npy")
        args.mean_dataset = np.load(path_cylinders + "/mean_dataset.npy")
        args.dist_max = np.load(path_cylinders + "/dist_max.npy")
        args.z_max = np.max(
            z_all)  # maximum z value for data normalization, obtained from the normalized dataset analysis
        cylinders_dataset_by_plot = torch.load(path_cylinders + '/cylinders_dataset_by_plot.pt')
        cylinder_rasters_gt_by_plot = torch.load(path_cylinders + '/cylinder_rasters_gt_by_plot.pt')
    else:
        all_points, dataset, mean_dataset, col_full, gt_rasters_dataset, dist_max = open_ply_all(args)
        args.mean_dataset = mean_dataset
        args.gt_rasters_dataset = gt_rasters_dataset
        args.dist_max = dist_max
        z_all = all_points[:, 2]
        args.z_max = np.max(
            z_all)  # maximum z value for data normalization, obtained from the normalized dataset analysis

        print("create cylinders")
        print(gt_rasters_dataset.keys())
        cylinders_dataset_by_plot, cylinder_rasters_gt_by_plot = create_grids(dataset, gt_rasters_dataset, args, train=None)
        create_dir(path_cylinders)
        torch.save(cylinders_dataset_by_plot, path_cylinders + '/cylinders_dataset_by_plot.pt')
        torch.save(cylinder_rasters_gt_by_plot, path_cylinders + '/cylinder_rasters_gt_by_plot.pt')
        np.save(path_cylinders + "/mean_dataset.npy", mean_dataset)
        np.save(path_cylinders + "/col_full.npy", col_full)
        np.save(path_cylinders + "/all_zzzz.npy", z_all)
        np.save(path_cylinders + "/dist_max.npy", dist_max)


    # args.z_max = args.plot_radius
    cylinders_dataset = sum(cylinders_dataset_by_plot.values(), [])
    cylinder_rasters_gt = sum(cylinder_rasters_gt_by_plot.values(), [])

    #
    # if len(args.inference_pl)>0:
    #     all_points, dataset, mean_dataset, col_full, gt_rasters_dataset = open_ply_all(args, inference=True)
    #     cylinders_dataset_by_plot_inf, cylinder_rasters_gt_by_plot_inf = create_grids(dataset, gt_rasters_dataset, args, train=None)



    pl_id_plus_cylinders = np.empty((0, 2), dtype=int)
    for pl_id, cylinders_of_plot in cylinders_dataset_by_plot.items():
        nbr_cyl = len(cylinders_of_plot)
        new_list = np.concatenate((np.full((nbr_cyl), pl_id).reshape(-1, 1),
                                   np.arange(len(pl_id_plus_cylinders), len(pl_id_plus_cylinders) + nbr_cyl).reshape(-1,
                                                                                                                     1)),
                                  1)
        pl_id_plus_cylinders = np.concatenate((pl_id_plus_cylinders, new_list), 0)





    params = {'phi': 0.8379113482270742, 'a_g': 0.3192297150916541, 'a_v': 417.26435504747633, 'loc_g': -3.208996207954415e-29,
     'loc_v': -112.90408244005155, 'scale_g': 0.4400063325755004, 'scale_v': 0.302502487153544}
    params = {'phi': 0.8483857369709003, 'a_g': 0.10653676198633916, 'a_v': 496.2167067837195, 'loc_g': -6.37970607833523e-30, 'loc_v': -126.11144055207234, 'scale_g': 0.19297981609158193, 'scale_v': 0.2806149611888358} #loglikelihood=-0.7568642649448962
    # params = {'phi': 0.8083130074439174, 'a_g': 0.37997424715822514, 'a_v': 1171.3291314532107,
    #           'loc_g': -4.122287330446177e-29, 'loc_v': -181.71013449013003, 'scale_g': 0.3961613455687545,
    #           'scale_v': 0.1669330117749968}

    print_stats(stats_file, str(params), print_to_console=True)


    train_list = args.train_pl
    test_list = args.test_pl

    index_dict = {}


    print("creating datasets")
    # generate the train and test dataset
    test_set = tnt.dataset.ListDataset(pl_id_plus_cylinders[[i for i in range(len(pl_id_plus_cylinders[:, 0])) if pl_id_plus_cylinders[:, 0][i] in test_list]][:, 1],
                                       functools.partial(cloud_loader, dataset=cylinders_dataset, gt_raster=cylinder_rasters_gt, train=False, index_dict=index_dict, args=args))
    train_set = tnt.dataset.ListDataset(pl_id_plus_cylinders[[i for i in range(len(pl_id_plus_cylinders[:, 0])) if pl_id_plus_cylinders[:, 0][i] in train_list]][:, 1],
                                        functools.partial(cloud_loader, dataset=cylinders_dataset, gt_raster=cylinder_rasters_gt, train=True, index_dict=index_dict, args=args))


    trained_model, final_train_losses_list, final_test_losses_list = train_full(args, params, train_set, test_set)

    # save the trained model
    PATH = os.path.join(stats_path, "model_ss_" + str(args.subsample_size) + "_dp_" + str(args.diam_pix) + ".pt")
    torch.save(trained_model, PATH)
#

if __name__ == "__main__":
    main()

