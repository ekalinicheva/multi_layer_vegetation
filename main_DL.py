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
from inference import inference


print(torch.cuda.is_available())
np.random.seed(42)
torch.cuda.empty_cache()

def main():
    # We keep track of time and stats
    start_time = time.time()
    print(time.strftime("%H:%M:%S", time.gmtime(start_time)))
    run_name = str(time.strftime("%Y-%m-%d_%H%M%S"))
    args.run_name = run_name

    args.stats_path = os.path.join(os.path.join(args.path, args.folder_results), run_name) + "/"
    create_dir(args.stats_path)
    print("Results folder: ", args.stats_path)

    args.stats_file = os.path.join(args.stats_path, "stats.txt")


    #
    # inference_pl = np.arange(28, 31)
    # inference_pl = [131, 132]
    #
    # for ip in inference_pl:
    #     inference_pl_ = np.asarray([ip*10+3])
    #     for pl in inference_pl_:
    #         args.inference_pl = np.asarray([pl])
    #         print(pl)
    #         if os.path.exists(args.path_inference + "Placette_" + str(pl) + "/Pl_" + str(pl) + "_inference_data_clipped_xyzinr.ply"):
    #
    # for pl in inference_pl:
    #     args.inference_pl = np.asarray([pl])
    #     print(args.path_inference + "Placette_" + str(pl) + "/Pl_" + str(pl) + "_inference_data_clipped_xyzinr.ply")
    #     if os.path.exists(args.path_inference + "Placette_" + str(pl) + "/Pl_" + str(pl) + "_inference_data_clipped_xyzinr.ply"):


    folder_name_params = "rg_" + str(args.regular_grid_size) + "_sg_" + str(args.sample_grid_size) + '_plot_r_' + str(args.plot_radius) + '_pixel_size_' + str(args.pixel_size) + '_min_pts_cyl_' + str(args.min_pts_cylinder) + "_n_plots_" + str(len(args.pl_id_list))
    path_cylinders = 'saved_data/' + folder_name_params + "/"

    print_stats(args.stats_file, str(args), print_to_console=True)  # save all the args parameters

    # We open las files and create a dataset
    print("Loading data in memory")
    args.gt_rasters_dataset = None
    if not args.inference:
        if os.path.exists(path_cylinders) and len(os.listdir(path_cylinders)) != 0:
            z_all = np.load(path_cylinders + "/all_zzzz.npy")
            args.mean_dataset = np.load(path_cylinders + "/mean_dataset.npy")
            args.z_max = np.max(
                z_all)  # maximum z value for data normalization, obtained from the normalized dataset analysis
            xy_min_coords_by_plot = torch.load(path_cylinders + '/xy_min_coords_by_plot.pt')
            cylinders_dataset_by_plot = torch.load(path_cylinders + '/cylinders_dataset_by_plot.pt')
            cylinder_rasters_gt_by_plot = torch.load(path_cylinders + '/cylinder_rasters_gt_by_plot.pt')
            if "d" in args.input_feats:
                args.dist_max = torch.load(path_cylinders + "/dist_max.pt").numpy()[0]
        else:
            all_points, dataset, mean_dataset, col_full, gt_rasters_dataset, dist_max = open_ply_all(args)
            args.mean_dataset = mean_dataset
            args.gt_rasters_dataset = gt_rasters_dataset


            print("create cylinders")
            print(gt_rasters_dataset.keys())
            cylinders_dataset_by_plot, cylinder_rasters_gt_by_plot, xy_min_coords_by_plot = create_grids(dataset, gt_rasters_dataset, args)
            args.dist_max = dist_max
            z_all = all_points[:, 2]
            args.z_max = np.max(
                z_all)  # maximum z value for data normalization, obtained from the normalized dataset analysis
            create_dir(path_cylinders)
            torch.save(cylinders_dataset_by_plot, path_cylinders + '/cylinders_dataset_by_plot.pt')
            torch.save(cylinder_rasters_gt_by_plot, path_cylinders + '/cylinder_rasters_gt_by_plot.pt')
            torch.save(xy_min_coords_by_plot, path_cylinders + '/xy_min_coords_by_plot.pt')
            np.save(path_cylinders + "/mean_dataset.npy", mean_dataset)
            np.save(path_cylinders + "/col_full.npy", col_full)
            np.save(path_cylinders + "/all_zzzz.npy", z_all)
            if "d" in args.input_feats:
                torch.save(torch.Tensor([dist_max]), path_cylinders + "/dist_max.pt")

        cylinder_rasters_gt = sum(cylinder_rasters_gt_by_plot.values(), [])

    else:
        all_points, dataset, mean_dataset, col_full = open_ply_inference(args)
        args.mean_dataset = mean_dataset

        cylinders_dataset_by_plot, _, xy_min_coords_by_plot = create_grids(dataset, None, args)

    xy_min_coords = sum(xy_min_coords_by_plot.values(), [])
    cylinders_dataset = sum(cylinders_dataset_by_plot.values(), [])


    pl_id_plus_cylinders = np.empty((0, 2), dtype=int)
    for pl_id, cylinders_of_plot in cylinders_dataset_by_plot.items():
        nbr_cyl = len(cylinders_of_plot)
        new_list = np.concatenate((np.full((nbr_cyl), pl_id).reshape(-1, 1),
                                   np.arange(len(pl_id_plus_cylinders), len(pl_id_plus_cylinders) + nbr_cyl).reshape(-1,
                                                                                                                     1)),
                                  1)
        pl_id_plus_cylinders = np.concatenate((pl_id_plus_cylinders, new_list), 0)


    del cylinders_dataset_by_plot, cylinders_of_plot, xy_min_coords_by_plot


    params = {'phi': 0.8379113482270742, 'a_g': 0.3192297150916541, 'a_v': 417.26435504747633, 'loc_g': -3.208996207954415e-29,
     'loc_v': -112.90408244005155, 'scale_g': 0.4400063325755004, 'scale_v': 0.302502487153544}
    params = {'phi': 0.8483857369709003, 'a_g': 0.10653676198633916, 'a_v': 496.2167067837195, 'loc_g': -6.37970607833523e-30, 'loc_v': -126.11144055207234, 'scale_g': 0.19297981609158193, 'scale_v': 0.2806149611888358} #loglikelihood=-0.7568642649448962
    # params = {'phi': 0.8083130074439174, 'a_g': 0.37997424715822514, 'a_v': 1171.3291314532107,
    #           'loc_g': -4.122287330446177e-29, 'loc_v': -181.71013449013003, 'scale_g': 0.3961613455687545,
    #           'scale_v': 0.1669330117749968}

    print_stats(args.stats_file, str(params), print_to_console=True)


    train_list = args.train_pl
    test_list = args.test_pl
    if args.inference:
        inference_list = args.inference_pl

    index_dict = {}


    print("creating datasets")
    # generate the train and test dataset
    if not args.inference:
        test_set = tnt.dataset.ListDataset(pl_id_plus_cylinders[[i for i in range(len(pl_id_plus_cylinders[:, 0])) if pl_id_plus_cylinders[:, 0][i] in test_list]][:, 1],
                                           functools.partial(cloud_loader, dataset=cylinders_dataset, gt_raster=cylinder_rasters_gt, min_coords=xy_min_coords, train=False, index_dict=index_dict, args=args))
        train_set = tnt.dataset.ListDataset(pl_id_plus_cylinders[[i for i in range(len(pl_id_plus_cylinders[:, 0])) if pl_id_plus_cylinders[:, 0][i] in train_list]][:, 1],
                                            functools.partial(cloud_loader, dataset=cylinders_dataset, gt_raster=cylinder_rasters_gt, min_coords=xy_min_coords, train=True, index_dict=index_dict, args=args))
        train_full(args, params, train_set, test_set)
    else:
        inference_set = tnt.dataset.ListDataset(pl_id_plus_cylinders[[i for i in range(len(pl_id_plus_cylinders[:, 0])) if pl_id_plus_cylinders[:, 0][i] in inference_list]][:, 1],
                                            functools.partial(cloud_loader, dataset=cylinders_dataset, gt_raster=None, min_coords=xy_min_coords, train=False, index_dict=index_dict, args=args))
        inference(inference_set, args)


if __name__ == "__main__":
    main()

