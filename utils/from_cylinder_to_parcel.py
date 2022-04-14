import torch
import numpy as np
from torch_scatter import scatter_max, scatter_mean
from utils.useful_functions import create_tiff, create_ply
from utils.open_ply_all import open_ply_all
# from utils.create_mesh import create_dsm
from osgeo import gdal
from model.accuracy import print_stats


def reconstruct(pred_pointwise_all, cloud_all, stats_path, args, epoch_nb, cm, cm_2d):
    if pred_pointwise_all.is_cuda:
        cloud_all = cloud_all.cuda()

    pred_pointwise_all = pred_pointwise_all.detach()

    # we get the origins of each point and the plot id, so we can reconstruct the whole plot from overlapping cylinders
    origins = cloud_all[:, -1].type(torch.int64)
    plots = cloud_all[:, -2]

    # we combine the xyz coordinates and the features with the predictions and then we apply scatter_mean
    combined_results = torch.cat((pred_pointwise_all, cloud_all), 1)
    reconstructed_cloud = scatter_mean(combined_results.T, origins).T[torch.unique(origins)]
    del cloud_all, combined_results


    all_plots = torch.unique(plots)
    n_pred_class = args.n_class
    for b in range(len(all_plots)):
        pl_id = all_plots[b]
        print_stats(args.stats_file, "Plot " + str(int(pl_id.cpu())), print_to_console=True)
        reconstructed_cloud_plot = reconstructed_cloud[reconstructed_cloud[:, -2] == pl_id]
        xy = reconstructed_cloud_plot[:, pred_pointwise_all.size(1):pred_pointwise_all.size(1)+2].T
        xy_round = torch.floor(xy * (1 / args.pixel_size)) / (1 / args.pixel_size)

        raster_size = (torch.ceil(torch.max(xy_round, 1)[0]) - torch.floor(torch.min(xy_round, 1)[0])) / args.pixel_size + 1
        raster_size = [(torch.max(xy_round[0]) - torch.floor(torch.min(xy_round[0])) + args.pixel_size) / args.pixel_size, (torch.ceil(torch.max(xy_round[1]) + args.pixel_size) - torch.min(xy_round[1])) / args.pixel_size]

        new_xy = ((xy_round - (torch.floor(torch.min(xy_round, dim=1).values)).reshape(2, 1))/args.pixel_size).int()    # no matter what, we always clip by whole coordinates
        yx = new_xy[[1, 0], :]  # we swap x and y t0 be able to pass to ij discrete coords
        yx[0] = raster_size[1].int() - 1 - yx[0]   #we inverse the Y axis values, cause geo coords and raster indices do not increase in the same direction

        unique, index = torch.unique(yx, dim=1, return_inverse=True)
        unique = unique.long()

        # We get rid of stem class and ground points class
        reconstructed_cloud_plot_pred = torch.cat((reconstructed_cloud_plot[:, 1:3], reconstructed_cloud_plot[:, 3:args.n_class-1].max(1)[0].reshape(-1, 1)), 1)


        pos = reconstructed_cloud_plot[:, n_pred_class:n_pred_class+3]
        gt_label = reconstructed_cloud_plot[:, n_pred_class+args.n_input_feats]

        point_class_predictions = torch.argmax(reconstructed_cloud_plot[:, :n_pred_class], 1)

        if cm is not None:
            cm.clear()
            cm.add(gt_label.cpu().numpy(), point_class_predictions.cpu().detach().numpy())
            cm.class_IoU(args)


        pred_raster = torch.full((raster_size[1].int(), raster_size[0].int(), args.nb_stratum), 0, dtype=torch.float)
        if pred_pointwise_all.is_cuda:
            pred_raster = pred_raster.cuda()

        pixel_max = scatter_max(reconstructed_cloud_plot_pred.T, index)[0]


        pred_raster[unique[0], unique[1]] = pixel_max.T.float()
        pred_raster = pred_raster.permute(2, 0, 1).cpu().numpy()
        # geo = [torch.floor(torch.min(xy_round[0])).cpu().numpy(), args.pixel_size, 0, torch.ceil(torch.max(xy_round[1])).cpu().numpy() + args.pixel_size, 0, -args.pixel_size]
        geo = [torch.floor(torch.min(xy_round[0])).cpu().numpy(), args.pixel_size, 0, torch.ceil(torch.max(xy_round[1])+ args.pixel_size).cpu().numpy(), 0, -args.pixel_size]

        pred_name_tiff = stats_path + "Pl_" + str(pl_id.cpu().int().numpy()) + "_predicted_coverage_ep_"+str(epoch_nb)+".tif"
        pred_name_ply = stats_path + "Pl_" + str(pl_id.cpu().int().numpy()) + "_predicted_coverage_ep_"+str(epoch_nb)+".ply"

        create_tiff(args.nb_stratum, pred_name_tiff, int(raster_size[0]), int(raster_size[1]), gdal.GDT_Float32, pred_raster, geo)
        create_ply(pos.cpu().numpy(), point_class_predictions.cpu().numpy(), reconstructed_cloud_plot_pred.cpu().numpy(), gt_label.cpu().numpy(), args, pred_name_ply)

        if args.gt_rasters_dataset is None and cm is not None:
            _, _, _, _, args.gt_rasters_dataset, _ = open_ply_all(args, selected_placettes=args.test_pl)


        if args.pixel_size==0.5 and cm_2d is not None:
            gt_raster = args.gt_rasters_dataset[int(pl_id.cpu())][0][:, :raster_size[1].int(), :raster_size[0].int()]
            if len(gt_raster) == 4:
                gt_raster = gt_raster[1:]
            strata_names = ['shrub', 'understory', 'canopy']
            for s in range(len(pred_raster)):
                print_stats(args.stats_file, "Computing for stratum: " + strata_names[s], print_to_console=True)
                cm_2d.clear()
                pred_binary_raster = np.zeros(len(pred_raster[s].flatten()))
                pred_binary_raster[pred_raster[s].flatten()>=0.5] = 1
                cm_2d.add(gt_raster[s].flatten(), pred_binary_raster)
                cm_2d.class_IoU(args)
                cm_2d.overall_accuracy(args)
            print_stats(args.stats_file, "Computing for all strata", print_to_console=True)
            cm_2d.clear()
            pred_binary_raster = np.zeros(len(pred_raster.flatten()))
            pred_binary_raster[pred_raster.flatten() >= 0.5] = 1
            cm_2d.add(gt_raster.flatten(), pred_binary_raster)
            cm_2d.class_IoU(args)
            cm_2d.overall_accuracy(args)