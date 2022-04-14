import warnings
warnings.simplefilter(action='ignore')

import torchnet as tnt
import gc
import time
from torch.utils.tensorboard import SummaryWriter
from model.accuracy import print_stats


from pynvml import *

# We import from other files
from data_loader.loader import *
from utils.reproject_to_2d_and_predict_plot_coverage import *
from model.loss_functions import *
from utils.from_cylinder_to_parcel import reconstruct


np.random.seed(42)

def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return torch.cat(list(tuple_of_tensors), dim=1)



def evaluate(model, PCC, test_set, params, args, epoch_nb, cm, cm_2d, last_epoch=False):
    """eval on test set"""

    model.eval()

    loader = torch.utils.data.DataLoader(test_set, collate_fn=cloud_collate,
                                         batch_size=args.batch_size, shuffle=False, drop_last=False)

    # loader = torch.utils.data.DataLoader(test_set, collate_fn=cloud_collate, batch_size=args.batch_size, shuffle=False)
    loss_meter_3d = tnt.meter.AverageValueMeter()
    loss_meter_raster = tnt.meter.AverageValueMeter()
    loss_meter_log = tnt.meter.AverageValueMeter()
    loss_meter = tnt.meter.AverageValueMeter()
    loss_meter_ent = tnt.meter.AverageValueMeter()


    pred_pointwise_all = []
    cloud_all = []

    cm.clear()
    cm_2d.clear()
    for index_batch, (cloud, gt, gt_points, yx) in enumerate(loader):
        if PCC.is_cuda:
            gt = gt.cuda()
            gt_points = gt_points.cuda()

        start_encoding_time = time.time()

        pred_pointwise, pred_pointwise_b, pred_pointwise_logits = PCC.run(model, cloud, cm)  # compute the prediction
        pred_pointwise, pred_pointwise_logits = pred_pointwise.detach(), pred_pointwise_logits.detach()
        end_encoding_time = time.time()
        if last_epoch:            # if it is the last epoch, we get time stats info
            print("encoding time", end_encoding_time - start_encoding_time)
        pred_pixels, pred_rasters = project_to_2d(pred_pointwise, pred_pointwise_b, yx, PCC, args)  # compute plot prediction

        flatten_pred = pred_rasters.detach().cpu().numpy().flatten()
        binary_pred = np.zeros(len(flatten_pred))
        binary_pred[flatten_pred>=0.5]=1
        cm_2d.add(gt.cpu().numpy().flatten(), binary_pred)

        # we compute two losses (negative loglikelihood and the absolute error loss for 2 or 3 stratum)
        loss_3d = loss_cross_entropy(pred_pointwise_logits, gt_points)
        loss_raster = loss_bce(pred_rasters, gt, args)
        if args.logl:
            loss_log, likelihood = loss_loglikelihood(pred_pointwise, cloud, params, PCC,
                                                      args)  # negative loglikelihood loss
            loss_meter_log.add(loss_log.item())


        if args.ent:
            loss_e = loss_entropy(pred_pixels)
            loss_meter_ent.add(loss_e.item())

        if args.logl:
            if args.ent:
                loss = loss_3d + args.r * loss_raster + args.m * loss_log + args.e * loss_e

            else:
                loss = loss_3d + args.r * loss_raster + args.m * loss_log

        else:
            if args.ent:
                loss = loss_3d + args.r * loss_raster + args.e * loss_e
            else:
                loss = loss_3d + args.r * loss_raster

        loss_meter_3d.add(loss_3d.item())
        loss_meter_raster.add(loss_raster.item())
        loss_meter.add(loss.item())

        for c in range(len(cloud)):
            cloud[c][:2] = cloud[c][:2] * args.plot_radius + (cloud[c][-4:-2] + args.plot_radius) + args.mean_dataset.reshape(2, 1)
            cloud[c][2] = cloud[c][2] * args.plot_radius


        pred_pointwise_all.append(pred_pointwise)
        cloud_all.append(tuple_of_tensors_to_tensor(cloud).T)
        del cloud, loss, pred_rasters, pred_pointwise, pred_pointwise_b, pred_pointwise_logits, pred_pixels
        gc.collect()


    print_stats(args.stats_file, "Computing for all 3D points", print_to_console=True)
    cm.class_IoU(args)
    cm.overall_accuracy(args)

    print_stats(args.stats_file, "Computing for all 2D rasters", print_to_console=True)
    cm_2d.class_IoU(args)
    cm_2d.overall_accuracy(args)


    pred_pointwise_all = torch.cat((pred_pointwise_all), 0)
    cloud_all = torch.cat((cloud_all), 0)
    reconstruct(pred_pointwise_all, cloud_all, args.stats_path, args, epoch_nb, cm, cm_2d)

    # print(loss_meter_raster.value()[0])
    del loader

    return loss_meter.value()[0], loss_meter_3d.value()[0], loss_meter_raster.value()[0], loss_meter_log.value()[0], loss_meter_ent.value()[0]
