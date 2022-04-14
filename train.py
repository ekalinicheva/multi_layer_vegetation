import warnings
warnings.simplefilter(action='ignore')


import torchnet as tnt
import gc
from torch.utils.tensorboard import SummaryWriter
from model.accuracy import print_stats


# We import from other files
from data_loader.loader import *
from utils.reproject_to_2d_and_predict_plot_coverage import *
from model.loss_functions import *

np.random.seed(42)

def train(model, PCC, train_set, params, optimizer, cm, cm_2d, args):
    """train for one epoch"""
    model.train()
    if args.nbr_training_samples is not None:
        if args.nbr_training_samples > len(train_set):
            indices = np.random.choice(len(train_set), args.nbr_training_samples, replace=True)
        else:
            indices = np.random.choice(len(train_set), args.nbr_training_samples, replace=False)
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        # the loader function will take care of the batching
        loader = torch.utils.data.DataLoader(train_set, collate_fn=cloud_collate, sampler=sampler,
                                             batch_size=args.batch_size, drop_last=True)
    else:
        loader = torch.utils.data.DataLoader(train_set, collate_fn=cloud_collate,
                                             batch_size=args.batch_size, shuffle=True, drop_last=True)

    # will keep track of the loss
    loss_meter = tnt.meter.AverageValueMeter()
    loss_meter_3d = tnt.meter.AverageValueMeter()
    loss_meter_raster = tnt.meter.AverageValueMeter()
    loss_meter_log = tnt.meter.AverageValueMeter()
    loss_meter_ent = tnt.meter.AverageValueMeter()


    cm.clear()
    cm_2d.clear()
    for index_batch, (cloud, gt, gt_points, yx) in enumerate(loader):

        if PCC.is_cuda:
            gt = gt.cuda()
            gt_points = gt_points.cuda()

        optimizer.zero_grad()  # put gradient to zero
        pred_pointwise, pred_pointwise_b, pred_pointwise_logits = PCC.run(model, cloud, cm)  # compute the pointwise prediction
        pred_pixels, pred_rasters = project_to_2d(pred_pointwise, pred_pointwise_b, yx, PCC, args)  # compute plot prediction
        # we compute two losses (negative loglikelihood and the absolute error loss for 2 or 3 stratum)
        flatten_pred = pred_rasters.detach().cpu().numpy().flatten()
        binary_pred = np.zeros(len(flatten_pred))
        binary_pred[flatten_pred>=0.5]=1
        cm_2d.add(gt.cpu().numpy().flatten(), binary_pred)

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

        loss.backward()
        optimizer.step()

        loss_meter_3d.add(loss_3d.item())
        loss_meter_raster.add(loss_raster.item())
        loss_meter.add(loss.item())
        gc.collect()

    print_stats(args.stats_file, "Computing for all 3D points", print_to_console=True)
    cm.class_IoU(args)
    cm.overall_accuracy(args)

    print_stats(args.stats_file, "Computing for all 2D rasters", print_to_console=True)
    cm_2d.class_IoU(args)
    cm_2d.overall_accuracy(args)


    del loader, pred_pointwise, pred_pointwise_b, pred_pointwise_logits, pred_pixels, pred_rasters, cloud, gt, gt_points
    return loss_meter.value()[0], loss_meter_3d.value()[0], loss_meter_raster.value()[0], loss_meter_log.value()[0], loss_meter_ent.value()[0]