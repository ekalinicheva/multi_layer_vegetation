import argparse
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold
import time
import pickle

    # We import from other files
from train import *
from testing import *
from model.accuracy import *
from model.model_pointnet import PointNet
from model.model_pointnet2 import PointNet2
from utils.point_cloud_classifier import PointCloudClassifier
from utils.useful_functions import *
from utils.confusion_matrix import ConfusionMatrix


def train_full(args, params, train_set, test_set):
    """The full training loop"""
    # initialize the model
    # model = PointNet(args.MLP_1, args.MLP_2, args.MLP_3, args)
    if args.train_model:
        model = PointNet2(args)
    else:
        model = torch.load(args.path_model + "epoch_" + str(args.trained_ep) +".pt")
    print("model done")
    # writer = SummaryWriter(results_path + "runs/"+run_name + "fold_" + str(fold_id) +"/")
    writer = SummaryWriter(args.results_path + "runs/" + args.run_name + "/")

    print('Total number of parameters: {}'.format(sum([p.numel() for p in model.parameters()])))
    print(model)

    # args.lr = model.lea

    # define the classifier

    class_names = ["ground", "shrub", "understory", "leaves", "pines", "stems"] if args.n_class==6 else ["ground", "shrub", "understory", "leaves", "pines", "alder", "stems"]
    class_names_2d = ["nothing", "vegetation"]
    cm = ConfusionMatrix(class_names)
    cm_2d = ConfusionMatrix(class_names_2d)

    PCC = PointCloudClassifier(args)


    # define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)

    if args.n_epoch>0:
        for i_epoch in range(args.n_epoch):
            if not args.train_model:
                i_epoch += args.trained_ep
            scheduler.step()
            # train one epoch
            train_losses = train(model, PCC, train_set, params, optimizer, cm, cm_2d, args)
            writer = write_to_writer(writer, args, i_epoch, train_losses, train=True)
            gc.collect()

            if (i_epoch + 1) % args.n_epoch_test == 0:
                start_time = time.time()
                if (i_epoch + 1) == args.n_epoch + args.trained_ep:  # if last epoch, we creare 2D images with points projections
                    test_losses = evaluate(model, PCC, test_set, params, args, i_epoch+1, cm, cm_2d, last_epoch=True)
                else:
                    test_losses = evaluate(model, PCC, test_set, params, args, i_epoch+1, cm, cm_2d)
                gc.collect()
                writer = write_to_writer(writer, args, i_epoch, test_losses, train=False)

                torch.save(model, args.stats_path + "/epoch_" + str(i_epoch + 1) + '.pt')
                end_time = time.time()
                print("Time", end_time - start_time)
    else:
        test_losses = evaluate(model, PCC, test_set, params, args, args.trained_ep, cm, cm_2d)
        train_losses = None

    writer.flush()

    final_train_losses_list = train_losses
    final_test_losses_list = test_losses
    return model, final_train_losses_list, final_test_losses_list
