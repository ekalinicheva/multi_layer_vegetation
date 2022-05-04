import warnings
warnings.simplefilter(action='ignore')

import torchnet as tnt
import gc
import time
from torch.utils.tensorboard import SummaryWriter
from model.accuracy import print_stats


import torch.optim as optim
from model.model_pointnet2 import PointNet2
# We import from other files
from data_loader.loader import *
from utils.reproject_to_2d_and_predict_plot_coverage import *
from model.loss_functions import *
from utils.from_cylinder_to_parcel import reconstruct
from utils.point_cloud_classifier import PointCloudClassifier



np.random.seed(42)

def tuple_of_tensors_to_tensor(tuple_of_tensors):
    return torch.cat(list(tuple_of_tensors), dim=1)



def inference(inference_set, args):
    """eval on test set"""
    model = PointNet2(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    PCC = PointCloudClassifier(args)

    try:
        checkpoint = torch.load(args.path_model + "epoch_" + str(args.trained_ep) + ".pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    except:
        model = torch.load(args.path_model + "epoch_" + str(args.trained_ep) +".pt")




    model.eval()

    loader = torch.utils.data.DataLoader(inference_set, collate_fn=cloud_collate,
                                         batch_size=args.batch_size, shuffle=False, drop_last=False)


    pred_pointwise_all = []
    cloud_all = []

    for index_batch, (cloud, _, _, yx, xy_min_cyl) in enumerate(loader):
        start_encoding_time = time.time()

        pred_pointwise, pred_pointwise_b, pred_pointwise_logits = PCC.run(model, cloud, None)  # compute the prediction
        pred_pointwise = pred_pointwise.detach()
        end_encoding_time = time.time()

        for c in range(len(cloud)):
            cloud[c][:2] = cloud[c][:2] * args.plot_radius + (xy_min_cyl[c].reshape(2, 1) + args.plot_radius) + args.mean_dataset.reshape(2, 1)
            cloud[c][2] = cloud[c][2] * args.plot_radius


        pred_pointwise_all.append(pred_pointwise)
        cloud_all.append(tuple_of_tensors_to_tensor(cloud).T)
        del cloud, pred_pointwise, pred_pointwise_b, pred_pointwise_logits
        gc.collect()
    print("encoding time", end_encoding_time - start_encoding_time)


    pred_pointwise_all = torch.cat((pred_pointwise_all), 0)
    cloud_all = torch.cat((cloud_all), 0)
    reconstruct(pred_pointwise_all, cloud_all, args.stats_path, args, 0, None, None)

    del loader