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



def inference(model, PCC, inference_set, params, args):
    """eval on test set"""

    model.eval()

    loader = torch.utils.data.DataLoader(inference_set, collate_fn=cloud_collate,
                                         batch_size=args.batch_size, shuffle=False, drop_last=False)


    pred_pointwise_all = []
    cloud_all = []

    for index_batch, (cloud, _, _, yx) in enumerate(loader):

        start_encoding_time = time.time()

        pred_pointwise, pred_pointwise_b, pred_pointwise_logits = PCC.run(model, cloud, None)  # compute the prediction
        pred_pointwise = pred_pointwise.detach()
        end_encoding_time = time.time()
        print("encoding time", end_encoding_time - start_encoding_time)

        for c in range(len(cloud)):
            cloud[c][:2] = cloud[c][:2] * args.plot_radius + (cloud[c][-4:-2] + args.plot_radius) + args.mean_dataset.reshape(2, 1)
            cloud[c][2] = cloud[c][2] * args.plot_radius


        pred_pointwise_all.append(pred_pointwise)
        cloud_all.append(tuple_of_tensors_to_tensor(cloud).T)
        del cloud, pred_pointwise, pred_pointwise_b, pred_pointwise_logits
        gc.collect()


    pred_pointwise_all = torch.cat((pred_pointwise_all), 0)
    cloud_all = torch.cat((cloud_all), 0)
    reconstruct(pred_pointwise_all, cloud_all, args.stats_path, args, 0, None, None)

    del loader