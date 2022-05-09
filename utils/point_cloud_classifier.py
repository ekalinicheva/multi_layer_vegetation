import torch
import numpy as np
import sys
from sklearn.neighbors import NearestNeighbors
np.set_printoptions(threshold=sys.maxsize)


x = [5, 15]
y = [1, 0.5]
slope, intercept = np.polyfit(x, y, 1)


def smart_weight(z):
    if z==0:
        w = 0.5
    elif 0<z<=5:
        w = 1
    elif 5<z<=15:
        10/(z-5)


class PointCloudClassifier:
    """
    The main point cloud classifier Class
    deal with subsampling the tiles to a fixed number of points
    and interpolating to the original clouds
    """

    def __init__(self, args):
        self.subsample_size = args.subsample_size   # number of points to subsample each point cloud in the batches
        self.n_input_feats = 3                      # size of the point descriptors in input
        if len(args.input_feats) > 3:
            self.n_input_feats = len(args.input_feats)
        self.n_class = args.n_class                 # number of classes in the output
        self.is_cuda = args.cuda                    # whether to use GPU acceleration
        self.indices_to_keep = []
        self.smart_sampling = args.smart_sampling
        self.plot_radius = args.plot_radius
        all_possible_feats = "xyzinrd"
        for f in range(len(all_possible_feats)):
            if all_possible_feats[f] in args.input_feats:
                self.indices_to_keep.append(f)


    def run(self, model, clouds, cm):
        """
        INPUT:
        model = the neural network
        clouds = list of n_batch tensors of size [n_feat, n_points_i]: batch of point clouds
        OUTPUT:
        pred = [sum_i n_points_i, n_class] float tensor : prediction for each element of the
             batch in a single tensor
        """

        n_batch = len(clouds)
        # will contain the prediction for all clouds in the batch
        prediction_batch = torch.zeros((self.n_class, 0))
        prediction_batch_logits = torch.zeros((self.n_class, 0))


        # batch_data contain all the clouds in the batch subsampled to self.subsample_size points
        sampled_clouds = torch.Tensor(n_batch, self.n_input_feats, self.subsample_size)
        if self.is_cuda:
            sampled_clouds = sampled_clouds.cuda()
            prediction_batch = prediction_batch.cuda()
            prediction_batch_logits = prediction_batch_logits.cuda()

        # build batches of the same size
        for i_batch in range(n_batch):
            # load the elements in the batch one by one and subsample/ oversample them
            # to a size of self.subsample_size points

            cloud = clouds[i_batch][self.indices_to_keep, :]    # we choose features, but do not keep origins indices
            n_points = cloud.shape[1]  # number of points in the considered cloud
            if n_points > self.subsample_size:

                if self.smart_sampling:
                    # We do smart sampling, so all points between 0 and 5m are always taken, 0-points and points higher than 15m p=0.5.
                    # For points between 5 and 15m we do linear regression, so probs vary between 0.99 and 0.5.
                    z = cloud[2] * self.plot_radius
                    weight = np.full_like(z, 0.5)
                    weight[(z > 0) & (z <= 5)] = 1
                    weight[(z > 5) & (z <= 15)] = z[(z > 5) & (z <= 15)] * (-0.05) + 1.25
                    condition = np.asarray((z > 0) & (z <= 5))
                    selected_points = np.concatenate((np.where(condition)[0],
                                                      np.random.choice(np.where(~condition)[0], self.subsample_size - condition.sum(),
                                                                       replace=False, p=weight[np.where(~condition)] / weight[np.where(~condition)].sum())), 0)

                else:
                    selected_points = np.random.choice(n_points, self.subsample_size,
                                                       replace=False)

            else:
                if self.subsample_size - n_points < n_points:
                    selected_points = np.concatenate((np.arange(n_points),
                                                      np.random.choice(n_points, self.subsample_size - n_points,
                                                           replace=False)), 0)  #we use some points several times, without replacement
                else:
                    selected_points = np.concatenate((np.arange(n_points),
                                                      np.random.choice(n_points, self.subsample_size - n_points,
                                                            replace=True)), 0)
            cloud = cloud[:, selected_points]  # reduce the current cloud to the selected points

            sampled_clouds[i_batch, :, :] = cloud.clone()  # place current sampled cloud in sampled_clouds

        point_logits, point_prediction = model(sampled_clouds)  # classify the batch of sampled clouds
        assert (point_prediction.shape == torch.Size([n_batch, self.n_class, self.subsample_size]))

        # interpolation to original point clouds
        prediction_batches = []
        for i_batch in range(n_batch):
            # get the original point clouds positions
            cloud = clouds[i_batch]
            # and the corresponding sampled batch (only xyz position)
            sampled_cloud = sampled_clouds[i_batch, :3, :]
            n_points = cloud.shape[1]
            knn = NearestNeighbors(1, algorithm='kd_tree').fit(sampled_cloud.cpu().permute(1, 0))
            # select for each point in the original point cloud the closest point in sampled_cloud
            _, closest_point = knn.kneighbors(cloud[:3, :].permute(1, 0).cpu())
            closest_point = closest_point.squeeze()
            prediction_cloud = point_prediction[i_batch, :, closest_point]
            prediction_batch = torch.cat((prediction_batch, prediction_cloud), 1)
            prediction_cloud_logits = point_logits[i_batch, :, closest_point]
            prediction_batch_logits = torch.cat((prediction_batch_logits, prediction_cloud_logits), 1)
            prediction_batches.append(prediction_cloud)
            if cm is not None:
                cm.add(cloud[-3].cpu().numpy(), torch.argmax(prediction_cloud, 0).cpu().detach().numpy())
        return prediction_batch.permute(1, 0), prediction_batches, prediction_batch_logits.permute(1, 0)
