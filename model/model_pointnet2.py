import torch
import torch.nn as nn
import os
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_interpolate, PointConv, fps, radius, global_max_pool
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN


# Architecture is adapated from the following example: https://github.com/rusty1s/pytorch_geometric/blob/master/examples/pointnet2_segmentation.py
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn, radius_num_point=32):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.radius_num_point = radius_num_point
        self.conv = PointConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(
            pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=self.radius_num_point
        )
        edge_index = torch.stack([col, row], dim=0)
        # x_dst = None if x is None else x[idx]
        # x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(
        *[
            Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
            if batch_norm
            else Seq(Lin(channels[i - 1], channels[i]), ReLU())
            for i in range(1, len(channels))
        ]
    )


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super(FPModule, self).__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip


class PointNet2(torch.nn.Module):
    def __init__(self, args):
        super(PointNet2, self).__init__()
        self.is_cuda = args.cuda
        self.subsample_size = args.subsample_size
        self.n_class = args.n_class
        self.drop = args.drop
        self.n_input_feats = args.n_input_feats - 2  # - x and y
        self.set_patience_attributes(args)
        self.log_embeddings = args.log_embeddings
        self.nbr_mlp = len(args.r_num_pts) + 1
        self.args = args
        ndim = 3


        if self.nbr_mlp==3:
            MLP1 = [self.n_input_feats + ndim, 32, 32]
            MLP2 = [MLP1[-1] + ndim, 64, 64]
            MLP3 = [MLP2[-1] + ndim, 64, 128]
            self.sa1_module = SAModule(args.ratio, args.rr[0], MLP(MLP1), args.r_num_pts[0])
            self.sa2_module = SAModule(args.ratio, args.rr[1], MLP(MLP2), args.r_num_pts[1])
            # self.sa3_module = SAModule(args.ratio[2], args.rr[2], MLP(MLP3), args.r_num_pts[2])
            self.sa3_module = GlobalSAModule(MLP(MLP3))

            MLP3_fp = [MLP3[-1] + MLP2[-1], 64, 64]
            MLP2_fp = [MLP3_fp[-1] + MLP1[-1], 64, 32]
            MLP1_fp = [MLP2_fp[-1] + self.n_input_feats, 32, 32]
            self.fp3_module = FPModule(1, MLP(MLP3_fp))
            self.fp2_module = FPModule(3, MLP(MLP2_fp))
            self.fp1_module = FPModule(3, MLP(MLP1_fp))
        else:   # if equal 4
            MLP1 = [self.n_input_feats + ndim, 16, 32]
            MLP2 = [MLP1[-1] + ndim, 32, 64]
            MLP3 = [MLP2[-1] + ndim, 64, 128]
            MLP4 = [MLP3[-1] + ndim, 128, 128]
            self.sa1_module = SAModule(args.ratio[0], args.rr[0], MLP(MLP1), args.r_num_pts[0])
            self.sa2_module = SAModule(args.ratio[1], args.rr[1], MLP(MLP2), args.r_num_pts[1])
            self.sa3_module = SAModule(args.ratio[2], args.rr[2], MLP(MLP3), args.r_num_pts[2])
            self.sa4_module = GlobalSAModule(MLP(MLP4))

            MLP4_fp = [MLP4[-1] + MLP3[-1], 128, 128]
            MLP3_fp = [MLP4_fp[-1] + MLP2[-1], 128, 64]
            MLP2_fp = [MLP3_fp[-1] + MLP1[-1], 64, 32]
            MLP1_fp = [MLP2_fp[-1] + self.n_input_feats, 32, 32, 32]
            self.fp4_module = FPModule(1, MLP(MLP4_fp))
            self.fp3_module = FPModule(3, MLP(MLP3_fp))
            self.fp2_module = FPModule(3, MLP(MLP2_fp))
            self.fp1_module = FPModule(3, MLP(MLP1_fp))

        self.lin1 = torch.nn.Linear(MLP1_fp[-1], MLP1_fp[-1])
        self.lin2 = torch.nn.Linear(MLP1_fp[-1], self.n_class)

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        if self.is_cuda:
            self = self.cuda()

    def forward(self, cloud_data):
        # xyz = self.get_long_form(cloud_data["xyz"])
        # cloud = cloud_data["cloud"]

        xyz = self.get_long_form(cloud_data[:, :3, :])
        # REMOVE x and y from consideration
        cloud_feat = cloud_data[:, 2:self.n_input_feats+2, :].clone()
        # cloud_feat[:, 0, :] = cloud_feat[:, 0, :] * self.args.plot_radius / self.args.z_max

        batch_size = cloud_feat.shape[0]
        cloud_feat = self.get_long_form(cloud_feat)
        batch = torch.from_numpy(
            np.concatenate(
                [np.full((self.subsample_size, 1), b) for b in range(batch_size)]
            ).squeeze()
        )

        if self.is_cuda:
            sa0_out = (
                cloud_feat.cuda(),
                xyz.cuda(),
                batch.cuda(),
            )
        else:
            sa0_out = (
                cloud_feat,
                xyz,
                batch,
            )

        if self.nbr_mlp == 3:
            sa1_out = self.sa1_module(*sa0_out)
            sa2_out = self.sa2_module(*sa1_out)
            sa3_out = self.sa3_module(*sa2_out)

            if self.log_embeddings:
                self.last_G_tensor = sa3_out[0]

            fp3_out = self.fp3_module(*sa3_out, *sa2_out)
            fp2_out = self.fp2_module(*fp3_out, *sa1_out)
            x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        else:
            sa1_out = self.sa1_module(*sa0_out)
            sa2_out = self.sa2_module(*sa1_out)
            sa3_out = self.sa3_module(*sa2_out)
            sa4_out = self.sa4_module(*sa3_out)

            if self.log_embeddings:
                self.last_G_tensor = sa4_out[0]

            fp4_out = self.fp4_module(*sa4_out, *sa3_out)
            fp3_out = self.fp3_module(*fp4_out, *sa2_out)
            fp2_out = self.fp2_module(*fp3_out, *sa1_out)
            x, _, _ = self.fp1_module(*fp2_out, *sa0_out)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.drop, training=self.training)
        scores_pointwise = self.lin2(x)

        proba_pointwise = self.softmax(scores_pointwise)


        return self.get_batch_format(scores_pointwise), self.get_batch_format(proba_pointwise)

    @staticmethod
    def get_long_form(data):
        """Get tensor of shape (N*B,f) from shape (B,f,N)"""
        return torch.cat(list(data), 1).transpose(1, 0)

    def get_batch_format(self, data):
        """Get tensor of shape (B,f,N) from shape (N*B,f), dividing by nb of points in each cloud."""
        data = torch.split(data, self.subsample_size, dim=0)
        return torch.stack(data).transpose(1, 2)

    def set_patience_attributes(self, args):
        """Reset patience. Useful when we load a pretrained model."""
        self.stopped_early = False
        self.best_metric_value = 10 ** 6
        self.best_metric_epoch = 1
        self.patience_in_epochs = args.patience_in_epochs

    def stop_early(self, val_metric, epoch, args):
        """Save best model state until now, based on a validation metric to minimize, if no improvement over n epochs."""
        if val_metric < self.best_metric_value:
            self.best_metric_value = val_metric
            self.best_metric_epoch = epoch
            self.save_state(args)
        else:
            if epoch < args.epoch_to_start_early_stop:
                return False
            if epoch >= self.best_metric_epoch + self.patience_in_epochs:
                self.stopped_early = True
                return True
        return False

    def save_state(self, args):
        """Save model state in stats_path."""
        checkpoint = {
            "best_metric_epoch": self.best_metric_epoch,
            "state_dict": self.state_dict(),
            "best_metric_value": self.best_metric_value,
        }

        crossvalidating = args.current_fold_id > 0
        save_path = os.path.join(
            args.stats_path,
            f"PCC_model_{'fold_n='+str(args.current_fold_id) if crossvalidating else 'full'}.pt",
        )
        torch.save(checkpoint, save_path)

    def load_state(self, save_path):
        """Load model state from a path."""
        if self.is_cuda:
            checkpoint = torch.load(save_path)
        else:
            checkpoint = torch.load(save_path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.best_metric_epoch = checkpoint["best_metric_epoch"]
        self.best_metric_value = checkpoint["best_metric_value"]
        return self

    def load_best_state(self, args):
        """Load best model state from early stopping checkpoint. Does not load the optimizer state."""
        crossvalidating = args.current_fold_id > 0
        save_path = os.path.join(
            args.stats_path,
            f"PCC_model_{'fold_n='+str(args.current_fold_id) if crossvalidating else 'full'}.pt",
        )
        self.load_state(save_path)
        return self