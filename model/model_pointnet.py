# %%
import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)



class PointNet(nn.Module):
    """
    The PointNet network for semantic segmentation
    """

    def __init__(self, MLP_1, MLP_2, MLP_3, args):
        """
        initialization function
        MLP_1, LMP2 and MLP3 = int array, size of the layers of multi-layer perceptrons
        for example MLP1 = [32,64]
        n_class = int,  the number of class
        input_feat = int, number of input feature
        subsample_size = int, number of points to which the tiles are subsampled

        """

        super(PointNet, self).__init__()  # necessary for all classes extending the module class
        self.is_cuda = args.cuda
        self.subsample_size = args.subsample_size
        self.n_class = args.n_class
        self.drop = args.drop
        self.soft = args.soft
        self.input_feat = args.n_input_feats

        # since we don't know the number of layers in the MLPs, we need to use loops
        # to create the correct number of layers
        m1 = MLP_1[-1]  # size of the first embeding F1
        m2 = MLP_2[-1]  # size of the second embeding F2

        # build MLP_1: input [input_feat x n] -> f1 [m1 x n]
        modules = []
        for i in range(len(MLP_1)):  # loop over the layer of MLP1
            # note: for the first layer, the first in_channels is feature_size
            modules.append(
                nn.Conv1d(in_channels=MLP_1[i - 1] if i > 0 else self.input_feat, out_channels=MLP_1[i], kernel_size=1))
            modules.append(nn.BatchNorm1d(MLP_1[i]))
            modules.append(nn.ReLU(True))
        # this transform the list of layers into a callable module
        self.MLP_1 = nn.Sequential(*modules)

        # build MLP_2: f1 [m1 x n] -> f2 [m2 x n]
        modules = []
        for i in range(len(MLP_2)):
            modules.append(nn.Conv1d(in_channels=MLP_2[i - 1] if i > 0 else m1, out_channels=MLP_2[i], kernel_size=1))
            modules.append(nn.BatchNorm1d(MLP_2[i]))
            modules.append(nn.ReLU(True))
        self.MLP_2 = nn.Sequential(*modules)

        # build MLP_3: f1 [(m1 + m2) x n] -> output [k x n]
        modules = []
        for i in range(len(MLP_3)):
            modules.append(
                nn.Conv1d(in_channels=MLP_3[i - 1] if i > 0 else (m1 + m2), out_channels=MLP_3[i], kernel_size=1))
            modules.append(nn.BatchNorm1d(MLP_3[i]))
            modules.append(nn.ReLU(True))
        # note: the last layer do not have normalization nor activation
        modules.append(nn.Dropout(p=self.drop))
        modules.append(nn.Conv1d(MLP_3[-1], self.n_class, 1))
        self.MLP_3 = nn.Sequential(*modules)

        self.maxpool = nn.MaxPool1d(self.subsample_size)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        if self.is_cuda:
            self = self.cuda()

        self.apply(init_weights)


    def forward(self, input):
        """
        the forward function producing the embeddings for each point of 'input'
        input = [n_batch, input_feat, subsample_size] float array: input features
        output = [n_batch,n_class, subsample_size] float array: point class logits
        """
        # print(input.size())
        if self.is_cuda:
            input = input.cuda()
        f1 = self.MLP_1(input)
        f2 = self.MLP_2(f1)
        G = self.maxpool(f2)
        Gf1 = torch.cat((G.repeat(1, 1, self.subsample_size), f1), 1)
        out_pointwise = self.MLP_3(Gf1)
        if self.soft:
            out_pointwise = self.softmax(out_pointwise)
        else:
            out_pointwise = self.sigmoid(out_pointwise)
        return out_pointwise
