import numpy as np
from sklearn.metrics import confusion_matrix
from model.accuracy import print_stats
import matplotlib.pyplot as plt



class ConfusionMatrix:
    def __init__(self, class_names):
        self.n_class = len(class_names)
        self.CM = np.zeros((self.n_class, self.n_class), dtype=int)
        self.class_names = class_names



    def clear(self):
        self.CM = np.zeros((self.n_class, self.n_class))

    def add(self, gt, pred, nodata=-1):
        self.CM += confusion_matrix(gt[gt != nodata], pred[gt != nodata], labels=list(range(self.n_class)))

    def overall_accuracy(self, args=None):  # percentage of correct classification
        if self.CM.shape[0]>2:  #for 3D classification, we don't consider ground and shrub
            ind = [0, 2, 3, 4, 5]
            oa = round(100 * np.delete(np.delete(self.CM, 1, axis=1), 1, axis=0).trace() / np.delete(np.delete(self.CM, 1, axis=0), 1, axis=1).sum(), 2)
        else:
            oa = round(100 * self.CM.trace() / self.CM.sum(), 2)
        if args is not None:
            print_stats(args.stats_file, ("OA " + str(oa) + "%"), print_to_console=True)
        else:
            print("OA " + str(oa) + "%")
        return 100 * self.CM.trace() / self.CM.sum()

    def class_IoU(self, args=None):
        ious = np.full(self.n_class, 0.)
        for i_class in range(self.n_class):
            ious[i_class] = self.CM[i_class, i_class] / \
                            (-self.CM[i_class, i_class] \
                             + self.CM[i_class, :].sum() \
                             + self.CM[:, i_class].sum())

        # do not count classes that are not present in the dataset in the mean IoU
        if len(ious)>2:   #for 3D classification, we don't consider ground and shrub
            ind = [0, 2, 3, 4, 5] if self.n_class==6 else [0, 2, 3, 4, 5, 6]
            meanIoU = round(100 * (ious[ind]).mean(), 2)
        else:
            meanIoU = round(100 * ious.mean(), 2)

        if args is not None:
            print_stats(args.stats_file, (' | '.join('{} : {:3.2f}%'.format(name, 100 * iou) for name, iou in zip(self.class_names, ious))), print_to_console=True)
            print_stats(args.stats_file, ("meanIoU " + str(meanIoU) + "%"), print_to_console=True)
        else:
            print(' | '.join('{} : {:3.2f}%'.format(name, 100 * iou) for name, iou in zip(self.class_names, ious)))
            print("meanIoU " + str(meanIoU) + "%")

        return 100 * np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()

    def plot_matrix(self, path):
        plt.rcParams.update({'font.size': 16})

        plt.rcParams["font.family"] = "Times New Roman"


        ind = [0, 2, 3, 4, 5]
        matrix = self.CM[ind][:, ind]
        print(matrix)
        matrix_norm = matrix.astype('float') / matrix.sum(axis=-1)[:, np.newaxis]
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(matrix_norm, cmap=plt.cm.YlGn)
        ax.set(xlabel="Predicted", ylabel='Ground Truth')
        # plt.title("Confusion matrix", y=1.225)
        fig.colorbar(cax)
        labels = ["ground", "understory", 'deciduous', "coniferous", 'stem']
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xticklabels(labels, rotation=45)

        # plt.xticks(rotation=45)
        ax.tick_params(length=0)



        for i, j in ((x, y) for x in range(len(matrix))
                     for y in range(len(matrix))):
            if matrix[j][i]>=10000:
                ax.annotate(str(int(matrix[j][i]/1000))+"k", xy=(i, j), ha='center', va='center')
            else:
                ax.annotate(str(matrix[j][i]), xy=(i, j), ha='center', va='center')

        # plt.ylabel("Ground Truth")
        # plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(path + 'Confusion_matrix.pdf', format='pdf', dpi=600)
        plt.close('all')