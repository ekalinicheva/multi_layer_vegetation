import numpy as np
from open_ply_all import open_ply_all
from plyfile import PlyData, PlyElement
from config import args

all_points, dataset, mean_dataset, col_full, gt_rasters_dataset, max_dist = open_ply_all(args)


for pl_id in dataset.keys():

    if "d" in args.input_feats:
        ply_array = np.ones(
            len(dataset[pl_id]), dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                                        ('intensity', 'f4'), ('num_returns', 'u1'), ('return_num', 'u1'),
                                        ('dist', 'f4'), ("class", "u1")])
    else:
        ply_array = np.ones(
            len(dataset[pl_id]), dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                                        ('intensity', 'f4'), ('num_returns', 'u1'), ('return_num', 'u1'),
                                        ("class", "u1")])

    ply_array["x"] = dataset[pl_id][:, 0]
    ply_array["y"] = dataset[pl_id][:, 1]
    ply_array["z"] = dataset[pl_id][:, 2]

    # dataset[pl_id][:, :2] = dataset[pl_id][:, :2] - mean_dataset

    ply_array["intensity"] = dataset[pl_id][:, 3]
    ply_array["num_returns"] = dataset[pl_id][:, 4]
    ply_array["return_num"] = dataset[pl_id][:, 5]



    if "d" in args.input_feats:
        ply_array["dist"] = dataset[pl_id][:, -2]

    ply_array["class"] = dataset[pl_id][:, -1]

    ply_file = PlyData([PlyElement.describe(ply_array, 'vertex')], text=True)
    ply_file.write(args.path +
        "/Dataset_6classes/Placette_" + str(pl_id) + ".ply")