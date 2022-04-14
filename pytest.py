import os
import torch
# args_log_reg = [0.5, 0.25, 0.75, 0.1, 1]
# args_log_reg = [0.1, 0.25, 0.5, 0.75]
#
# args_raster_reg = [0.1, 0.25, 0.5]
# # args_raster_reg = [0.25, 0.5]
# # args_raster_reg = [1, 0.75]
#
# for args1 in args_raster_reg:
#     for args2 in args_log_reg:
#         torch.cuda.empty_cache()
#         command = 'python main_DL.py --r ' + str(args1) + " --m " + str(args2)
#         os.system(command)


# # subsample_size = [4096*4, 4096*3]
# r_reg = [0.5, 0.75, 1, 1.5, 0.5, 2]
# radius = ["0.2 0.4"]
# # r_num_pts = ["1024 128", "512 128", "256 64", "2048 256"]
# r_num_pts = ["8192 2048", "4096 1024"]


#
# commands = ['python main_DL.py --pixel_size 2 --regular_grid_size 4  --sample_grid_size 2', 'python main_DL.py --pixel_size 0.5 --n_epoch 100', 'python main_DL.py --pixel_size 1', 'python main_DL.py --pixel_size 0.25', 'python main_DL.py --m 0.5',
#             'python main_DL.py --subsample_size 8192', 'python main_DL.py --plot_radius 10 --regular_grid_size 10 --sample_grid_size 2', 'python main_DL.py --plot_radius 2  --regular_grid_size 2 --sample_grid_size 1']
#

commands = ['python main_DL.py --m 0 --n_epoch 75',
            'python main_DL.py --pixel_size 0.5 --n_epoch 100'
            ]
# commands = ['python main_DL.py --plot_radius 10 --regular_grid_size 10 --sample_grid_size 2 --nbr_training_samples 300',
#             'python main_DL.py --plot_radius 2  --regular_grid_size 2 --sample_grid_size 1 --subsample_size 8192 --nbr_training_samples 2000 --batch_size 20',
#             'python main_DL.py --pixel_size 0.5 --r 0',
#             'python main_DL.py --pixel_size 0.5 --n_epoch 100',
#             ]



for command in commands:
    torch.cuda.empty_cache()
    print(command)
    os.system(command)

# commands = ['python main_DL.py --plot_radius 2  --regular_grid_size 2 --sample_grid_size 1 --subsample_size 8192 --nbr_training_samples 2000 --batch_size 20',
#             'python main_DL.py --pixel_size 0.5 --n_epoch 100',
#             ]
#
#
#
# for command in commands:
#     torch.cuda.empty_cache()
#     print(command)
#     os.system(command)

exit()

# subsample_size = [4096*4, 4096*3]
r_reg = [1, 0.75]
radius = ["0.1 0.4 0.8", "0.1 0.25 0.5"]
# r_num_pts = ["1024 128", "512 128", "256 64", "2048 256"]
r_num_pts = ["12288 4096 1024", "8192 2048 512"]


# radius = ["0.1 0.2 0.4", "0.2 0.4 0.75"]
# r_num_pts = ["2048 1024 128", "1024 256 64", "512 128 32", "256 64 16"]


for args1 in r_reg:
    for args2 in radius:
        for args3 in r_num_pts:
            torch.cuda.empty_cache()
            print('python main_DL.py --r ' + str(args1) + " --rr " + str(args2) + " --r_num_pts " + str(args3))
            command = 'python main_DL.py --r ' + str(args1) + " --rr " + str(args2) + " --r_num_pts " + str(args3)
            os.system(command)

# subsample_size = [4096*5, 4096*6]
# radius = ["0.1 0.2", "0.2 0.5", "0.05 0.4"]
# r_num_pts = ["1024 256", "512 256"]
#
#
# for args1 in subsample_size:
#     for args3 in r_num_pts:
#         for args2 in radius:
#             torch.cuda.empty_cache()
#             command = 'python main_DL.py --subsample_size ' + str(args1) + " --rr " + str(args2) + " --r_num_pts " + str(args3)
#             os.system(command)


