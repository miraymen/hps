import configargparse
import numpy as np
import os


def str2bool(inp):
    return inp.lower() in 'true'


def config_parser():
    parser = configargparse.ArgumentParser()

    # Experiment Setup
    parser.add_argument('--config', is_config_file=True, default='configs/garment_render.txt',
                        help='config file path')


    parser.add_argument("--file_name", type=str, default="Ahmad_MPI_BIB_EG_book")
    parser.add_argument("--normal_thresh", type=float, default=0.999)
    parser.add_argument("--window_frames", type=int, default=100, help="window of the frame to be used")

    # values of the init file to be used - parameters corresponding to init position
    parser.add_argument("--imu_filt_start", type=float, default=6.0, help="init value of imu filter coefficient")
    parser.add_argument("--vel_thresh", type=float, default=0.3, help="final velocity threshold")
    parser.add_argument("--refined", type=str2bool, default=True, help="the refined ")

    # values of the velocity based initialization to be used - parameters corresponding to init position
    parser.add_argument("--gamma", type=float, default=6.0, help="frames to look forward and behind to get the current velocity")
    parser.add_argument("--valid_vel_thresh", type=float, default=0.3, help="final velocity threshold, 1000 if percent is to be taken to account, 10000 if no initialization")
    parser.add_argument("--comp_factor", type=float, default=1.02, help="comparison factor to classify a cam velocity as valid")
    parser.add_argument("--comp_factor2", type=float, default=0.98, help="comparison factor to classify a cam velocity as valid")
    parser.add_argument("--percent", type=float, default=0.98, help="what percent of the velocities to consider as valid")

    parser.add_argument("--moveZ", type=float, default=0.05)
    parser.add_argument("--z_sit_min", type=float, default=0.05)
    parser.add_argument("--z_sit_max", type=float, default=0.05)

    # gamma = 6, valid_vel_thresh = 1000, comp_factor = 1.02, comp_factor2 = 0.98, type = 'mean', percent = 0.2
    # values of start and end
    parser.add_argument("--start", type=int, default=0, help="window number from where to start")

    ## PARAMETERS
    parser.add_argument("--wt_ft_vel", type=float, default=300)
    parser.add_argument("--wt_ft_cont", type=float, default=500)
    parser.add_argument("--wt_st_cont", type=float, default=300)
    parser.add_argument("--wt_st_vel", type=float, default=100)
    parser.add_argument("--wt_pen", type=float, default=70)

    parser.add_argument("--trans_smooth_weight", type=float, default=0)
    parser.add_argument("--trans_imu_smth", type=float, default=400)

    parser.add_argument("--smooth_glob", type=float, default=100)
    parser.add_argument("--v2v_weight", type=float, default=0,
                        help="Weight of the vertex to weight loss used for smoothing")

    parser.add_argument("--trans_max", type=float, default=0.0)
    parser.add_argument("--rot_max", type=float, default=0.0,
                        help="max angle in radians that is allowed before penalization about 2 degree here")

    parser.add_argument("--init_vel_weight", type=float, default=0)
    parser.add_argument("--init_trans_weight", type=float, default=400)
    parser.add_argument("--init_rot_weight", type=float, default=100)
    parser.add_argument("--init_v2v_weight", type=float, default=0)
    parser.add_argument("--head_orientation", type=float, default=0)

    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learn_rate", type=float, default=0.001)

    parser.add_argument("--opt_vars", type=str, default="trans_glob",
                        help="trans|trans_allpose|trans_glob")

    parser.add_argument("--radius", type=float, default=4)

    return parser


def get_config():
    parser = config_parser()
    cfg = parser.parse_args()

    return cfg
