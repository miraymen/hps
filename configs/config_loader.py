import configargparse
import numpy as np
import os


def str2bool(inp):
    return inp.lower() in 'true'


def config_parser():
    parser = configargparse.ArgumentParser()

    # Experiment Setup
    parser.add_argument('--config', is_config_file=True, help='config file path')

    # Sequence and scene parameters
    parser.add_argument("--file_name", type=str, default="Ahmad_MPI_BIB_EG_book")
    parser.add_argument("--normal_thresh", type=float, default=0.999)
    parser.add_argument("--window_frames", type=int, default=100, help="window of the frame to be used")
    parser.add_argument("--moveZ", type=float, default=0.05)
    parser.add_argument("--start", type=int, default=0, help="window number from where to start")
    parser.add_argument("--radius", type=float, default=4)

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

    # Sitting parameters
    parser.add_argument("--z_sit_min", type=float, default=0.05)
    parser.add_argument("--z_sit_max", type=float, default=0.05)

    # Optimization parameters - global
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learn_rate", type=float, default=0.001)
    parser.add_argument("--opt_vars", type=str, default="trans_glob",
                        help="trans|trans_allpose|trans_glob")

    # Optimization parameters - main
    parser.add_argument("--wt_ft_vel", type=float, default=300)
    parser.add_argument("--wt_ft_cont", type=float, default=500)
    parser.add_argument("--wt_st_cont", type=float, default=300)
    parser.add_argument("--wt_st_vel", type=float, default=100)
    parser.add_argument("--wt_pen", type=float, default=70)
    parser.add_argument("--wt_rot_smth", type=float, default=70)
    parser.add_argument("--wt_trans_imu_smth", type=float, default=70)
    parser.add_argument("--wt_head_or", type=float, default=70)
    parser.add_argument("--wt_pose_prior", type=float, default=70)

    # Optimization parameters - auxillary
    parser.add_argument("--rot_max", type=float, default=0.03,
                        help="max angle in radians that is allowed before penalization about 2 degree here")

    # Optimization parameters - connecting
    parser.add_argument("--init_vel_weight", type=float, default=0)
    parser.add_argument("--init_trans_weight", type=float, default=400)
    parser.add_argument("--init_rot_weight", type=float, default=100)

    return parser


def get_config():
    parser = config_parser()
    cfg = parser.parse_args()

    return cfg
