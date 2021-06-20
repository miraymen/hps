'''
Transform IMU trajectories by finding the orientation correction
Plug this imu trajectory into the camera localization trajectory
Read the unfiltered json files
Run velocity filter to get valid frames
Do not interpolate yet
Rotate IMU positions

conda :py3d

Requires:
1. bake file
2. json cam localization file
'''
import argparse
import sys
import numpy as np
import trimesh
import cv2

sys.path.append("../")
from global_vars import *
from imu_based_filter import ImuFilterPkl


class Solver:
    def __init__(self, velocity_threshold,
                 replace_cam_z,
                 cor_type,
                 imu_filt_fact_strt,
                 refine):

        self.params = 'filt_strt-{}_vel_filt_th-{}_repcamz-{}_cortype-{}'.format(imu_filt_fact_strt,
                                                                                              velocity_threshold,
                                                                                              replace_cam_z,
                                                                                              cor_type
                                                                                              )

        self.velocity_threshold = velocity_threshold


        self.replace_cam_z = replace_cam_z

        self.cor_type = cor_type

        self.imu_filt_fact_strt = imu_filt_fact_strt

        self.refine = refine

    def ranges(self, nums):
        nums = sorted(set(nums))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        return list(zip(edges, edges))

    def fill_optimize(self):

        ranges_numbers = self.ranges(self.invalid_indices)
        print(len(ranges_numbers))
        rot_trans = np.identity(3)

        for i, one_range in enumerate(ranges_numbers):
            start = one_range[0] - 1
            end = one_range[1] + 1
            if not end == self.end_index:
                target_vec = np.copy(self.cam_trans[end] - self.cam_trans[start])
                inp_vec = np.copy(self.imu_trans[end] - self.imu_trans[start])
                inp_vec[2] = 0
                target_vec[2] = 0
                rot_trans = rot_mat(inp_vec, target_vec)

            for i in range(start + 1, end):
                imu_vel = np.copy(self.imu_vel[i - 1])
                imu_vel[2] = 0
                self.cam_trans[i] = self.cam_trans[i - 1] + np.matmul(rot_trans, imu_vel)

    def refine_vals(self):

        ranges_numbers = self.ranges(self.invalid_indices)

        rot_trans = np.identity(3)

        for i, one_range in enumerate(ranges_numbers):
            # print(i)
            start = one_range[0] - 1
            end = one_range[1] + 1
            if not end == self.end_index:

                target_vec = np.copy(self.cam_trans[end] - self.cam_trans[start])
                inp_vec = np.copy(self.imu_trans[end] - self.imu_trans[start])
                inp_vec[2] = 0
                target_vec[2] = 0

                rot_trans = rot_mat(inp_vec, target_vec)
                mult_fact = np.linalg.norm(target_vec, ord=2) / np.linalg.norm(inp_vec, ord=2)

                for i in range(start + 1, end):
                    imu_vel = np.copy(self.imu_vel[i - 1])
                    imu_vel[2] = 0
                    self.cam_trans[i, 0:2] = self.cam_trans[i - 1, 0:2] + mult_fact * np.matmul(rot_trans, imu_vel)[0:2]
            else:
                print('last invalid')

                for i in range(start + 1, end):
                    imu_vel = np.copy(self.imu_vel[i - 1])
                    imu_vel[2] = 0
                    self.cam_trans[i, 0:2] = self.cam_trans[i - 1, 0:2] + np.matmul(rot_trans, imu_vel)[0:2]


    def run_one(self, file_name):
        """
        Args:
            file_name:
        Returns:
        """
        self.file_name = file_name

        self.traj_path = FILTER_PATH + 'trajectories_new/' + file_name + '/' + self.params + '/'

        self.pose_traj_path = FILTER_PATH + 'pose_traj_files_new/' + file_name + '/' + self.params + '/'

        os.makedirs(self.traj_path, exist_ok=True)
        os.makedirs(self.pose_traj_path, exist_ok=True)

        self.cam_trans, self.cam_pose, self.imu_trans, self.imu_pose = sync_data(file_name, type='bake')

        self.imu_pose_new = np.copy(self.imu_pose)

        self.end_index = self.cam_trans.shape[0]

        self.imu_vel = compute_velocity(self.imu_trans)

        big_iter = 1

        self.invalid_indices = []

        self.max_vel = 100

        # ==========================
        #    Start of filtering
        # ==========================

        while self.max_vel > self.velocity_threshold and big_iter < 300:

            print('Iteration : ', big_iter)

            # ==================
            #   Filtering Part
            # ==================

            factor = np.floor(big_iter / 10) + self.imu_filt_fact_strt

            filter = ImuFilterPkl(factor=factor, cor_type=self.cor_type)

            if big_iter > 1:
                self.invalid_indices_iter_p = np.copy(self.invalid_indices_iter)

            self.cam_trans, self.invalid_indices_iter, self.max_vel = filter.run_array(self.imu_trans,
                                                                                           self.cam_trans)

            self.invalid_indices = np.union1d(self.invalid_indices, self.invalid_indices_iter).astype(int)

            self.valid_indices = list(set(range(0, self.end_index)) - set(self.invalid_indices))

            print('Invalid indices: {}/{} '.format(len(self.invalid_indices), self.end_index))

            print('Current iter invalid : {}'.format(len(self.invalid_indices_iter)))

            # If the first index is filtered out, use IMU data straightaway
            if np.linalg.norm(self.cam_trans[0, :]) == 0:
                self.cam_trans[0, :] = self.imu_trans[0, :]

            # ===============================
            #    Fill missing values part
            # ===============================

            self.fill_optimize()

            if self.replace_cam_z:
                self.cam_trans[:, 2] = self.imu_trans[:, 2]

            # =================
            #   Visualization
            # =================

            colors = np.zeros(self.cam_trans.shape)
            colors[self.invalid_indices, 0] = 255
            colors[self.invalid_indices, 1] = 255
            colors[0:50, 0] = 255
            colors[0:50, 1] = 0

            trimesh.points.PointCloud(vertices=self.cam_trans, colors=colors).export(
                self.traj_path + '/iter_{}.ply'.format(big_iter))

            self.max_vel, max_vel_ind = filter.get_max_vel(self.cam_trans)
            print('Max Velocity', self.max_vel)
            print('Max velocity at', max_vel_ind)

            big_iter += 1

            save_dict = {
                'poses': self.imu_pose_new,
                'transes': self.cam_trans,
                'invalid_indices': self.invalid_indices
            }

            save_file = self.pose_traj_path + 'unrefined.pkl'
            save_pkl(save_file, save_dict)

        # ====================================
        #   At the end save unrefined file
        # ====================================

        save_dict = {
                'poses': self.imu_pose_new,
                'transes': self.cam_trans,
                'invalid_indices': self.invalid_indices
            }
        save_file = self.pose_traj_path + 'unrefined.pkl'
        save_pkl(save_file, save_dict)

        trimesh.points.PointCloud(vertices=self.cam_trans, colors=colors).export(
            self.traj_path + '/final_unrefine.ply'.format(big_iter))

        # ====================================
        #    At the end save refined file
        # ====================================

        if self.refine:
            self.refine_vals()

            save_dict = {
                'poses': self.imu_pose_new,
                'transes': self.cam_trans,
                'invalid_indices': self.invalid_indices
            }

            save_file = self.pose_traj_path + 'refined.pkl'
            save_pkl(save_file, save_dict)

            trimesh.points.PointCloud(vertices=self.cam_trans, colors=colors).export(
                self.traj_path + '/final_refine.ply'.format(big_iter))

    def run_all(self):
        all_file_names = os.listdir(CAM_PATH)
        all_file_names.sort()
        for file in all_file_names:
            print("============================")
            print(file)
            try:
                self.run_one(file)
            except:
                print('not found', file)
                continue


if __name__ == "__main__":
    def str2bool(inp):
        return inp.lower in ['true']

    parser = argparse.ArgumentParser()


    parser.add_argument("--file_name", default="SUB7_MPI_BIB_OG_long", help = "sequenc ename")

    parser.add_argument("--gamma", default=100, help="the jump to be used for computing ")

    parser.add_argument("--velocity_threshold", type=float, default=3.0, help="max velocity allowed in metres/frame")

    parser.add_argument("--imu_filt_fact_strt", type=float, default=10.0, help="imu filter factor start, only active when"
                                                                               " - i.e what comparitive factor defines a valid velocity ")

    parser.add_argument("--replace_cam_z", type=str2bool, default=True, help="replace z value with imu z values")

    parser.add_argument("--cor_type", type=str, default="xy", help="xy | xyz")

    parser.add_argument("--refine", type=str2bool, default=False, help="scale the vectors or not")

    args = parser.parse_args()

    solver = Solver(
        velocity_threshold = args.velocity_threshold,
        replace_cam_z = args.replace_cam_z,
        cor_type = args.cor_type,
        imu_filt_fact_strt = args.imu_filt_fact_strt,
        refine = args.refine
    )

    if args.file_name == "all":
        solver.run_all()
    else:
        solver.run_one(args.file_name)
