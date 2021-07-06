'''
    INITIALIZATION GENERATES BAKE FILE
    Requires the basic.pkl file
    conda: py3d
'''

import pickle as pkl
import argparse
import json
import torch

from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import torchgeometry as tgm
from torch.nn import functional as F
import sys

sys.path.append('../')
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from global_vars import *


class Init_Optimizer():
    def __init__(self):
        self.get_args()
        self.load_variables()

    def get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--file_name", type = str, default = "SUB4_MPI_Etage6_working_standing")
        parser.add_argument("--window_frames", type=int, default=1, help="window of the frame to be used")
        parser.add_argument("--cam_file_type", type=str, default="filtered", help = "filtered|unfiltered")
        parser.add_argument("--rot_wt", type = float, default = 10, help = "rotation weight")
        parser.add_argument("--vert_wt", type = float, default = 0.5, help = "translation weight")
        parser.add_argument("--learn_rate", type = float, default = 0.03, help = "translation weight")
        parser.add_argument("--iterations", type = int, default = 300, help = "iterations")
        parser.add_argument("--person", type=str, default="SUB4", help = "SUB1-7")
        parser.add_argument("--folder", type=str, default='false')

        self.args = parser.parse_args()

        # OUPUT FILES
        self.init_file = INIT_PATH + self.args.file_name + ".json"
        self.imu_file = IMU_PATH + self.args.file_name + '/basic.pkl'
        self.imu_bake_file = IMU_PATH + self.args.file_name + '/bake.pkl'
        self.FLIP_VAR = True

        print('PROCESSING')
        print(self.args.file_name)

    def bake(self, imu_dict, TRANS):
        """
        Bakes the imu dictionary by multiplying each pose and distance by the TRANS matrix
        :param imu_dict: the input imu dictionary
        :param TRANS: the transformation matrix
        :return: The transformed dictionary
        """
        imu_poses = imu_dict['poses']
        imu_transes = imu_dict['transes']
        imu_transes_save = np.copy(imu_transes)
        for i in range(imu_poses.shape[0]):
            glob_or = imu_poses[i, :3]
            rot_matrix = R.from_rotvec(glob_or).as_matrix()
            new_glob = np.matmul(TRANS, rot_matrix)
            new_glob_ax = R.from_matrix(new_glob).as_rotvec()
            imu_poses[i, :3] = new_glob_ax

            trans = np.reshape(imu_transes[i], (3, 1))
            imu_transes_save[i] = np.reshape(np.matmul(TRANS, trans), (3,))

        ret_dict = {
                    'poses': imu_poses,
                    'transes': imu_transes_save
                    }

        return ret_dict

    def flip(self, verts):
        new_verts_store = verts.clone()
        verts[:, :, 1] = -new_verts_store[:, :, 2]
        verts[:, :, 2] = new_verts_store[:, :, 1]
        return verts


    def load_variables(self):
        self.betas = data_from_file(self.args.file_name)['betas']
        self.betas_torch = torch.tensor(self.betas, requires_grad = False).unsqueeze(0).repeat(
                self.args.window_frames, 1)
        self.gender = data_from_file(self.args.file_name)['gender']

        self.data_transes_cam, self.data_poses_cam, self.data_transes_imu, self.data_poses_imu = sync_data(self.args.file_name, 'basic')

        self.smpl_layer = SMPL_Layer(
            center_idx=0,
            gender=self.gender,
            model_root=MODEL_PATH)

        self.faces = self.smpl_layer.th_faces
        self.smpl_layer.cuda()

    def smpl_pass(self, pose_params, trans_params, rot_mat):
        """
        A forward pass through the SMPL model
        :param pose_params: The pose parameters
        :param trans_params: The translation
        :param rot_mat: The rotation matrix applied to the veritces and the head orientation
        :return: SMPL vertices and the head orientation
        """
        smpl_verts, _, orientations = self.smpl_layer(
            th_pose_axisang=pose_params.cuda(),
            th_betas=self.betas_torch.type(torch.FloatTensor).cuda()
        )
        if self.FLIP_VAR:
            new_verts = smpl_verts.permute(0,  2, 1)
            INIT_TRANS = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).unsqueeze(0).type(torch.FloatTensor).cuda()
            new_verts  = torch.bmm(INIT_TRANS, new_verts)
            new_verts = torch.bmm(rot_mat, new_verts)
            smpl_verts = new_verts.permute(0, 2, 1)
            head_or = torch.bmm(INIT_TRANS, orientations[:, 15,: ,: ])

        else:
            head_or = orientations[:, 15, :, :]

        head_or = torch.bmm(rot_mat, head_or)

        rep_trans = trans_params.unsqueeze(1).repeat(1, 6890, 1)
        smpl_verts = smpl_verts + rep_trans.cuda()

        return smpl_verts, head_or

    def define_window(self, index):
        return np.sort(list(range(index, index + self.args.window_frames))).astype(int)

    def run(self):
        window = self.define_window(0)
        pose_target_cam = self.data_poses_cam[window]
        trans_target_cam = torch.tensor(self.data_transes_cam[window], requires_grad = False).type(torch.FloatTensor).cuda()

        init_trans = trans_target_cam.clone()
        init_trans[:, 2] = init_trans[:, 2] - 0.8

        pose_target_cam_mat = torch.tensor(R.from_quat(pose_target_cam).as_matrix()).type(torch.FloatTensor).cuda()
        PY_TRANS5 = torch.tensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]]).type(torch.FloatTensor).cuda()

        pose_target_cam_mat = torch.bmm(pose_target_cam_mat, PY_TRANS5)
        pose_imu = torch.tensor(self.data_poses_imu[window], requires_grad = False).type(torch.FloatTensor).cuda()
        opt_var = torch.tensor([0.0 ], requires_grad = True)
        dum_var = torch.tensor([0.0, 0.0 ])
        optimizer = torch.optim.Adam([opt_var], self.args.learn_rate, betas=(0.9, 0.999))

        for it in range(self.args.iterations):
            optimizer.zero_grad()
            opt_for = torch.cat((dum_var, opt_var), dim = 0).unsqueeze(0)

            rot_trans = batch_rodrigues(opt_for).cuda()
            smpl_verts, head_or = self.smpl_pass(pose_imu, init_trans, rot_trans)

            rot_loss = joint_orient_error(head_or, pose_target_cam_mat)
            vertex_loss = torch.norm(smpl_verts[:, 3163, :] - trans_target_cam, dim = 1, p = 2)
            tot_loss = self.args.rot_wt * rot_loss + self.args.vert_wt *vertex_loss
            tot_loss.backward()
            optimizer.step()

        print('tot loss : {} rot loss : {}'.format(tot_loss.item(), rot_loss.item()))

        opt_for = torch.cat((dum_var, opt_var), dim = 0).unsqueeze(0)
        rot_trans = batch_rodrigues(opt_for).squeeze(0)
        rot_trans = rot_trans.detach().cpu().numpy()
        final_trans = np.matmul(rot_trans, (np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])))
        self.rot_matrix = final_trans

    def save(self):
        """
        Saves the imu bake dictionary in the imu_bake_file
        """
        imu_dict = read_pkl_file(self.imu_file)
        bake_dict = self.bake(imu_dict, self.rot_matrix)

        cam_start_trans = self.data_transes_cam[0]
        imu_start_trans = self.data_transes_imu[0]

        move_vect = cam_start_trans - imu_start_trans
        move_vect[2] = 0
        bake_dict['transes'] = bake_dict['transes'] + (move_vect)
        save_pkl(self.imu_bake_file, bake_dict)

if __name__ == "__main__":
    init_opt = Init_Optimizer()
    init_opt.run()
    init_opt.save()