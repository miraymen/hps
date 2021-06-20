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
from psbody.mesh import Mesh
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

        parser.add_argument("--person", type=str, default="SUB1", help = "SUB1-7")

        parser.add_argument("--folder", type=str, default='false')

        self.args = parser.parse_args()

        ### OUPUT FILES
        self.init_file = INIT_PATH + self.args.file_name + ".pkl"

        self.imu_file = IMU_PATH + self.args.file_name + '/basic.pkl'
        self.cam_file = CAM_PATH + self.args.file_name + '/' + self.args.cam_file_type + '.pkl'

        self.beta_file = BETA_PATH + self.args.person + '.json'

        self.feet_dict = ['right_toe', 'left_toe', 'left_heel', 'right_heel']

        self.FLIP_VAR = True
        print('PROCESSING')
        print(self.args.file_name)

    def bake(self, imu_dict, TRANS):
        imu_poses = imu_dict['pose']
        imu_transes = imu_dict['trans']
        imu_transes_save = np.copy(imu_transes)
        for i in range(imu_poses.shape[0]):
            glob_or = imu_poses[i, :3]
            rot_matrix = R.from_rotvec(glob_or).as_matrix()
            new_glob = np.matmul(TRANS, rot_matrix)
            new_glob_ax = R.from_matrix(new_glob).as_rotvec()
            imu_poses[i, :3] = new_glob_ax

            trans = np.reshape(imu_transes[i], (3, 1))
            # trans[0,0] = -trans[0,0]
            imu_transes_save[i] = np.reshape(np.matmul(TRANS, trans), (3,))
            # import ipdb
            # ipdb.set_trace()
        ret_dict = {'pose': imu_poses,
                    'trans': imu_transes_save
                    }

        return ret_dict

    def flip(self, verts):

        new_verts_store = verts.clone()
        verts[:, :, 1] = -new_verts_store[:, :, 2]
        verts[:, :, 2] = new_verts_store[:, :, 1]
        return verts

    def read_pkl_file(self, file_name):
        with open(file_name, 'rb') as f:
            data = pkl.load(f)
        return data

    def read_json_file(self, file_name):
        with open(file_name, 'rb') as f:
            data = json.load(f)
        return data

    def load_variables(self):

        # self.scene = trimesh.load(self.mesh_file)

        # self.feet_verts = self.read_pkl_file(FOOT_FILE)

        self.imu_start = self.read_pkl_file(self.init_file)['imu_start']
        self.cam_start = self.read_pkl_file(self.init_file)['cam_start']

        self.diff = min(30, self.imu_start, self.cam_start)
        print(self.diff)
        self.imu_start = self.imu_start
        self.cam_start = self.cam_start

        self.betas = self.read_json_file(self.beta_file)['betas']
        self.betas_torch = torch.tensor(self.betas, requires_grad = False).unsqueeze(0).repeat(
                self.args.window_frames, 1)
        self.gender = self.read_json_file(self.beta_file)['gender']

        self.data_poses_imu = self.read_pkl_file(self.imu_file)['pose'][self.imu_start:, :]

        self.data_poses_cam = self.read_pkl_file(self.cam_file)['pose'][self.cam_start:, :]
        self.data_transes_cam = self.read_pkl_file(self.cam_file)['trans'][self.cam_start:, :]


        self.smpl_layer = SMPL_Layer(
            center_idx=0,
            gender=self.gender,
            model_root='/BS/aymen/work/20_09_29-Optimization/my_code2/smplpytorch/smplpytorch/native/models')

        self.faces = self.smpl_layer.th_faces
        self.smpl_layer.cuda()

    def joint_orient_error(self, pred_mat, gt_mat):
        r1 = pred_mat

        r2t = torch.transpose(gt_mat, 2, 1)
        # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
        r = torch.bmm(r1, r2t)

        # angles = []
        # Convert rotation matrix to axis angle representation and find the angle
        # import ipdb
        # ipdb.set_trace()
        pad_tensor = F.pad(r, [0, 1])
        residual = tgm.rotation_matrix_to_angle_axis(pad_tensor).contiguous()
        norm_res = torch.norm(residual, p=2, dim=1)

        return torch.mean(norm_res)

    def smpl_pass(self, pose_params, trans_params, rot_mat):
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

            # multiply head orient with rotation matrix
            # impose constraint on this orientation
            # loss on vertex position
            rot_loss = self.joint_orient_error(head_or, pose_target_cam_mat)
            vertex_loss = torch.norm(smpl_verts[:, 3163, :] - trans_target_cam, dim = 1, p = 2)
            tot_loss = self.args.rot_wt * rot_loss + self.args.vert_wt *vertex_loss
            tot_loss.backward()
            optimizer.step()
        print('tot loss : {} rot loss : {}'.format(tot_loss.item(), rot_loss.item()))

        opt_for = torch.cat((dum_var, opt_var), dim = 0).unsqueeze(0)
        rot_trans = batch_rodrigues(opt_for).squeeze(0)

        rot_trans = rot_trans.detach().cpu().numpy()
        print(rot_trans)
        final_trans = np.matmul(rot_trans, (np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])))
        print(final_trans)
        self.rot_matrix = final_trans

    def save(self):
        from global_vars import IMU_PATH
        imu_dir = IMU_PATH + self.args.file_name

        with open(self.imu_file, 'rb') as f:
            imu_dict = pkl.load(f)

        imu_name_bake = imu_dir + '/bake.pkl'

        #comes from transform_imu_file
        bake_dict = self.bake(imu_dict, self.rot_matrix)


        with open(self.cam_file, 'rb') as f:
            cam_dict = pkl.load(f)

        cam_start_trans = cam_dict['trans'][self.cam_start]
        imu_start_trans = bake_dict['trans'][self.imu_start]

        move_vect = cam_start_trans - imu_start_trans
        move_vect[2] = 0
        bake_dict['trans'] = bake_dict['trans'] + (move_vect)
        with open(imu_name_bake, 'wb') as fp:
            pkl.dump(bake_dict, fp)



if __name__ == "__main__":
    init_opt = Init_Optimizer()
    init_opt.run()
    init_opt.save()