import torch
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as cham
import numpy as np
from scipy.spatial.transform import Rotation as R

import torchgeometry as tgm
from torch.nn import functional as F
import cv2



def joint_angle_error(pred_mat, gt_mat, rot_max):
    """
    Compute the geodesic distance between the two input matrices.
    :param pred_mat: predicted rotation matrices. Shape: ( Seq, 3, 3)
    :param gt_mat: ground truth rotation matrices. Shape: ( Seq, 3, 3)
    :return: Mean geodesic distance between input matrices.
    """


    r1 = pred_mat


    r2t = torch.transpose(gt_mat, 2, 1)

    r = torch.bmm(r1, r2t)

    pad_tensor = F.pad(r, [0, 1])
    residual = tgm.rotation_matrix_to_angle_axis(pad_tensor).contiguous()
    norm_res = torch.norm(residual, p = 2, dim = 1)
    norm_res = norm_res.unsqueeze(1)
    if rot_max:
        norm_res = torch.nn.functional.relu(norm_res - rot_max)

    return torch.mean(norm_res)

def str2bool(inp):
    return inp.lower() in ["true"]


def trans_smooth(trans_params, trans_max):
    diffs = trans_params[:-1,:] - trans_params[1:, :]
    # diffs_new = diffs[:,:2]
    # import ipdb
    # ipdb.set_trace()
    if trans_max:
        diffs_new2 = torch.nn.functional.relu(torch.norm(diffs, dim = 1, p = 2) - trans_max)
        return torch.mean(diffs_new2)
    else:
        return torch.mean(torch.norm(diffs, dim =1, p = 2))




def matrot2aa(pose_matrot):
    '''
    :param pose_matrot: Nx1xnum_jointsx9
    :return: Nx1xnum_jointsx3
    '''
    batch_size = pose_matrot.size(0)
    homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])
    pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 1, -1, 3).contiguous()
    return pose

def aa2matrot(pose):
    '''
    :param Nx1xnum_jointsx3
    :return: pose_matrot: Nx1xnum_jointsx9
    '''
    batch_size = pose.size(0)
    pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, 1, -1, 9)
    return pose_body_matrot

def head_constraint(offset, pred_orientations, gt_orientations):
    '''
    :param offset:
    :param pred_orientations:
    :param gt_orientations:
    :param valid_indices:
    :return:
    '''
    PY_TRANS5 = torch.tensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]]).type(torch.FloatTensor).cuda()

    batch_pytrans = PY_TRANS5.repeat((pred_orientations.size(0), 1, 1)).type(torch.FloatTensor).cuda()
    batch_offset = offset.repeat((pred_orientations.size(0), 1, 1)).type(torch.FloatTensor).cuda()


    r1 = torch.bmm(pred_orientations, batch_offset)

    r = R.from_quat(gt_orientations)
    gt_matrs = r.as_matrix()
    r2 = torch.tensor(gt_matrs)
    r2 = r2.type(torch.FloatTensor)
    r2 = r2.cuda()
    r2 = torch.bmm(r2, batch_pytrans)

    error = joint_angle_error(r1, r2, 0)
    return error

def smooth_glob_ors(glob_orientations, valid_indices, rot_max):
    error = joint_angle_error(glob_orientations[:-1,:,:], glob_orientations[1:, :, :], valid_indices, rot_max)
    return error


def define_variables(data_poses_imu_window, data_transes_cam_window):
    # pose_params = torch.zeros(data_poses_imu_window.shape[0], 1, data_poses_imu_window.shape[1])
    pose_params = torch.from_numpy(data_poses_imu_window)
    pose_glob = pose_params[:,:3]
    pose_head = pose_params[:, 45:48]
    pose_between_glob_and_head = pose_params[:,3:45]
    pose_after_head = pose_params[:, 48:]

    pose_glob.requires_grad = False
    pose_glob = pose_glob.type(torch.FloatTensor).cuda()

    pose_head.requires_grad = False
    pose_head = pose_head.type(torch.FloatTensor).cuda()

    pose_between_glob_and_head.requires_grad = False
    pose_between_glob_and_head = pose_between_glob_and_head.type(torch.FloatTensor).cuda()

    pose_after_head.requires_grad = False
    pose_after_head = pose_after_head.type(torch.FloatTensor).cuda()

    trans_params = torch.from_numpy(data_transes_cam_window)
    trans_params.requires_grad = False
    trans_params = trans_params.type(torch.FloatTensor).cuda()

    return trans_params, pose_glob, pose_between_glob_and_head, pose_head, pose_after_head


def v2vloss(batch_vertices):
    """
    Args:
        batch_vertices:

    Returns:

    """
    vertex_diffs = batch_vertices[:-1] - batch_vertices[1:]
    # Frames-1 x 6890 x 3
    norm_verts = torch.norm(vertex_diffs, p = 2, dim = 2)
    # import ipdb
    # ipdb.set_trace()
    norm_per_frame = torch.mean(norm_verts, dim =1)
    norm_all = torch.mean(norm_per_frame, dim = 0)

    return norm_all

def trans_imu_smooth(trans_params, imu_trans):

    trans_diffs = torch.norm(trans_params[:-1,:] - trans_params[1:, :], dim =1)
    print(trans_diffs.shape)

    imu_diffs = torch.norm(torch.tensor(imu_trans[:-1,:] - imu_trans[1:, :], dtype = torch.float, device = 'cuda'), dim =1)
    print(imu_diffs.shape)

    diffs = trans_diffs - imu_diffs


    diffs_new2 = torch.nn.functional.relu(diffs)
    return torch.mean(diffs_new2)

