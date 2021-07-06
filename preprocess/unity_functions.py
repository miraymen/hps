"""
    Utility functions for transforming text files to pkl files
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
import math


def get_string(number):
    return "%05d" % number

def flip(new_verts):
    """
    Flip vertices along
    Args
        :param new_verts:
    Return:
    """
    new_verts_store = new_verts.copy()
    new_verts.flags.writeable = True

    new_verts[:, 0] = -new_verts[:, 0]
    new_verts[:, 1] = -new_verts_store[:, 2]
    new_verts[:, 2] = new_verts_store[:, 1]

    return new_verts


def rotation_mat(thetas):
    """
    conversion Euler angles of form "ZXY" used in Unity -> rotvec used in SMPL
    """
    x_rot = np.array([[1, 0, 0],
                      [0, math.cos(thetas[0]), -math.sin(thetas[0])],
                      [0, math.sin(thetas[0]), math.cos(thetas[0])]])

    y_rot = np.array([[math.cos(thetas[1]), 0, -math.sin(thetas[1])],
                      [0, 1, 0],
                      [math.sin(thetas[1]), 0, math.cos(thetas[1])]])

    z_rot = np.array([[math.cos(thetas[2]), math.sin(thetas[2]), 0],
                      [-math.sin(thetas[2]), math.cos(thetas[2]), 0],
                      [0, 0, 1]])

    return R.from_matrix(np.matmul(y_rot, np.matmul(x_rot, z_rot))).as_rotvec()


def process_smpl_trans(trans_path):
    """
    code for processing smpl pose extracted from unity
    """
    with open(trans_path, 'rb') as file:
        lines = file.readlines()
    file.close()
    for i in range(len(lines)):
        lines[i] = [float(x) for x in lines[i].split()]
    lines = np.array(lines[:-1])
    lines =  lines.reshape(-1, 3)

    # add this
    lines[:, 0] = -lines[:, 0]
    return lines

def process_smpl_pose(pose_path):
    """
    code for processing smpl pose extracted from unity
    """
    reorder_indices = [0, 1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 17,
                       12, 19, 18, 13, 20, 14, 21, 15, 22, 16, 23]
    with open(pose_path, 'rb') as file:
        lines = file.readlines()
    file.close()
    for i in range(len(lines)):
        lines[i] = [float(x) for x in lines[i].split()]
    lines = np.array(lines[:-1])
    lines = np.delete(lines, [3,4,5], 1)

    lines = lines.reshape(-1, 24, 3)
    lines = lines[:, reorder_indices]
    lines = np.radians(lines)

    lines_new = lines.reshape(-1, 3)
    rotvecs = np.zeros((lines_new.shape[0], 3))

    for i in range(len(lines_new)):
        rotvecs[i] = rotation_mat(lines_new[i])

    return rotvecs.reshape(-1, 72)


