import numpy as np
import os
import pickle as pkl
import json
import torch
import trimesh

import torchgeometry as tgm
from torch.nn import functional as F

from scipy.spatial.transform import Rotation as R

VIDEO_PATH = "Location of video files"
MVNX_PATH = "Location of mvnx files"

TXT_PATH =  "Location of txt files"

CONTACT_VERT_PATH = "Location of contact vertices files"

BETA_PATH = "Location of beta files "

SCENE_PATH = "Location of scene files"

IN_CAM_PATH = 'Location of Camera localization results'

PEOPLE = ['SUB1', 'SUB2', 'SUB3', 'SUB4', 'SUB5', 'SUB6', 'SUB7']

feet_dict = ['right_toe', 'left_toe', 'left_heel', 'right_heel']

AUDIO_PATH = "Location to store audio files"


CONTACT_PATH = "Location to store audio files"
SIT_PATH = "Location to store sit files"
INIT_PATH = "Location to store initialization files"

IMU_PATH = "location to store imu files"
CAM_PATH = "location to store camera files"

FILTER_PATH = "location to store filered files"


MODEL_PATH = 'location of SMPL model files'

def person_from_file(file_name):
    """
    Get the person involved in the filename
    Args:
        file_name:
    Returns:
    """

    if "double" in file_name.lower():

        check_str = file_name.lower()[-4:]
        if check_str == "SUB2":
            name = "SUB2"
        else:
            name = "SUB5"


    else:
        for person in PEOPLE:
            if person in file_name.lower():
                name = person

    return name

def scene_from_file(file_name):
    """
    Get the person involved in the filename
    Args:
        file_name:
    Returns:
    """
    file_name = file_name.lower()
    if "double" in file_name:
        scene = "MPI_EG"

    elif 'SUB2' in file_name:
        scene = "MPI_EG"
    elif 'SUB5' in file_name:
        scene = "MPI_GEB_AB"
    elif 'SUB4' in file_name:
        scene = "MPI_Etage6"

    elif 'SUB6' in file_name:
        scene = "MPI_BIBLIO_UG"

    elif 'SUB7' in file_name:
        scene = "MPI_BIBLIO_OG"

    elif 'SUB1' in file_name:
        if 'exercises' in file_name:
            scene = "MPI_BIB_AB_SEND"
        else:
            scene = "MPI_BIBLIO_EG"

    elif 'SUB3' in file_name:
        if 'long' in file_name:
            scene = "MPI_EG"
        else:
            scene = "MPI_KINO"
    print(scene)
    mesh_file = SCENE_PATH + scene + '/output/mesh/1M.ply'
    return mesh_file

def scenename_from_file(file_name):
    """
        Get the person involved in the filename
        Args:
            file_name:
        Returns:
        """
    file_name = file_name.lower()
    if "double" in file_name:
        scene = "MPI_EG"

    elif 'SUB2' in file_name:
        scene = "MPI_EG"
    elif 'SUB5' in file_name:
        scene = "MPI_GEB_AB"
    elif 'SUB4' in file_name:
        scene = "MPI_Etage6"

    elif 'SUB6' in file_name:
        scene = "MPI_BIBLIO_UG"

    elif 'SUB7' in file_name:
        scene = "MPI_BIBLIO_OG"

    elif 'SUB1' in file_name:
        if 'exercises' in file_name:
            scene = "MPI_BIB_AB_SEND"
        else:
            scene = "MPI_BIBLIO_EG"

    elif 'SUB3' in file_name:
        if 'long' in file_name:
            scene = "MPI_EG"
        else:
            scene = "MPI_KINO"

    return scene

def clean_scene_from_file(file_name):
    scene = scenename_from_file(file_name)
    print(scene)
    mesh_file = SCENE_PATH + scene + '/10M_clean.ply'
    return mesh_file

def normals_from_file(file_name, norm_thresh):
    scene = scenename_from_file(file_name)

    normal_file = SCENE_PATH + scene + '/10M_flat_vert{}.npy'.format(norm_thresh)
    return normal_file

def data_from_file(file_name):
    """
    Args:
        file_name:
    Returns:
    """
    person_name = person_from_file(file_name)
    print('person name: ', person_name)
    beta_file = BETA_PATH + person_name + '.json'
    data = read_json_file(beta_file)
    return data

def data_from_person(person):
    beta_file = BETA_PATH + person + '.json'
    data = read_json_file(beta_file)
    return data


def beta_from_file(file_name):
    """
    Gets the betas from the file_name
    Args:
        file_name:
    Returns:
    """
    data = data_from_file(file_name)
    return data['betas']

def gender_from_file(file_name):
    """
    Args:
        file_name:
    Returns:
    """
    data = data_from_file(file_name)
    return data['gender']

def imu_data(file_name, type):
    """
    Args:
        file_name:
    Returns:
    """
    imu_file = IMU_PATH + file_name + '/{}.pkl'.format(type)
    imu_data = read_pkl_file(imu_file)
    cam_start, imu_start = get_init_data(file_name)
    imu_trans = imu_data['transes'][imu_start:,:]
    imu_pose = imu_data['poses'][imu_start:,:]
    return imu_trans, imu_pose


def get_cam_file_path(filename):
    """
    For a given filename returns the path of the camera localization results path
    Args:
        filename:
    Returns: The json file
    """
    all_jsons = get_allpaths(IN_CAM_PATH, ".json")
    for name in all_jsons:
        if (filename + ".json" in name):
            return name


def convert_json_to_pickl( data):
    """
    Converts a json array to a pickle array
    Args:
        data: input json array
    Returns:
    """
    pose_arr = []
    trans_arr = []
    nones = []
    names = sorted(data.keys(), key=lambda x: int(x))
    path = [data[n] for n in names]

    for ind in range(len(path)):
        frame = path[ind]
        if frame != None:
            trans = np.array(frame['position'])
            trans_arr.append(trans)

            pose = np.roll(frame['quaternion'], -1)
            pose_arr.append(pose)
        else:
            trans_arr.append([0, 0, 0])
            pose_arr.append([0, 0, 0, 0])
            nones.append(ind)

    pose_arr = np.array(pose_arr)
    trans_arr = np.array(trans_arr)

    cam_dict = {
        'pose': pose_arr,
        'trans': trans_arr,
    }
    return cam_dict

def cam_data(file_name):
    """
    Args:
        file_name:
    Returns:
    """
    cam_file = get_cam_file_path(file_name)
    cam_data = read_json_file(cam_file)
    cam_data = convert_json_to_pickl(cam_data)

    cam_pose, cam_trans = cam_data['pose'], cam_data['trans']
    cam_start, imu_start = get_init_data(file_name)

    cam_pose = cam_pose[cam_start:, :]
    cam_trans = cam_trans[cam_start:, :]

    return cam_trans, cam_pose

def sync_data(file_name, type):
    """
    Takes a file_name and type of IMU file and returns the synchronized arrays
    cam_trans and cam_pose are unadultered arrays - no filtering or interpolation.
    This function converts vladimirs json files to pkl type arrays and then cuts the start and end
    Args:
        file_name: name of the file
        type: 'bake|vel|basic'
    Returns:

    """
    imu_trans, imu_pose = imu_data(file_name, type)
    cam_trans, cam_pose = cam_data(file_name)

    end_index = min(cam_trans.shape[0], imu_trans.shape[0])

    imu_trans = imu_trans[0:end_index, :]
    imu_pose = imu_pose[0:end_index, :]

    cam_trans = cam_trans[0:end_index, :]
    cam_pose = cam_pose[0:end_index, :]

    return cam_trans, cam_pose, imu_trans, imu_pose

def save_pkl(file_name , save_dict):
    """
    Saves a dictionary in the file name
    Args:
        file_name: File name with .pkl extension
        save_dict:
    Returns:
    """
    with open(file_name, 'wb') as f:
        pkl.dump(save_dict, f)

def get_allpaths(dir_name, extension):
    """
    Args:
        dir_name:
        extension:

    Returns: list of all files with the extension in the directory

    """
    return_list = []
    for dirpath, dirnames, filenames in os.walk(dir_name):
        for filename in [f for f in filenames if f.endswith(extension)]:
            return_list.append(os.path.join(dirpath, filename))
    return return_list

def read_pkl_file(file_name):
    """
    Reads a pickle file
    Args:
        file_name:

    Returns:

    """
    with open(file_name, 'rb') as f:
       data = pkl.load(f)
    return data

def read_json_file(file_name):
    """
    Reads a json file
    Args:
        file_name:

    Returns:

    """
    with open(file_name) as f:
        data = json.load(f)
    return data

def save_json_file(file_name, save_dict):
    """
    Saves a dictionary into json file
    Args:
        file_name:
        save_dict:

    Returns:

    """
    with open(file_name, 'w') as fp:
        json.dump(save_dict, fp, indent=4)

def get_init_data(file_name):
    """
    Args:
        file_name:
    Returns: camera starting position, imu starting position
    """
    init_file = INIT_PATH + file_name + '.pkl'
    init_data = read_pkl_file(init_file)
    return init_data['cam_start'], init_data['imu_start']


def rot_mat(veca, vecb):
    """
    Rotates veca to vecb
    Args:
        veca: vector which is to be rotated
        vecb: vector which is the intended direction

    Returns:
    Matrix that rotates vector a to vector b
    """

    veca_norm = veca / np.linalg.norm(veca, ord=2)
    vecb_norm = vecb / np.linalg.norm(vecb, ord=2)

    cross = np.cross(veca_norm, vecb_norm)
    sin = np.linalg.norm(cross, ord=2)
    cos = np.dot(veca_norm, vecb_norm)
    matrix = [[0, -cross[2], cross[1]], [cross[2], 0, - cross[0]], [-cross[1], cross[0], 0]]
    I = np.identity(3)
    final = I + matrix + (1 / (1 + cos)) * np.matmul(matrix, matrix)
    return final

def compute_velocity(inplist):
    """
    Computes forward difference of a positional array
    Args:
        inplist: the input data array
    Returns:
        the velocity computed using forward differences
    """
    # the next position minus the current position
    vel = inplist[1:, : ] - inplist[:-1, :]
    return np.copy(vel)

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat


def batch_rodrigues(axisang):
    """
    Converts a batch of
    Args:
        axisang: axisang N x 3
    Returns:

    """
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    # rot_mat = rot_mat.view(rot_mat.shape[0], 9)
    return rot_mat

def joint_orient_error(pred_mat, gt_mat):
    """ffind
    Find the orientation error between the predicted and GT matrices
    Args:
        pred_mat: Batch x 3 x 3
        gt_mat: Batch x 3 x 3
    Returns:

    """
    r1 = pred_mat
    r2t = torch.transpose(gt_mat, 2, 1)
    # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
    r = torch.bmm(r1, r2t)

    # Convert rotation matrix to axis angle representation and find the angle
    pad_tensor = F.pad(r, [0, 1])
    residual = tgm.rotation_matrix_to_angle_axis(pad_tensor)
    norm_res = torch.norm(residual, p=2, dim=1)

    return torch.mean(norm_res)

def get_all_file_names():
    """
    Returns all the file names of data points
    Returns:
    """
    all_file_names = os.listdir(CAM_PATH)
    all_file_names.sort()
    return all_file_names

def inttraj_as_ply(vertices, invalid_indices, name):
    """
    Args:
        vertices:
        invalid_indices:
        name:
    Returns:
    """
    colors = np.zeros(vertices.shape)
    colors[invalid_indices, 0] = 255
    trimesh.points.PointCloud(vertices=vertices, colors=colors).export(name + '.ply')

def multiply_pose(pose, TRANS):
    """
    Args:
        pose: the
        TRANS: 3 x 3 matrix
    Returns:
    """
    glob_or = pose[:3]
    rot_matrix = R.from_rotvec(glob_or).as_matrix()
    new_glob = np.matmul(TRANS, rot_matrix)
    new_glob_ax = R.from_matrix(new_glob).as_rotvec()
    pose[:3] = new_glob_ax
    return pose

def traj_array_as_ply(vertices, colors, name):
    """
    Args:
        vertices:
        colors:
        name:
    Returns:
    """
    trimesh.points.PointCloud(vertices=vertices, colors=colors).export(name+'.ply')



def get_av_velocity(data_transes_cam, index, gamma, type):
    """
    Args:
        data_transes_cam:
        index:
    Returns:
    """
    total_vel = []
    gamma = np.int(gamma / 2)

    end_index = data_transes_cam.shape[0]

    begin = max(0, index - gamma)
    end = min(end_index, index + gamma + 1)
    av_vel = np.array([0, 0, 0])

    for i in range(begin, index):
        # if not (i in self.invalid_indices):
        vel = (data_transes_cam[index] - data_transes_cam[i]) / (index - i)
        total_vel.append(vel)

    for i in range(index + 1, end):
        # if not (i in self.invalid_indices):
        vel = (data_transes_cam[i] - data_transes_cam[index]) / (i - index)
        total_vel.append(vel)

    total_vel = np.array(total_vel)

    if type == "mean":
        if len(total_vel) != 0:
            av_vel = np.mean(total_vel, axis=0)
        else:
            av_vel = np.array([0, 0, 0])
    elif type == "median":
        if len(total_vel) != 0:
            av_vel = np.median(total_vel, axis=0)
        else:
            av_vel = np.array([0, 0, 0])

    return av_vel

def get_foot_contacts(file_name):
    cam_start, imu_start = get_init_data(file_name)
    contact_path = CONTACT_PATH + file_name + '.pkl'
    data = read_pkl_file(contact_path)

    for str in feet_dict:
        data[str] = np.array(data[str][imu_start:])
    return data

def get_back_contacts(file_name):
    back_contact_pth = SIT_PATH + file_name + '.json'
    data = read_json_file(back_contact_pth)

    return np.array(data['sit_contacts'])

def get_all_contacts(file_name):
    data = get_foot_contacts(file_name)
    data['back'] = get_back_contacts(file_name)

    return data



def get_verts_dict():
    verts_file = CONTACT_VERT_PATH + 'all_new.json'
    data = read_json_file(verts_file)

    ret_dict ={}

    for str in feet_dict:
        ret_dict[str] = np.array(data[str]['verts'])

    ret_dict['back'] = np.array(data['back_new']['verts'])

    return ret_dict

def get_opt_init_data(file_name, imu_start, vel_thresh):


    file_pth = FILTER_PATH + 'pose_traj_files_new/' + file_name +\
               '/filt_strt-{}_vel_filt_th-{}_repcamz-True_cortype-xy/'.format(imu_start, vel_thresh) \
               + 'unrefined.pkl'

    data = read_pkl_file(file_pth)
    return data



def find_verts(verts, minx, maxx, miny, maxy, minz, maxz):

    verts_locs = np.where(verts[:, 0] <= maxx)
    verts_loc2 = np.where(verts[:, 0] >= minx)

    verts_locs3 = np.where(verts[:,1]<=maxy)
    verts_locs4 = np.where(verts[:,1]>= miny)

    verts_locs5 = np.where(verts[:,2]<= maxz)
    verts_locs6 = np.where(verts[:,2]>= minz)

    ret_verts = list(set(verts_locs[0].tolist()) & (set(verts_loc2[0].tolist())) & (set(verts_locs3[0].tolist())) &
                     set(verts_locs4[0].tolist()) & set(verts_locs5[0].tolist()) & (set(verts_locs6[0].tolist())))

    return ret_verts



def find_verts_z(verts, minz, maxz ):

    verts_locs = np.where(verts[:,2]<= maxz)
    verts_locs2 = np.where(verts[:,2]>= minz)

    ret_verts = list(set(verts_locs[0].tolist()) & (set(verts_locs2[0].tolist())))
    return ret_verts


if __name__ == "__main__":
    import ipdb
    ipdb.set_trace()
