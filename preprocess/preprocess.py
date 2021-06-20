# conda activate xsens
from moviepy.editor import *
import argparse

from scipy.signal import find_peaks
import librosa
import os.path
import pickle as pkl
import json

import sys
sys.path.append("../")

from xsens import mvnx

from global_vars import *

from transform_imu_file import *
from unity_functions import process_smpl_pose, process_smpl_trans
from velocity_filter import get_cam


def get_allpaths(dir_name, extension):
    return_list = []
    for dirpath, dirnames, filenames in os.walk(dir_name):
        for filename in [f for f in filenames if f.endswith(extension)]:
            return_list.append(os.path.join(dirpath, filename))
    return return_list

def get_contacts(contacts):
    LeftFoot_Toe = []
    RightFoot_Toe = []

    LeftFoot_Heel = []
    RightFoot_Heel = []

    for i in range(len(contacts)):
        LeftFoot_Heel.append(contacts[i]['LeftFoot_Heel'][0])
        RightFoot_Heel.append(contacts[i]['RightFoot_Heel'][0])

        LeftFoot_Toe.append(contacts[i]['LeftFoot_Toe'][0])
        RightFoot_Toe.append(contacts[i]['RightFoot_Toe'][0])


    new_dict = {
        'left_heel':LeftFoot_Heel,
        'left_toe': LeftFoot_Toe,
        'right_toe':RightFoot_Toe,
        'right_heel':RightFoot_Heel
    }

    return new_dict

def get_cam_start(video_name, audio_name):
    if not os.path.exists(audio_name):
        clip = AudioFileClip(video_name)
        clip.write_audiofile(audio_name)

    audio, sr = librosa.load(audio_name, sr = None)
    audio_short = audio[:50*sr]
    peaks, dict_ret = find_peaks(audio_short, height = 0)

    num = np.where(audio_short == np.max(dict_ret['peak_heights']))
    cam_start = float(num[0]*30)/sr
    return int(cam_start)

def get_imu_start(acc):
    rt_acc = []
    lt_acc = []
    for i in range(len(acc)):
        rt_acc.append(acc[i]['right_hand'])
        lt_acc.append(acc[i]['left_hand'])

    rt_acc = np.linalg.norm(np.array(rt_acc[:1000]), axis =1)
    lt_acc = np.linalg.norm(np.array(lt_acc[:1000]), axis = 1)

    peaks_rt, dict_ret_rt = find_peaks(rt_acc, height = 0)
    peaks_lt, dict_ret_lt = find_peaks(lt_acc, height = 0)

    rt_max = np.where(rt_acc == np.max(dict_ret_rt['peak_heights']))
    lt_max = np.where(lt_acc == np.max(dict_ret_lt['peak_heights']))

    print('Right start: ', rt_max[0][0])
    print('Left Start: ', lt_max[0][0])
    return int(lt_max[0][0])



def get_video_name(filename):
    all_videos = get_allpaths(VIDEO_PATH, ".MP4")
    for name in all_videos:
        if (filename + ".MP4" in name) or (filename + ".01.MP4" in name):
            return name

def get_cam_file_name(filename):
    all_jsons = get_allpaths(IN_CAM_PATH, ".json")
    for name in all_jsons:
        if (filename + ".json" in name) :
            return name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, default="SUB4_MPI_Etage6_eval")
    args = parser.parse_args()
    print('\nPROCESSING')
    print(args.file_name)
    ### ============DEFINE FILE NAMES ===========
    ## INPUT FILES
    pose_txt_file = TXT_PATH + args.file_name + '_pose.txt'
    trans_txt_file = TXT_PATH + args.file_name + '_trans.txt'



    in_cam_file = get_cam_file_name(args.file_name)
    mvnx_name = MVNX_PATH + args.file_name + ".mvnx"
    video_name = get_video_name(args.file_name)
    print(video_name)
    audio_name = AUDIO_PATH + args.file_name + ".wav"

    ### OUPUT FILES
    contact_name = CONTACT_PATH + args.file_name + ".pkl"
    init_name = INIT_PATH + args.file_name + ".pkl"

    imu_dir = IMU_PATH + args.file_name
    if not os.path.exists(imu_dir):
        os.makedirs(imu_dir)
    imu_name = imu_dir + '/basic.pkl'
    imu_name_bake = imu_dir + '/bake.pkl'

    # CAMERA OUTPUT FILES
    cam_name = CAM_PATH + args.file_name
    if not os.path.exists(cam_name):
        os.makedirs(cam_name)
    cam_name_filt = cam_name + '/filtered.pkl'
    cam_name_unfilt = cam_name + '/unfiltered.pkl'



    ### ============CONTACT AND INITIALIZATION INFORMATION===============

    cam_start = get_cam_start(video_name, audio_name)

    data = mvnx.MVNX(mvnx_name)
    acc = data.get_info('acceleration')
    imu_start = get_imu_start(acc)

    init_dict = {
        'imu_start' : imu_start,
        'cam_start' : cam_start
    }

    print('IMU starts at : {}'.format(imu_start))
    print('Camera starts at : {}'.format(cam_start))

    with open(init_name, 'wb') as fp:
        pkl.dump(init_dict, fp)

    ### SAVE
    contacts = data.get_info('footContacts')
    contact_dict = get_contacts(contacts)

    with open(contact_name, 'wb') as fp:
        pkl.dump(contact_dict, fp)

    ### PROCESS TXT FILES

    poses = process_smpl_pose(pose_txt_file)
    trans = process_smpl_trans(trans_txt_file)

    imu_dict = {
        'pose': poses,
        'trans': trans
    }
    print('Number of poses and transes in txt file : {} {} '.format(len(poses), len(trans)))
    with open(imu_name, 'wb') as f:
        pkl.dump(imu_dict, f)


    ### PROCESS CAM JSON FILES
    with open(in_cam_file) as json_file:
        cam_data = json.load(json_file)

    cam_fil_dict = get_cam(cam_data, threshold=0.1)
    with open(cam_name_filt, 'wb') as f:
        pkl.dump(cam_fil_dict, f)

    cam_unfil_dict = get_cam(cam_data, threshold=0)
    with open(cam_name_unfilt, 'wb') as f:
        pkl.dump(cam_unfil_dict, f)