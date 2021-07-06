"""
Gets all the frames which involve sitting.

Requires: bake files
"""
import sys
sys.path.append("../")
from global_vars import  *
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import torch
from tqdm import tqdm
import numpy as np

all_names = get_all_file_names()

for file in all_names:
    try:
        _, _ , imu_trans, imu_pose = sync_data(file, 'bake')
    except:
        continue

    print("Processing: {}".format(file))
    person = person_from_file(file)

    data = data_from_person(person)
    betas = data['betas']
    gender = data['gender']
    root_to_heel = data['root_to_heel']

    smpl = SMPL_Layer(gender=gender, model_root=MODEL_PATH)
    smpl = smpl.cuda()

    betas = torch.tensor(betas, dtype = torch.float, device = 'cuda')
    betas = betas.unsqueeze(0)

    left = 3468
    right = 6868

    valid_inds = []

    for i in tqdm(range(imu_pose.shape[0])):

        pose = imu_pose[i, :]
        pose = torch.tensor(pose,  dtype = torch.float, device = 'cuda')
        pose = pose.unsqueeze(0)

        trans = imu_trans[i, :]

        verts, _, _ = smpl(pose, betas)

        left_heel = torch.abs(verts[0,left,2] )
        rt_heel = torch.abs(verts[0,right,2])

        root_to_heel_fr = torch.max(left_heel, rt_heel).cpu().numpy()

        av_vel = np.linalg.norm(get_av_velocity(imu_trans, i, 6, 'mean'))

        if av_vel < 0.001 and root_to_heel_fr < 0.66 * root_to_heel:
            valid_inds.append(i)

    save_arr = np.zeros(imu_trans.shape[0])
    save_arr[valid_inds] = 1

    save_dict = {
        'sit_contacts': save_arr.tolist()
    }

    print('Number of sitting frames: {} of {}'.format(len(valid_inds), imu_pose.shape[0]))

    os.makedirs(SIT_PATH, exist_ok=True)
    save_file = SIT_PATH + file + '.json'
    save_json_file(save_file, save_dict)