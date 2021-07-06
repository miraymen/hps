import os
import numpy as np
import pickle
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import json
from glob import glob
from tqdm import tqdm

import sys
sys.path.append("../")

from global_vars import *


def resample(imu_data, contact_data, sync_data):

    indices = [
        'left_heel', 'right_heel', 'right_toe', 'left_toe'
    ]

    imu_start = sync_data['imu_start']

    poses = imu_data['poses'].reshape(-1, 24, 3)[imu_start:]
    translations = imu_data['transes'][imu_start:]

    rotations = [Rotation.from_rotvec(poses[:, i, :]) for i in range(poses.shape[1])]

    imu_times = np.arange(0, len(translations) * 1000, 1000)
    cam_times = np.arange(0, len(translations) * 1000, 1001)[:-1]

    rot_slerps = [Slerp(imu_times, r) for r in rotations]
    interp_rots = [rot_slerp(cam_times) for rot_slerp in rot_slerps]
    interp_poses = np.array([interp_rot.as_rotvec() for interp_rot in interp_rots]).transpose((1, 0, 2)).reshape(-1, 72)

    translation_intrp = interp1d(imu_times, translations, axis=0, kind='linear')
    interp_translation = translation_intrp(cam_times)

    imu_dict = {"poses": np.vstack([imu_data['poses'][:imu_start], interp_poses]),
                "transes": np.vstack([imu_data['transes'][:imu_start], interp_translation])}

    print('Original translation length', imu_data['transes'][imu_start:].shape)
    print('New translation length', imu_dict['transes'][imu_start:].shape)

    print('Original pose length', imu_data['poses'][imu_start:].shape)
    print('New pose length', imu_dict['poses'][imu_start:].shape)

    # ====================
    #  CONTACT RESAMPLED
    # ====================
    print('Original contact length', len(contact_data['left_heel'][imu_start:]))

    contact_dict = {}

    for index in indices:
        contacts = contact_data[index][imu_start:]

        imu_times = np.arange(0, len(contacts) * 1000, 1000)
        cam_times = np.arange(0, len(contacts) * 1000, 1001)[:-1]

        interp_func = interp1d(imu_times, contacts, axis=0, kind='nearest')
        interp_contacts = interp_func(cam_times)

        new_contact = contacts[:imu_start] + list(interp_contacts)
        contact_dict[index] = new_contact

    print('New contact length', len(new_contact[imu_start:]))


    return imu_dict, contact_dict


if __name__ == "__main__":
    seqnames = [os.path.basename(os.path.splitext(x)[0]) for x in
                glob(INIT_PATH + "/*.json")]
    processed_count = 0
    for seqname in seqnames:
        # read the bake files and the contact files
        sync_path = INIT_PATH + f"/{seqname}.json"
        imu_folder = IMU_PATH + f"/{seqname}"
        contact_file = CONTACT_PATH + f"/{seqname}.pkl"

        imu_file = os.path.join(imu_folder, "bake.pkl")

        if not os.path.isfile(imu_file):
            print(f"Error: {imu_file} doesn't exist")
            continue

        print("Processing ", seqname)

        imu_data = pickle.load(open(imu_file, 'rb'))
        contact_data = pickle.load(open(contact_file, 'rb'))
        sync_data = json.load(open(sync_path))

        imu_resample, contact_resample = resample(imu_data, contact_data, sync_data)

        # save everything
        pickle.dump(imu_resample, open(imu_file, "wb"))
        pickle.dump(contact_resample, open(contact_file, "wb"))

        processed_count += 1

    print(f"Processed {processed_count} out of {len(seqnames)}")