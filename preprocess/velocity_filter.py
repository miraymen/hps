import numpy as np
import json
import pickle

def none_interp(path_dict):
    names = sorted(path_dict.keys(), key=lambda x: int(x))
    path = [path_dict[n] for n in names]
    new_path_dict = {}
    for ind in range(len(path)):
        frame = path[ind]
        if frame is None:
            prev_diff = 0
            next_diff = 0
            prev_frame = None
            next_frame = None
            for diff in range(1,ind+1):
                if path[ind-diff] is not None:
                    prev_frame = path[ind-diff]
                    prev_diff = diff
                    break
            for diff in range(1, len(path)-ind):
                if path[ind+diff] is not None:
                    next_frame = path[ind+diff]
                    next_diff = diff
                    break
            if prev_frame is not None and next_frame is not None:
                coeffs = np.array([[next_diff], [prev_diff]], dtype=np.float)/(prev_diff+next_diff)
                frame = {k:(np.array([prev_frame[k],next_frame[k]])*coeffs).sum(0) for k in prev_frame.keys()}
        new_path_dict[names[ind]] = frame
    return new_path_dict


def get_next_frames(ind, path):
    """
    for this index return the closest next frame which is non none
    :param ind: input index
    :param path: json dict
    :return:
    """
    cur_ind = ind + 1
    count = 1
    while(path[cur_ind] is None and cur_ind < (len(path) - 1)):
        count = count + 1
        cur_ind = cur_ind + 1

    return path[cur_ind], count


def get_prev_frames(ind, path):
    """
    for this index return the closest previous frame which is non none
    :param ind: input index
    :param path:
    :return: frame and the count
    """
    cur_ind = ind - 1
    count = 1
    while (path[cur_ind] is None and cur_ind >0):
        count = count + 1
        cur_ind = cur_ind - 1

    return path[cur_ind], count


def velocity_filter(path_dict, velocity_thresh = 0.01):
    nones= 0
    names = sorted(path_dict.keys(), key=lambda x: int(x))
    path = [path_dict[n] for n in names]
    new_path_dict = {}
    vels = []

    invalid_indices = []

    for ind in range(len(path)):
        frame = path[ind]
        if ind>0 and ind<len(path)-1:

            next_frame, for_ct = get_next_frames(ind, path)
            prev_frame, prev_ct = get_prev_frames(ind, path)

            if prev_frame is None or next_frame is None:
                # The previous or next frame is None only when all frames from this to the last or the first frame are None
                frame = None
                nones += 1
                invalid_indices.append(ind)

            if frame is None:
                # The frame is initialized with None
                frame = None
                invalid_indices.append(ind)
            else:
                prev_velocity = np.sqrt((((np.array(frame['position']) - np.array(prev_frame['position']))/(prev_ct ))**2).sum())
                next_velocity = np.sqrt((((np.array(frame['position']) - np.array(next_frame['position']))/(for_ct ))**2).sum())
                vel = (prev_velocity+next_velocity)/2.
                vels.append(vel)
                if vel>velocity_thresh:
                    frame = None
                    invalid_indices.append(ind)
        new_path_dict[names[ind]] = frame

    print("Number of frames which end up with none without fault: ", nones)
    return new_path_dict, np.array(vels), invalid_indices

def get_string(number):
    return "%05d" % number

def convert_json_to_pickl(data, invalids):

    pose_arr = []
    trans_arr = []

    nones = []

    names = sorted(data.keys(), key = lambda x : int(x))
    path = [data[n] for n in names]

    for ind in range(len(path)):
        frame = path[ind]
        if frame != None:
            trans = np.array(frame['position'])
            trans_arr.append(trans)

            pose = np.roll(frame['quaternion'],-1)
            pose_arr.append(pose)
        else:
            trans_arr.append([0, 0, 0])
            pose_arr.append([0, 0, 0, 0])
            nones.append(ind)


    pose_arr = np.array(pose_arr)
    trans_arr = np.array(trans_arr)
    print("Shape of camera trans array: ", trans_arr.shape)
    print('Nones while saving ', len(nones))
    cam_dict = {
        'pose': pose_arr,
        'trans': trans_arr,
        'invalid_inds': np.array(invalids)
    }
    return cam_dict


def get_cam(json_array, threshold):
    if threshold > 0:
        max_vel = 1
        all_invalid = []
        iter = 1
        test = json_array
        while max_vel > threshold:
            print("Velocity filter iter : ", iter)
            iter = iter + 1
            test, vels, invalids = velocity_filter(test, velocity_thresh=threshold)
            max_vel = np.max(vels)
            all_invalid = np.union1d(all_invalid, invalids)

        all_invalid = all_invalid.astype(int)
        print("Total invalid frames: {} out of {} ".format(len(all_invalid), len(test)))
        output = none_interp(test)
        return convert_json_to_pickl(output, all_invalid)
    else:
        output = none_interp(json_array)

        return convert_json_to_pickl(output, [])

#
# def get_unfiltered_cam(json_array):
#     output = none_interp(json_array)
#     return convert_json_to_pickl(output, [])
