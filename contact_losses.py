"""
Contact Losses
"""

import torch
import ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as cham
import numpy as np

from global_vars import *

def vectorize_inds(inp, jump):
    out = []
    for i in range(len(inp)):
        if len(inp[i]) == 0:
            continue
        else:
            out.append(inp[i] + (i * jump))

    if len(out) >0:
        out = np.concatenate(out)
    return out


def map_int_to_inds():
    get_bin = lambda x, n: format(x, 'b').zfill(n)

    verts_dict = get_verts_dict()

    main_arr = []
    for i in range(32):
        arr = []
        bin_num = get_bin(i, 5)
        if int(bin_num[0]) == 1:
            arr.append(verts_dict['back'])

        if int(bin_num[1]) == 1:
            arr.append(verts_dict['left_toe'])

        if int(bin_num[2]) == 1:
            arr.append(verts_dict['left_heel'])

        if int(bin_num[3])  == 1:
            arr.append(verts_dict['right_toe'])

        if int(bin_num[4]) == 1:
            arr.append(verts_dict['right_heel'])

        if len(arr) > 0:
            arr = np.concatenate(arr, axis = 0)
        main_arr.append(arr)

    return np.array(main_arr)

def map_contacts_to_num(right_heel, right_toe, left_heel, left_toe, back):
    end = back.shape[0]
    arr = map_int_to_inds()
    nums = 16 * back + 8* left_toe[0:end] + 4* left_heel[0:end] + 2* right_toe[0:end] + right_heel[0:end]
    nums = np.array(nums).astype(int)

    return arr[nums]

def map_int_to_inds_feet():
    get_bin = lambda x, n: format(x, 'b').zfill(n)

    verts_dict = get_verts_dict()

    main_arr = []
    for i in range(16):
        arr = []
        bin_num = get_bin(i, 4)

        if int(bin_num[0]) == 1:
            arr.append(verts_dict['left_toe'])

        if int(bin_num[1]) == 1:
            arr.append(verts_dict['left_heel'])

        if int(bin_num[2])  == 1:
            arr.append(verts_dict['right_toe'])

        if int(bin_num[3]) == 1:
            arr.append(verts_dict['right_heel'])

        if len(arr) > 0:
            arr = np.concatenate(arr, axis = 0)
        main_arr.append(arr)

    return np.array(main_arr)

def map_contacts_to_num_feet(right_heel, right_toe, left_heel, left_toe):
    arr = map_int_to_inds()
    nums = 8* left_toe + 4* left_heel + 2* right_toe + right_heel
    nums = np.array(nums).astype(int)

    return arr[nums]


def get_foot_frames(file_name):
    foot_contacts = get_foot_contacts(file_name)
    frame_verts = map_contacts_to_num_feet(
                                        foot_contacts['right_heel'],
                                        foot_contacts['right_toe'],
                                        foot_contacts['left_heel'],
                                        foot_contacts['left_toe'],
                            )
    return frame_verts

def get_foot_vel_frames(file_name):
    foot_contacts = get_foot_contacts(file_name)

    vel_conts = {}
    for stra in feet_dict:
        vel_conts[stra] = map_contacts_to_vels(foot_contacts[stra])

    frame_verts = map_contacts_to_num_feet(
                                        foot_contacts['right_heel'],
                                        foot_contacts['right_toe'],
                                        foot_contacts['left_heel'],
                                        foot_contacts['left_toe'],
                            )
    return frame_verts


def map_contacts_to_vels(contact):
    new_contact1 = contact * np.roll(contact, -1)
    new_contact1[len(contact) - 1] = 0

    return new_contact1[0:len(contact) - 1]




def contact_constraint(scene_verts, smpl_verts, frame_verts, ):

    distChamfer = cham.chamfer_3DDist()
    smpl_verts = smpl_verts.reshape(-1, 3)
    verts = smpl_verts[ frame_verts, :]
    verts = verts.reshape(-1,3).unsqueeze(0)
    temp_loss, _, _, _ = distChamfer(
                verts.contiguous(), scene_verts.cuda())
    loss = torch.mean(temp_loss)


    return loss

def sit_contact_constraint(scene_verts, smpl_verts, indices, back_verts):
    distChamfer = cham.chamfer_3DDist()
    valid_verts = smpl_verts[indices][:, back_verts, :]
    verts = valid_verts.reshape(-1,3).unsqueeze(0)
    temp_loss, _, _, _ = distChamfer(
                verts.contiguous(), scene_verts.cuda())
    loss = torch.mean(temp_loss)

    return loss


def velocity_constraint(smpl_verts, frame_verts):

    vertex_diffs = smpl_verts[:-1] - smpl_verts[1:]
    vertex_diffs = vertex_diffs.reshape(-1, 3)

    valid_vertex_diffs = vertex_diffs[frame_verts, :]
    normed_vertex_diffs = torch.norm(valid_vertex_diffs,  p = 2, dim = 1)

    loss = torch.mean(normed_vertex_diffs)
    return loss

def sit_vel_constraint(smpl_verts, indices, back_verts):

    vertex_diffs = smpl_verts[:-1] - smpl_verts[1:]

    valid_vertex_diffs = vertex_diffs[indices, :,:][:, back_verts , :]
    normed_vertex_diffs = torch.norm(valid_vertex_diffs,  p = 2, dim = 1)

    loss = torch.mean(normed_vertex_diffs)
    return loss

def contacts2indices(contacts):
    indices = np.array(range(1, contacts.shape[0]+1))
    indices_valid = indices * contacts
    indices_valid = indices_valid

    indices_valid = indices_valid[indices_valid > 0 ]
    indices_valid = indices_valid - 1

    return indices_valid

