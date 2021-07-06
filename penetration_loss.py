"""
Defines the penetration loss
"""

import torch
import torch.nn.functional as F
import trimesh
import numpy as np
import os
from scipy.spatial import cKDTree as KDTree
# import torch.functional as F
import json


def create_grid_points_from_bounds(minimun, maximum, res):
    x = np.linspace(minimun, maximum, res)
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    Z = Z.reshape((np.prod(Z.shape),))

    points_list = np.column_stack((Z,Y,X))
    del X, Y, Z, x
    return points_list


def voxelize_pt_cloud(pt_cld_norm, grid_points, res, mean_trans):
    minz = np.min(pt_cld_norm[:, 2])

    floor_indices = np.where(grid_points[:, 2] < minz + 0.2)[0]

    kdtree = KDTree(grid_points)
    _, idx = kdtree.query(pt_cld_norm)
    occupancies = np.zeros(len(grid_points))
    occupancies[idx] = 1
    occupancies[floor_indices] = 0

    print('occ ocmputed')

    occ_reshape2 = np.reshape(occupancies, (res, res, res))

    # visualize
    occ2 = np.reshape(occupancies, (occupancies.shape[0], 1))
    occ_new = np.concatenate((occ2, occ2, occ2), axis=1)
    vis_occ = occ_new * grid_points
    vis_occ = vis_occ + mean_trans
    colors = np.zeros(vis_occ.shape)
    print('saving step')
    trimesh.points.PointCloud(vertices=vis_occ, colors=colors).export(
        "occ_vis.ply")

    ## end vis

    tens_occ = torch.tensor(occ_reshape2).type(torch.FloatTensor).cuda()
    tens_occ = tens_occ.unsqueeze(0).unsqueeze(0)

    return tens_occ



def get_centroids(vertices, faces):
    face_vertices = vertices[faces]
    face_centroids = torch.mean(face_vertices, dim=1)
    return face_centroids


def get_normals(vertices, faces):
    vec1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    vec2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]

    crosses = torch.cross(vec1, vec2)

    norms1 = torch.norm(crosses, dim=1, p=2)
    norms = torch.reshape(norms1, (norms1.shape[0], 1))
    norms = torch.cat((norms, norms, norms), dim=1)

    # print(norms.shape)
    vec_normed = crosses / norms
    vec_normed[norms1 == 0] = 0

    return vec_normed


def get_ptcld(vertices, faces):
    centroids = get_centroids(vertices, faces)
    vectors = get_normals(vertices, faces)
    delta = torch.rand(vectors.shape).cuda()
    delta = 0.01 * delta

    pt_cld = centroids - delta * vectors
    # pt_cld = torch.reshape(pt_cld, (self.batch, self.faces[0], 3))
    return pt_cld



def pen_constraint(verts, faces_batch, tens_occ, mean_trans, radius, vis = False):

    batch = verts.shape[0]
    verts = torch.reshape(verts, (batch * 6890, 3))

    mean_trans_tens = torch.tensor(mean_trans).type(torch.FloatTensor).cuda()

    in_pt_clds = get_ptcld(verts, faces_batch)
    in_pt_clds_grid = (in_pt_clds - mean_trans_tens) / radius



    in_pt_clds_grid = in_pt_clds_grid.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    samples = F.grid_sample(tens_occ, in_pt_clds_grid, mode='bilinear')

    loss = torch.mean(samples)

    if vis:
        in_pt_clds_grid_sq = in_pt_clds_grid.squeeze(0).squeeze(0).squeeze(0).squeeze(0)

        samples = samples.squeeze(0).squeeze(0).squeeze(0).squeeze(0)

        save_pts_out = in_pt_clds_grid_sq[torch.nonzero(samples == 0).squeeze(1)] * radius + mean_trans_tens
        save_pts_out = save_pts_out.detach().cpu().numpy()
        colors_out = np.zeros(save_pts_out.shape)
        colors_out[:, 1] = 255

        trimesh.points.PointCloud(vertices=save_pts_out, colors=colors_out).export(
             './out_pts.ply')

        if loss > 0:
            save_pts = in_pt_clds_grid_sq[torch.nonzero(samples).squeeze(1)] * radius + mean_trans_tens
            save_pts_np = save_pts.detach().cpu().numpy()
            colors = np.zeros(save_pts_np.shape)
            colors[:, 0] = 255

            trimesh.points.PointCloud(vertices=save_pts_np, colors=colors).export(
                './in_pts.ply')

    return loss
