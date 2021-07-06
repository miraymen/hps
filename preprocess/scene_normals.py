"""
Compute Scene normals
"""
import trimesh
import os
import numpy as np
import open3d as o3d

import sys
sys.path.append("../")
from global_vars import *

if __name__ == "__main__":

    scenes = ["MPI_EG",
              "MPI_Etage6",
              "MPI_BIBLIO_UG",
              "MPI_BIBLIO_OG",
              "MPI_BIBLIO_EG",
              "MPI_BIB_AB_SEND",
              "MPI_KINO",
              'MPI_GEB_AB'
              ]

    normals_params = [0.95, 0.99, 0.999]

    for norm_param in normals_params:
        for scene in scenes:

            print('Processing Scene: {} with Parameter: {}'.format(scene, norm_param))
            normal_path = SCENE_PATH + '{}/10M_flat_vert{}.npy'.format(scene, norm_param)
            vis_flat_verts_path = SCENE_PATH + '{}/vis_pt_cld{}.ply'.format(scene, norm_param)
            cln_path = SCENE_PATH + '/{}/10M_clean.ply'.format(scene)

            try:
                inp_pcd = trimesh.load(cln_path)
            except:
                continue

            pcd = o3d.io.read_point_cloud(cln_path)
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

            normals = np.asarray(pcd.normals)

            norms1 = np.linalg.norm(normals, axis=1)
            norms = np.reshape(norms1, (norms1.shape[0], 1))
            norms = np.concatenate((norms, norms, norms), axis=1)

            normals = normals / norms

            flat_surf_verts = np.where(normals[:, 2] < -norm_param)[0]
            flat_surf_verts2 = np.where(normals[:, 2] > norm_param)[0]

            tot_verts = np.union1d(flat_surf_verts, flat_surf_verts2)
            np.save(normal_path, tot_verts)
            vis_mesh = trimesh.points.PointCloud(inp_pcd.vertices[tot_verts], colors=inp_pcd.colors[tot_verts]).export(vis_flat_verts_path)

