'''
conda activate py3d
'''

import pickle as pkl
import argparse
import numpy as np
import time

from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from datetime import datetime
import torch.nn.functional as F
import shutil
import configs.config_loader as cfg_loader
import torch
from scipy.spatial.transform import Rotation as R
import trimesh

from losses import *
from contact_losses_resampled import *
from helper import *
from global_vars import *
from vel_init_new import *
from penetration_loss import *


class Optimizer():
    def __init__(self):
        self.get_args()
        self.init()

    def get_args(self):
        self.args = cfg_loader.get_config()

    def init(self):
        # load scene
        self.mesh_file = clean_scene_from_file(self.args.file_name)

        # define scene for local window
        self.grid_points = create_grid_points_from_bounds(-self.args.radius, self.args.radius, 256)

        # imu and camera data
        self.cam_trans,self.cam_pose, self.imu_trans, self.imu_pose = sync_data(self.args.file_name, 'bake')
        self.end_index = self.imu_pose.shape[0]


        data2 = get_opt_init_data(self.args.file_name, self.args.imu_filt_start, self.args.vel_thresh)
        # self.pose, self.trans, self.invalid_indices = data2['poses'], data2['transes'], data2['invalid_indices']
        self.pose, self.trans = data2['poses'], data2['transes']
        print('trans shape', self.trans.shape)

        self.velinit = VelInit(gamma=self.args.gamma,
                               valid_vel_thresh=self.args.valid_vel_thresh,
                               comp_factor=self.args.comp_factor,
                               comp_factor2 = self.args.comp_factor2,
                               type='mean',
                               percent = self.args.percent)

        self.back_verts = torch.tensor(get_verts_dict()['back'], dtype = torch.long, device = 'cuda')

        initdata = self.velinit.run(self.pose, self.imu_trans, self.trans)
        self.pose, self.trans = initdata['poses'], initdata['transes']

        # ================
        #      CONTACT
        # ================
        self.foot_frame_verts = get_foot_frames(self.args.file_name)
        print('foot vert shape', self.foot_frame_verts.shape)

        self.sit_contacts = get_back_contacts(self.args.file_name)
        print('number sitting frames', np.sum(self.sit_contacts))

        # ====================
        #      VELOCITY
        # ====================
        self.frame_vel_verts = get_foot_vel_frames(self.args.file_name)
        self.sit_vels = map_contacts_to_vels(self.sit_contacts)


        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        date = datetime.date(now)

        self.save_folder = SAVE_FOLDER + self.args.file_name + '/vel-{}_imu-{}_ref-{}_thresh-{}_comp_fact1-{}_comp_fact2-{}_perc-{}/{}_{}_opt-{}_win-{}_moveZ-{}_ft-cont-{}_ft-vel-' \
                                                               '{}_sit-cot-{}_sit-vel-{}_pen-{}_rot-{}_trans-{}_v2v-{}_trimu-{}_intr-{}_inrt-{}_/'.\
            format(
            self.args.vel_thresh, self.args.imu_filt_start, self.args.refined,
            self.args.valid_vel_thresh,
            self.args.comp_factor, self.args.comp_factor2, self.args.percent,
            date, current_time,
            self.args.opt_vars,
            self.args.window_frames, self.args.moveZ, self.args.wt_ft_cont, self.args.wt_ft_cont, self.args.wt_st_cont, self.args.wt_st_vel, self.args.wt_pen,
            self.args.smooth_glob, self.args.trans_smooth_weight, self.args.v2v_weight, self.args.trans_imu_smth,
            self.args.init_trans_weight, self.args.init_rot_weight
            )

        os.makedirs(self.save_folder, exist_ok=True)

        init_dict ={
            'poses':self.pose,
            'transes': self.trans
        }

        save_pkl(self.save_folder + 'init.pkl', init_dict)

        self.scene = trimesh.load(self.mesh_file)

        self.betas = beta_from_file(self.args.file_name)
        self.gender = gender_from_file(self.args.file_name)

        if self.args.moveZ:
            self.trans[:,2] = self.trans[:, 2] - self.args.moveZ


        self.smpl_layer = SMPL_Layer(
            center_idx=0,
            gender=self.gender,
            model_root='/BS/aymen/work/20_09_29-Optimization/my_code2/smplpytorch/smplpytorch/native/models')

        self.faces = self.smpl_layer.th_faces
        self.smpl_layer.cuda()

        flat_vert_file = normals_from_file(self.args.file_name, self.args.normal_thresh)
        self.flat_verts = np.load(flat_vert_file)

        print('len flat vert', len(self.flat_verts))

    def get_head_offset(self):
        PY_TRANS5 = torch.tensor([[[-1, 0, 0], [0, -1, 0], [0, 0, 1]]]).type(torch.FloatTensor).cuda()
        # batch_pytrans = PY_TRANS5.repeat((pred_orientations.size(0), 1, 1)).type(torch.FloatTensor).cuda()

        pose_params = torch.tensor(self.pose[0]).unsqueeze(0).type(torch.FloatTensor).cuda()
        trans_params = torch.tensor([[0, 0 ,0]]).type(torch.FloatTensor).cuda()
        betas_torch = torch.tensor(self.betas, requires_grad=False).unsqueeze(0).type(torch.FloatTensor).cuda()

        verts, orients =  self.smpl_pass(pose_params, trans_params, betas_torch)

        rhead0 =  orients[:,15,:,:].type(torch.FloatTensor).cuda()

        rcam0 = torch.tensor(R.from_quat(np.reshape(self.cam_pose[0], (1,4))).as_matrix()).type(torch.FloatTensor).cuda()
        rcam0 = torch.bmm(rcam0, PY_TRANS5)
        self.offset = torch.bmm(rhead0.permute(0,2,1), rcam0)


    def smpl_pass(self, pose_params, trans_params, betas):
        smpl_verts, _, orientations = self.smpl_layer(
            th_pose_axisang=pose_params.cuda(),
            th_betas=betas.cuda()
        )

        rep_trans = trans_params.unsqueeze(1).repeat(1, 6890, 1)
        smpl_verts = smpl_verts + rep_trans.cuda()
        return smpl_verts, orientations

    def define_window(self, index):
        start = index
        end = min(self.end_index, index + self.args.window_frames)
        return np.sort(list(range(start, end))).astype(int)



    def optimize(self):
        for i in range(self.args.start, self.end_index, self.args.window_frames):
            # camera indices
            window = self.define_window(i)
            betas_torch_batch = torch.tensor(self.betas, requires_grad=False).unsqueeze(0).repeat(
                len(window), 1)
            # contacts for window

            # Initialize pose and trans variables
            transes = self.trans[window]
            poses = self.pose[window]

            cam_poses = self.cam_pose[window]

            cont_verts = self.foot_frame_verts[window]
            cont_verts = vectorize_inds(cont_verts, 6890)
            cont_verts = torch.tensor(cont_verts, dtype = torch.long, device = 'cuda')

            vel_verts = self.frame_vel_verts[window[0:len(window) -1]]
            vel_verts = vectorize_inds(vel_verts, 6890)
            vel_verts = torch.tensor(vel_verts, dtype=torch.long, device='cuda')

            sit_window = self.sit_contacts[window]
            sit_indices = torch.tensor(contacts2indices(sit_window), dtype = torch.long, device = 'cuda')

            sit_vel_window = self.sit_vels[window[0:len(window) -1]]
            print(sit_vel_window.shape)
            sit_vel_inds = torch.tensor(contacts2indices(sit_vel_window), dtype = torch.long, device = 'cuda')

            # compute average of the translations over window:
            mean_trans = np.mean(transes, axis = 0)

            # Vertices in the local window - the whole cube
            verts_inds_cube = find_verts(self.scene.vertices, mean_trans[0] - self.args.radius, mean_trans[0] + self.args.radius,
                                    mean_trans[1] - self.args.radius, mean_trans[1] + self.args.radius, mean_trans[2] - self.args.radius,
                                    mean_trans[2] + self.args.radius
                                    )

            print('verts cube', len(verts_inds_cube))

            verts_cube = self.scene.vertices[verts_inds_cube]

            # Lower half of the cube
            verts_inds_half_cube = find_verts(verts_cube, mean_trans[0] - self.args.radius, mean_trans[0] + self.args.radius,
                                    mean_trans[1] - self.args.radius, mean_trans[1] + self.args.radius, mean_trans[2] - self.args.radius,
                                    mean_trans[2]
                                              )
            print('verts half cube', len(verts_inds_half_cube))

            window_half_pt_cld = trimesh.Trimesh(vertices = verts_cube[verts_inds_half_cube])
            window_half_pt_cld.export('test.ply')

            print('flat', len(self.flat_verts))
            window_pt_cloud = trimesh.Trimesh(vertices = verts_cube)
            window_pt_cloud_norm = window_pt_cloud.vertices - mean_trans

            tens_occ = voxelize_pt_cloud(window_pt_cloud_norm, self.grid_points, 256, mean_trans)

            flat_verts_inds = np.intersect1d(np.array(verts_inds_cube), self.flat_verts)

            flat_verts = self.scene.vertices[flat_verts_inds]
            scene_verts_half_pt_cld = torch.from_numpy(window_half_pt_cld.vertices).unsqueeze(0).type(torch.FloatTensor)



            if np.sum(sit_window) >0:
                sit_mean_trans = np.mean(transes[sit_indices.cpu().numpy()], axis = 0)
                print(sit_mean_trans)
                verts_inds2 = find_verts_z(flat_verts, sit_mean_trans[2] - self.args.z_sit_min, sit_mean_trans[2] + self.args.z_sit_max)
                print(len(verts_inds2))

                sit_scene_verts = torch.from_numpy(flat_verts[verts_inds2]).unsqueeze(0).type(torch.FloatTensor)
                big_iter = 3
            else:
                big_iter = 1
            batch = len(window)
            faces_batch = self.faces.unsqueeze(0).repeat(batch, 1, 1)

            for j in range(batch):
                faces_batch[j] = faces_batch[j] + 6890 * j

            faces_batch = torch.reshape(faces_batch, (batch * self.faces.shape[0], 3)).type(torch.LongTensor)

            # =======
            trans_params, pose_glob, pose_between_glob_and_head, pose_head, pose_after_head = define_variables(poses,
                                                                                                               transes)

            for iter in range(big_iter):
                if iter == 0:
                    if np.sum(sit_window) > 0:
                        # sitting first iter
                        self.args.learn_rate = 0.01
                        self.args.iterations = 1000
                        self.args.init_trans_weight = 4000

                elif iter == 1:
                    # sitting second iter
                    self.args.wt_pen = 600
                    self.args.learn_rate = 0.01
                    self.args.iterations = 500
                    self.args.init_trans_weight = 5000

                else:
                    # sitting third and fourth iter
                    self.args.wt_pen = 400*(iter+1)
                    # self.args.learn_rate = 0.001 * (1 / (iter *2))
                    self.args.learn_rate = 0.001
                    self.args.iterations = 200
                    self.args.init_trans_weight = (iter +1) * 10000

                if np.sum(sit_window)>0:
                    trans_params.requires_grad = True
                    # optimizer = torch.optim.Adam([trans_params], self.args.learn_rate, betas=(0.9, 0.999))
                    optimizer = torch.optim.Adam([trans_params], lr = self.args.learn_rate)
                else:
                    if self.args.opt_vars == "trans":
                        trans_params.requires_grad = True
                        # optimizer = torch.optim.Adam([trans_params], self.args.learn_rate, betas=(0.9, 0.999))
                        optimizer = torch.optim.Adam([trans_params], lr = self.args.learn_rate)

                    if self.args.opt_vars == "trans_allpose":
                        trans_params.requires_grad = True
                        pose_glob.requires_grad = True
                        pose_between_glob_and_head.requires_grad = True
                        pose_head.requires_grad = True
                        pose_after_head.requires_grad = True
                        optimizer = torch.optim.Adam([trans_params, pose_glob, pose_between_glob_and_head, pose_head, pose_after_head], self.args.learn_rate, betas=(0.9, 0.999))

                    if self.args.opt_vars == "trans_glob":
                        trans_params.requires_grad = True
                        pose_glob.requires_grad = True
                        optimizer = torch.optim.Adam([trans_params, pose_glob], self.args.learn_rate)

                pose_params = torch.cat([pose_glob, pose_between_glob_and_head, pose_head, pose_after_head], dim = 1)
                # define contact and velocity indices once per window

                # lambda1 = lambda epoch: 0.99 * epoch
                # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1], last_epoch=-1)

                for it in range(self.args.iterations):

                    str_time = time.time()
                    optimizer.zero_grad()

                    smpl_verts, orientations = self.smpl_pass(pose_params, trans_params, betas_torch_batch)

                    tot_loss = 0.0
                    print_str = ''

                    print('contact vertices shape', cont_verts.shape)

                    if self.args.wt_ft_cont and cont_verts.shape[0] > 0:
                        contact_loss = contact_constraint(scene_verts_half_pt_cld, smpl_verts, cont_verts)
                        tot_loss += self.args.wt_ft_cont * contact_loss
                        print_str = print_str + 'Foot Contact : {} \n'.format(contact_loss.item())

                    if self.args.wt_ft_vel and cont_verts.shape[0] > 0:
                        vel_loss = velocity_constraint(smpl_verts, vel_verts)
                        tot_loss += self.args.wt_ft_vel* vel_loss
                        print_str = print_str + 'Velocity : {} \n'.format(vel_loss.item())

                    if self.args.wt_st_cont and np.sum(sit_window) >0:
                        st_cont_loss = sit_contact_constraint(sit_scene_verts, smpl_verts, sit_indices, self.back_verts)

                        tot_loss += self.args.wt_st_cont * st_cont_loss
                        print_str = print_str + 'Sit Contact : {} \n'.format(st_cont_loss.item())

                    if self.args.wt_st_vel and np.sum(sit_vel_window ) > 0:
                        st_vel_loss = sit_vel_constraint( smpl_verts, sit_vel_inds, self.back_verts)
                        tot_loss += self.args.wt_st_vel * st_vel_loss
                        print_str = print_str + 'Sit Velocity : {} \n'.format(st_vel_loss.item())

                    if self.args.wt_pen:
                        pen_loss = pen_constraint(smpl_verts, faces_batch, tens_occ, mean_trans, self.args.radius)
                        tot_loss += self.args.wt_pen * pen_loss
                        print_str = print_str + 'Penetration : {} \n'.format(pen_loss.item())

                    if self.args.trans_smooth_weight:
                        trans_smooth_loss = trans_smooth(trans_params.cuda(), self.args.trans_max)
                        tot_loss += self.args.trans_smooth_weight * trans_smooth_loss
                        print_str = print_str + 'Smooth trans : {} \n'.format(trans_smooth_loss.item())

                    if self.args.smooth_glob:
                        glob_smooth_loss = joint_orient_error(orientations[:-1, 0, :, :],
                                                                orientations[1:, 0, :, :])
                        tot_loss += self.args.smooth_glob* glob_smooth_loss
                        print_str = print_str + 'Rot smooth : {} \n'.format(glob_smooth_loss.item())

                    if self.args.v2v_weight:
                        v2v_loss_val = v2vloss(smpl_verts)
                        tot_loss += self.args.v2v_weight * v2v_loss_val
                        print_str = print_str + 'V2V : {} \n'.format(v2v_loss_val.item())

                    if self.args.trans_imu_smth:
                        trans_imu_smth_loss = trans_imu_smooth(trans_params, self.imu_trans[window])
                        tot_loss += self.args.trans_imu_smth * trans_imu_smth_loss

                        print_str = print_str + 'Trans IMU smth : {} \n'.format(trans_imu_smth_loss.item())


                    if self.args.head_orientation:
                        head_orientaiton_error = head_constraint(self.offset, orientations[:, 15, :, :],
                                                                 cam_poses)
                        tot_loss += self.args.head_orientation * head_orientaiton_error
                        print_str = print_str + 'Head Orientation : {} \n'.format(head_orientaiton_error.item())

                    # =======================
                    #
                    # =======================
                    if i != 0:
                        if self.args.init_vel_weight:
                                prev_pose_params = torch.tensor(self.pose[i -1]).type(torch.FloatTensor).cuda().unsqueeze(0)
                                prev_trans_params = torch.tensor(self.trans[i -1]).type(torch.FloatTensor).cuda().unsqueeze(0)

                                betas_torch_prev = torch.tensor(self.betas, requires_grad=False).unsqueeze(0)

                                prev_smpl_verts, _ = self.smpl_pass(prev_pose_params, prev_trans_params, betas_torch_prev)
                                verts_forward = torch.cat([prev_smpl_verts, smpl_verts[0].unsqueeze(0)], dim = 0)

                                vel_verts = torch.tensor(self.frame_vel_verts[window[len(window)-1]], dtype = torch.long, device = 'cuda')

                                vel_loss_init = velocity_constraint(verts_forward, vel_verts)
                                tot_loss += self.args.init_vel_weight * vel_loss_init
                                print_str = print_str + 'Init velocity : {} \n'.format(vel_loss_init.item())

                        if self.args.init_trans_weight:
                            if i != 0:
                                init_trans_diff = torch.tensor(self.trans[i - 1]).type(torch.FloatTensor).cuda() - trans_params[0]
                                init_trans_norm = torch.norm(init_trans_diff, p = 2)

                                imu_imu_diff = torch.tensor(self.imu_trans[i] - self.imu_trans[i-1]).type(torch.FloatTensor).cuda()
                                init_imu_norm = torch.norm(imu_imu_diff, p = 2)

                                init_trans_loss = torch.nn.functional.relu(init_trans_norm - init_imu_norm)

                                tot_loss += self.args.init_trans_weight * init_trans_loss
                                print_str = print_str + 'Init trans : {} \n'.format(init_trans_loss.item())

                        if self.args.init_rot_weight:
                            if i!= 0:
                                init_rot = self.pose[i - 1, :3]
                                r = R.from_rotvec(init_rot)
                                gt_matrs = r.as_matrix()
                                init_rot = torch.tensor(gt_matrs).type(torch.FloatTensor).cuda().unsqueeze(0)

                                win_first_rot = orientations[0, 0, :, :].unsqueeze(0)
                                init_rot_loss = joint_orient_error(win_first_rot, init_rot)
                                tot_loss += self.args.init_rot_weight * init_rot_loss
                                print_str = print_str + 'Init rot : {} \n'.format(init_rot_loss.item())


                    print_str = 'Total loss: {} \n'.format(tot_loss.item()) + print_str
                    print(print_str)

                    tot_loss.backward()
                    optimizer.step()
                    # scheduler.step(it)
                    pose_params = torch.cat([pose_glob, pose_between_glob_and_head, pose_head, pose_after_head], dim=1)

                    end_time = time.time()
                    print('Window number: {} Iter: {}, time taken {}'.format(i, it, end_time - str_time))
                    print('big iter {}'.format(iter))
                    print(self.args.file_name)
                # fit
                print('saving')
                self.trans[window] = trans_params.detach().cpu().numpy()
                self.pose[window] = pose_params.detach().cpu().numpy()

                save_dict = {
                    'poses': self.pose[:(i + len(window))],
                    'transes': self.trans[:(i + len(window))]
                }
                save_pkl(self.save_folder + 'save_params.pkl', save_dict)


if __name__ == "__main__":

    optimizer = Optimizer()
    try:
        optimizer.optimize()
    except KeyboardInterrupt:
        shutil.rmtree(optimizer.save_folder)
