
'''
read the orientation matrix files.
read init and actual init files -
find the nearest index in the valid indices to every invalid index
'''
import sys

sys.path.append("../")
from global_vars import *


class VelInit():
    def __init__(self, gamma=6, valid_vel_thresh=1000, comp_factor=1.02, comp_factor2 = 0.98, type='mean', percent = 0.2):
        """
        Args:
            gamma: how many frames in front or behind do you look to get the velocity
            valid_vel_thresh: what are valid velocities . If 1000 only the top percent velocities are valid
            comp_factor:
            comp_factor2:
            type:
            percent:
        """
        self.gamma = gamma
        self.valid_vel_thresh = valid_vel_thresh
        self.comp_factor = comp_factor
        self.type = type
        self.comp_factor2 = comp_factor2
        self.percent = percent

    def get_av_velocity(self, data_transes_cam, index):
        """
        Args:
            data_transes_cam:
            index:
        Returns:
        """
        total_vel = []
        gamma = np.int(self.gamma / 2)

        begin = max(0, index - gamma)
        end = min(self.end_index, index + gamma + 1)
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

        if self.type == "mean":
            if len(total_vel) != 0:
                av_vel = np.mean(total_vel, axis=0)
            else:
                av_vel = np.array([0, 0, 0])
        elif self.type == "median":
            if len(total_vel) != 0:
                av_vel = np.median(total_vel, axis=0)
            else:
                av_vel = np.array([0, 0, 0])

        av_vel[2] = 0
        return av_vel

    def get_vel_thresh(self, cam_trans, imu_trans):
        all_mags = []
        for i in range(cam_trans.shape[0]):
            imu_vel = self.get_av_velocity(imu_trans, i)
            cam_vel = self.get_av_velocity(cam_trans, i)

            imu_vel_mag = np.linalg.norm(imu_vel, ord=2)
            cam_vel_mag = np.linalg.norm(cam_vel, ord=2)

            if cam_vel_mag <= self.comp_factor * imu_vel_mag:
                all_mags.append(cam_vel_mag)
        print('Length velocities over threshold {}'.format(len(all_mags)))
        p = (self.percent * cam_trans.shape[0]) / len(all_mags)
        p = max(100 - p * 100, 0)
        print(p)
        p = np.percentile(all_mags, p)
        print(p)

        return p

    def get_matrices(self, cam_trans, imu_trans):
        self.matrices = np.zeros((cam_trans.shape[0], 3, 3))
        self.matrices[0] = np.identity(3)

        self.valid_matrix_indices = [0]

        for i in range(cam_trans.shape[0]):
            imu_vel = self.get_av_velocity(imu_trans, i)
            cam_vel = self.get_av_velocity(cam_trans, i)

            imu_vel_mag = np.linalg.norm(imu_vel, ord=2)
            cam_vel_mag = np.linalg.norm(cam_vel, ord=2)

            if cam_vel_mag <= self.comp_factor * imu_vel_mag and cam_vel_mag > self.valid_vel_thresh and (self.comp_factor2 * imu_vel_mag <= cam_vel_mag ):
                self.matrices[i, :, :] = rot_mat(imu_vel, cam_vel)
                self.valid_matrix_indices.append(i)

    def get_nearest_valid(self, index, valid_indices):
        array = np.asarray(valid_indices)
        idx = (np.abs(array - index)).argmin()
        return array[idx]

    def get_closest_prev_valid(self, index, valid_indices):
        array = np.asarray(valid_indices)
        max_val = np.max(array)
        array2 = array - index
        array2[array2 > 0] = max_val + 1

        array2 = np.abs(array2)

        id = array2.argmin()
        return array[id]

    def run(self, pose, imu_trans, cam_trans):

        self.imu_trans, self.pose = imu_trans, pose

        self.trans = cam_trans

        self.end_index = self.trans.shape[0]
        print('Sequence length: {}'.format(self.trans.shape))

        if self.valid_vel_thresh == 1000:
            self.valid_vel_thresh = self.get_vel_thresh(self.trans, self.imu_trans)

        self.get_matrices(cam_trans=self.trans, imu_trans=self.imu_trans)

        # print(valid_indices)
        # self.new_poses2 = np.copy(self.pose)
        print('Length of final valid velocites after passing all tests', len(self.valid_matrix_indices))

        for i in range(self.trans.shape[0]):
            nearest_index = self.get_closest_prev_valid(i, self.valid_matrix_indices)
            # print('  {} nearest {}'.format(i, nearest_index))
            # print(self.pose[i])
            self.pose[i] = multiply_pose(self.pose[i], self.matrices[nearest_index])
            # print(self.new_poses2[i])

        nearest_mult_dict = {
            'transes': self.trans,
            'poses': self.pose
        }

        return nearest_mult_dict

if __name__ == "__main__":

    '''
    # 
    # valid-vel_thresh == 1000 when you get the top forty percentile
    # '''
    # all_files = get_all_file_names()
    #
    # for file in all_files:
    #     solver = Solver(gamma=10, valid_vel_thresh=1000, comp_factor=1, type='mean')
    #
    #     try:
    #         file_name = "/BS/aymen/static00/DATA_CVPR21/Vis_and_test/cam_orient_correct2/pose_traj_files/{}/" \
    #                     "optimize/filt_strt-2.0_vel_filt_th-0.5_filt_fact-1000_repcamz-True_cortype-xy/{}_refined_not.pkl".format(
    #             file, file)
    #         solver.run(file_name=file, file_path=file_name)
    #     except:
    #         print("-==================--")
    #         print('does not exist')