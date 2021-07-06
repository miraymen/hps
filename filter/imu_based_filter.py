'''
Code to filter out the incorrect localization results
'''

import argparse
import sys
import numpy as np

sys.path.append("../")
# Own Imports
from global_vars import  *


class ImuFilterPkl:
    def __init__(self, factor=4 , vel_type="prev", cor_type = "xy"):
        """
        Args:
            threshold: The maximum velocity threshold allowed
        """
        self.factor = factor
        self.vel_type = vel_type
        self.cor_type = cor_type

    def get_velocity(self, curr_pos, next_pos, prev_pos):
        """
        Args:
            curr_pos:
            next_pos:
            prev_pos:
        Returns:
        """

        bck_diff = curr_pos - prev_pos
        fr_diff = next_pos - curr_pos

        if self.cor_type == "xy":
            bck_diff[2] = 0
            fr_diff[2] = 0

        prev_velocity = np.sqrt((bck_diff ** 2).sum())
        next_velocity = np.sqrt((fr_diff ** 2).sum())
        if self.vel_type == "both":
            vel = (prev_velocity + next_velocity) / 2.
        if self.vel_type == "next":
            vel = next_velocity
        if self.vel_type == "prev":
            vel = prev_velocity

        return vel

    def visualize(self, file_name, trans_array):
        """
        Returns:

        """
        os.makedirs(IMU_FILT_PATH, exist_ok=True)
        save_path = IMU_FILT_PATH + '{}_factor-{}_vel-{}.ply'.format(file_name, self.factor, self.vel_type)
        colors = np.zeros(trans_array.shape)
        trimesh.points.PointCloud(vertices=trans_array, colors=colors).export(save_path)

    def get_max_vel(self,  cam_trans):
        vels = []

        for ind in range(len(cam_trans)):
            if ind > 0 and ind < len(cam_trans) - 1:

                next_frame_pos = cam_trans[ind + 1, :]
                prev_frame_pos = cam_trans[ind - 1, :]
                cur_frame_pos = cam_trans[ind, :]

                cam_vel = self.get_velocity(cur_frame_pos, next_frame_pos, prev_frame_pos)
                vels.append(cam_vel)

        max_vel = np.max(vels)
        max_vel_ind = np.where(vels == max_vel)[0]
        return max_vel, max_vel_ind

    def run_array(self, imu_trans, cam_trans):
        """
        Takes as input synced imu and camera translation arrays,
        Returns the filtered camera translation array
        Args:
            imu_trans:
            cam_trans:
        Returns:
        """
        max_vel = 0.0
        end_index = cam_trans.shape[0]
        nones = 0
        invalid_indices = []

        # return camera position array
        trans_array_ret = np.copy(cam_trans)

        # if self.vel_type == "next":
        #     start = -1
        #
        # if self.vel_type == "both":
        #     start = 0

        for ind in range(end_index):
            if ind > 0 and ind < len(cam_trans) :
                #exclude the first and the last index

                if ind != len(cam_trans) - 1:
                    next_frame_pos = np.copy(cam_trans[ind + 1, :])
                    next_frm_imu_pos = np.copy(imu_trans[ind + 1, :])
                else:
                    next_frame_pos = np.copy(cam_trans[ind, :])
                    next_frm_imu_pos = np.copy(imu_trans[ind, :])

                prev_frame_pos = np.copy(cam_trans[ind - 1, :])
                cur_frame_pos = np.copy(cam_trans[ind, :])

                prev_frm_imu_pos = np.copy(imu_trans[ind - 1, :])
                cur_frm_imu_pos = np.copy(imu_trans[ind, :])

                if self.cor_type == "xy":
                    next_frame_pos[2] = 0
                    prev_frame_pos[2] = 0
                    cur_frame_pos[2] = 0

                    next_frm_imu_pos[2] = 0
                    prev_frm_imu_pos[2] = 0
                    cur_frm_imu_pos[2] = 0


                if np.linalg.norm(prev_frame_pos) == 0 or np.linalg.norm(next_frame_pos) == 0:
                    # The previous or next frame is invalid
                    trans_array_ret[ind, :] = 0
                    nones += 1
                    invalid_indices.append(ind)

                elif np.linalg.norm(cur_frame_pos) == 0:
                    # The frame is invalid from the start
                    invalid_indices.append(ind)
                    nones += 1

                else:
                    # Now filter values based on the velocity
                    imu_vel = self.get_velocity(cur_frm_imu_pos, next_frm_imu_pos, prev_frm_imu_pos)
                    cam_vel = self.get_velocity(cur_frame_pos, next_frame_pos, prev_frame_pos)
                    max_vel = max(cam_vel, max_vel)
                    if cam_vel >= self.factor * imu_vel:
                        # If the velocity
                        trans_array_ret[ind, :] = 0
                        invalid_indices.append(ind)
                        nones += 1

        # print("Invalid frames: ", nones)
        return trans_array_ret, invalid_indices, max_vel

    def run(self, file_name):
        """
        Args:
            file_name:
        Returns:
        test: The camera location with outliers removed. All outlier indices have value [0, 0, 0]
        all_invalid : list of invalid indices
        """

        cam_trans, cam_pose, imu_trans, imu_pose = sync_data(file_name, 'bake')
        trans_array_ret, invalid_indices, max_vel = self.run_array(imu_trans, cam_trans)
        self.visualize(file_name, trans_array_ret)
        return trans_array_ret, invalid_indices, max_vel

    def run_all(self):
        """
        Returns:
        """
        all_names = os.listdir(CAM_PATH)
        for file_name in all_names:
            self.run(file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type = str, help = "all for all the files", default = "Bharat_MPI_Etage6_eval")
    parser.add_argument("--factor", default = 2, type = float, help = "The factor of multiplication allowed")
    parser.add_argument("--vel_type", type = str, default = "next", help = "next|both")
    args = parser.parse_args()

    filter = ImuFilterPkl(factor = args.factor, vel_type=args.vel_type)

    if args.file_name == "all":
        filter.run_all()
    else:
        filter.run(args.file_name)



