import os
import numpy as np
import cv2

from lib.visualization import plotting
from lib.visualization.video import play_trip
from superpoint.superpoint import SuperPoint
from superpoint_test import SuperpointNet, frame2tensor
from matching import Matching
import torch

from tqdm import tqdm
import math
from scipy.spatial.transform import Rotation as R
import pandas as pd
from scale_confirm import qua2euler, qua2rm
from match_query import relocalize


config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1,
        },
    'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
    }
device = 'cuda' if torch.cuda.is_available() else 'cpu'
db_dir = '/media/yushichen/LENOVO_USB_HDD/IPIN_res/site_A/seq_1/'
db_descriptor_dir = os.path.join(db_dir,'output_feature')
db_gt_dir = os.path.join(db_dir, 'poses.txt')


class VisualOdometry():
    def __init__(self, data_dir, method='orb', matcher='flann'):
        # intrinsic
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        # self.gt_poses = self._load_poses(os.path.join(data_dir,"poses.txt"))
        self.images = self._load_images(os.path.join(data_dir,"image_l"))
        self.orb = cv2.ORB_create(3000)
        self.sift = cv2.xfeatures2d.SIFT_create(3000)
        self.superpoint = SuperpointNet(config, device)
        self.method = method
        self.matcher = matcher

        #orb
        if self.method == 'orb':
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
        # sift
        elif self.method == 'sift' or self.method == 'superpoint':
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)

        if self.matcher == 'flann':
            self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        else:
            self.superglue = Matching(config).eval().to(device)

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file

        Returns
        -------
        poses (ndarray): The GT poses
        """
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images (list): grayscale images
        """
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def get_matches(self, i):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        if self.matcher == 'flann':
            # Find the keypoints and descriptors with ORB
            if self.method == 'orb':
                kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
                kp2, des2 = self.orb.detectAndCompute(self.images[i], None)
            elif self.method == 'sift':
                kp1, des1 = self.sift.detectAndCompute(self.images[i - 1], None)
                # temp = []
                # for i in range(len(kp1)):
                #     temp.append(list(kp1[i].pt))
                kp2, des2 = self.sift.detectAndCompute(self.images[i], None)
            elif self.method == 'superpoint':
                frame_1 = frame2tensor(self.images[i-1], device)
                frame_2 = frame2tensor(self.images[i], device)
                kp1, des1 = self.superpoint(frame_1)
                kp2, des2 = self.superpoint(frame_2)

            # Find matches
            matches = self.flann.knnMatch(des1, des2, k=2)

            # Find the matches there do not have a to high distance
            good = []
            try:
                for m, n in matches:
                    if m.distance < 0.8 * n.distance:
                        good.append(m)
            except ValueError:
                pass
        else:
            frame_1 = frame2tensor(self.images[i - 1], device)
            frame_2 = frame2tensor(self.images[i], device)
            pred = self.superglue({'image0': frame_1, 'image1': frame_2})
            kpts0 = pred['keypoints0'][0].cpu().numpy()
            scores0 = pred['scores0'][0].cpu().detach().numpy()
            kpts1 = pred['keypoints1'][0].cpu().numpy()
            scores1 = pred['scores1'][0].cpu().detach().numpy()
            matches = pred['matches0'][0].cpu().numpy()
            confidence = pred['matching_scores0'][0].cpu().detach().numpy()
            valid = matches > -1
            mkpts0 = kpts0[valid]
            mkpts1 = kpts1[matches[valid]]
            mconf0 = confidence[valid]
            good = []
            for j in range(len(matches)):
                if matches[j] != -1:
                    good.append(cv2.DMatch(j, matches[j], 1-confidence[j]))
            kp1, kp2 = [], []
            for j in range(len(kpts0)):
                kp1.append(cv2.KeyPoint(kpts0[j][0], kpts0[j][1], scores0[j]))
            for j in range(len(kpts1)):
                kp2.append(cv2.KeyPoint(kpts1[j][0], kpts1[j][1], scores1[j]))
            kp1, kp2 = tuple(kp1), tuple(kp2)


        draw_params = dict(matchColor = -1, # draw matches in green color
                 singlePointColor = None,
                 matchesMask = None, # draw only inliers
                 flags = 2)

        img3 = cv2.drawMatches(self.images[i-1], kp1, self.images[i],kp2, good ,None,**draw_params)
        text = "frame %f" % (i)
        cv2.putText(img3, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        cv2.imshow("image", img3)
        cv2.waitKey(200)

        # Get the image points form the good matches
        q1 = np.float32([kp1[m.queryIdx].pt for m in good])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good])
        return q1, q2

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, threshold=1)

        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2)

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            # print('Divide: ', np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     (np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1)+(1e-5)))
            # print('relative scale: ', relative_scale)
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        t = t * relative_scale

        return [R1, t]


def main():
    # data_dir = "KITTI_sequence_1"  # Try KITTI_sequence_2 too
    data_dir = 'ipin_1'
    method = 'superpoint'
    matcher = 'superglue'
    vo = VisualOdometry(data_dir, method=method, matcher=matcher)
    images = os.listdir(os.path.join(data_dir, 'image_l'))
    images.sort()
    pose_path = os.path.join(data_dir,'poses.txt')
    pose_df = pd.read_csv(pose_path, sep=' ', header=None)
    image_dir = os.path.join(data_dir, 'image_l')



    # play_trip(vo.images)  # Comment out to not play the trip
    gt_path = []
    estimated_path = []
    # for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size+200, 3), dtype=np.uint8)
    half_traj_img_size = int(0.75 * traj_img_size)
    # draw_scale = 1
    draw_scale = 0.5
    initial_pose = []
    initial_qua = []
    initial_rotation = 0.0
    # with open(data_dir + '_' + method + '.txt', 'w+') as f:
    inital_longitude, initial_latitude = pose_df.iloc[0][1], pose_df.iloc[0][2]
    relocalization = False
    for i in tqdm(range(len(images))):
        if i == 0:
            qua_pose = [pose_df.iloc[0][4], pose_df.iloc[0][5], pose_df.iloc[0][6],
                        pose_df.iloc[0][7]] # w, x, y, z
            yaw, pitch, roll = qua2euler(qua_pose[0], qua_pose[1], qua_pose[2], qua_pose[3])
            yaw_degree = yaw * 180 / np.pi
            print('yaw: ', yaw_degree)
            rm = qua2rm(qua_pose[0], 0, qua_pose[3], 0)
            # change = [[ 1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            # [ 0.00000000e+00, 2.22044605e-16, -1.00000000e+00],
            # [ 0.00000000e+00, 1.00000000e+00, 2.22044605e-16]]
            # rm = np.matmul(rm, change)
            # r4 = R.from_euler('zxy', [yaw, roll, pitch], degrees=False)
            # rm = r4.as_matrix()
            cur_pose = np.array([
                [rm[0][0], rm[0][1], rm[0][2], 0.0],
                [rm[1][0], rm[1][1], rm[1][2], 0.0],
                [rm[2][0], rm[2][1], rm[2][2], 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ])  # the first pose is from gt

            # cur_pose = np.array([
            #     [1., 0., 0., 0.0],
            #     [0., 1. , 0., 0.0],
            #     [0., 0., 1., 0.0],
            #     [0.0, 0.0, 0.0, 1.0]
            # ]) # the first pose is from gt
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))

        Rm = cur_pose[0:3, 0:3]
        r3 = R.from_matrix(Rm)
        qua = r3.as_quat()  # x y z w
        # convert R to euler
        temp = R.from_matrix(Rm)
        euler = temp.as_euler('zxy', degrees=True)  # The second element is z
        print('current euler: ', euler)
        if i == 0 or relocalization == True:
            initial_qua = qua
            initial_rotation = euler[-1]
            relocalization = False
        print('rotation difference: ', euler[-1] - initial_rotation)
        # detect big rotation
        image_path = os.path.join(image_dir, images[i])

        if initial_rotation >= -90 and initial_rotation <= 90:
            if (euler[-1] > initial_rotation - 90) and (euler[-1] <= initial_rotation + 90):
                print('ok')
            else:
                relocalization = True
                res = relocalize(image_path, db_descriptor_dir, db_gt_dir)
                # exit(-1)
        elif initial_rotation < -90 and initial_rotation >= -180:  # three
            base_rotation = initial_rotation + 180
            if (euler[-1] > base_rotation - 90) and (euler[-1] <= base_rotation + 90):
                relocalization = True
                res = relocalize(image_path, db_descriptor_dir, db_gt_dir)
                # exit(-1)
            else:
                print('ok')
        else:  # four
            base_rotation = initial_rotation - 180
            if (euler[-1] > base_rotation - 90) and (euler[-1] <= base_rotation + 90):
                relocalization = True
                res = relocalize(image_path, db_descriptor_dir, db_gt_dir)
                # exit(-1)
            else:
                print('ok')
        if relocalization:
            correct_longitude, correct_latitude = res[1], res[2]
            correct_w, correct_x, correct_y, correct_z = res[4], res[5], res[6], res[7]
            rm = qua2rm(correct_w, 0, correct_z, 0)
            x = (correct_longitude - inital_longitude)*3.62*(1e6)
            y = 0.0
            z = (correct_latitude - initial_latitude)*5*(1e6)
            cur_pose = np.array([
                [rm[0][0], rm[0][1], rm[0][2], x],
                [rm[1][0], rm[1][1], rm[1][2], y],
                [rm[2][0], rm[2][1], rm[2][2], z],
                [0.0, 0.0, 0.0, 1.0]
            ])

        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3])) # current pose with x, y
        # print('predict: x, y, z ',(cur_pose[0, 3], cur_pose[2, 3], cur_pose[1, 3]))
        x, y, z = cur_pose[0, 3], cur_pose[2, 3], cur_pose[1, 3]
        longitude, latitude = pose_df.iloc[i][1], pose_df.iloc[i][2]
        x_true, y_true = (longitude-inital_longitude)*3.62*(1e6), (latitude - initial_latitude)*5*(1e6)
        gt_path.append((x_true, y_true))  # gt pose
        # convert R to quaterion




        draw_x, draw_y = 400 + int(draw_scale * x) , 300 - int(draw_scale * y)
        true_x, true_y = int(draw_scale*x_true) + 400, 300 - int(draw_scale*y_true)
        cv2.circle(traj_img, (draw_x, draw_y), 1, (i * 255 / 4540, 255 - i * 255 / 4540, 0),
                   1)  # estimated from green to blue
        cv2.circle(traj_img, (true_x, true_y), 1,(0, 0, 255), 1)  # groundtruth in red
        # write text on traj_img
        cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
        gps_text = 'longitude: %6f  latitude: %6f ' % (longitude, latitude)
        print(gps_text)
        cv2.putText(traj_img, gps_text, (20, 80), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        pred_text = 'pred longitude: %6f  pred latitude: %6f ' % (x/3.62*(1e-6)+inital_longitude, y/5.*(1e-6)+initial_latitude)
        cv2.putText(traj_img, pred_text, (20, 120), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        print(pred_text)
        print('real quaterion: ', pose_df.iloc[i][4], pose_df.iloc[i][5], pose_df.iloc[i][6], pose_df.iloc[i][7])
        print('pred quaterion: ', qua[-1], qua[0], qua[1], qua[2])
        print('real tendency: ', pose_df.iloc[i][4]-pose_df.iloc[0][4], pose_df.iloc[i][5]-pose_df.iloc[0][5],
              pose_df.iloc[i][6]-pose_df.iloc[0][6], pose_df.iloc[i][7]-pose_df.iloc[0][7])
        print('pred tendency: ', qua[-1]-initial_qua[-1], qua[0]-initial_qua[0], qua[1]-initial_qua[1], qua[2]-initial_qua[2])

        text = "frame %f : x=%2f y=%2f z=%2f" % (i, x, y, z)
        cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

        # show
        cv2.imshow('Trajectory', traj_img)
        # f.write(image_path[i] + ' ' + str(x) + ' ' + str(y) + ' ' + str(qua[-1]) + ' ' + str(qua[0])
        #         + ' ' + str(qua[1]) + ' ' + str(qua[2]) + '\n')

    # plotting.visualize_paths_without_gt(estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + method + ".html")
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")
    cv2.waitKey(1)

if __name__ == "__main__":
    main()
