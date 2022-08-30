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
import pymap3d as pm
import math
from scipy.spatial.transform import Rotation as R
import pandas as pd
from scale_confirm import qua2euler, qua2rm
from match_query import relocalize
from startingpoint_prediction import starting_point_prediction
import time
import angel_trans


path = '/media/yushichen/LENOVO_USB_HDD/projects/VisualOdometry/ipin_4/image_l'
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
db_dir = '/media/yushichen/LENOVO_USB_HDD/IPIN_res/site_A/seq_4/'
db_descriptor_dir = os.path.join(db_dir,'output_feature')
db_gt_dir = os.path.join(db_dir, 'poses.txt')
with_image_retrieval = False
with_start_point_prediction = True

longitude_scale = 0.4 #3.258
latitude_scale = 0.6 #4.5


class VisualOdometry():
    def __init__(self, data_dir, method='orb', matcher='flann'):
        # intrinsic
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        # self.gt_poses = self._load_poses(os.path.join(data_dir,"poses.txt"))
        # self.images = self._load_images(os.path.join(data_dir,"image_l"))
        self.gt_poses = self._load_poses(os.path.join(data_dir, 'pose_correct.txt'))
        # self.images = self._load_images(os.path.join(data_dir, 'image_l'))
        self.images = []
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
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1) /
                                     (np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1) + (1e-5)))
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
    print('System Start!')
    data_dir = 'ipin_5'
    method = 'superpoint'
    matcher = 'superglue'
    vo = VisualOdometry(data_dir, method=method, matcher=matcher)
    # image_path = os.listdir(path)
    images = os.listdir(os.path.join(data_dir, 'image_l'))
    images.sort()
    pose_path = os.path.join(data_dir, 'pose_converted.txt')
    # pose_df = pd.read_csv(pose_path, sep=' ', header=None)
    image_dir = os.path.join(data_dir, 'image_l')

    # play_trip(vo.images)  # Comment out to not play the trip
    gt_path = []
    estimated_path = []
    # for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size + 200, 3), dtype=np.uint8)
    half_traj_img_size = int(0.75 * traj_img_size)
    # draw_scale = 1
    draw_scale = 0.5
    vo.gt_poses = np.array(vo.gt_poses)
    poses = []
    delta = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    delta2 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    # pose_1
    # r = np.array([[9.57241851e-01, -2.88029769e-01, -2.69609328e-02], [2.89164278e-01, 9.49930449e-01, 1.18389870e-01],
    #               [-8.48879594e-03, -1.21123877e-01, 9.92601101e-01]])
    # pose_2
    # r = np.array(
    #     [[9.47843716e-01, -3.18466057e-01, 1.31019666e-02], [3.13217925e-01, 9.38263990e-01, 1.46816947e-01],
    #      [-5.90493177e-02, -1.35055750e-01, 9.89076904e-01]])

    # pose_5
    # r = np.array(
    #     [[-3.72803118e-01, -9.15745933e-01, -1.49757207e-01], [9.27776507e-01, -3.70605005e-01, -4.33898925e-02],
    #      [-1.57666527e-02, -1.55117106e-01, 9.87770265e-01]])

    # r = np.array(
    #     [[1.000000e+00, 9.043680e-12, 2.326809e-11], [9.043683e-12, 1.000000e+00, 2.392370e-10],
    #      [2.326810e-11, 2.392370e-10, 9.999999e-01]
    #      ])
    # pose_4
    # r = np.array(
    #     [[-8.35375522e-01, 5.38562801e-01, 1.09990214e-01], [-5.49491341e-01, -8.12962741e-01, -1.92745554e-01],
    #      [-1.43876398e-02, -2.21453587e-01, 9.75064769e-01]])
    # r = r @ delta @ delta2
    # t = np.zeros((3,))
    # align_transformation = np.eye(4)
    # align_transformation[:3:, :3] = r
    # align_transformation[:3, 3] = t
    global_lat, global_lon, global_alt = 0, 0, 0
    scale = 5.5
    sc1 = 1 / scale
    relocalization = False
    floor = 0.0
    for i in tqdm(range(0, len(images), 10)):
        image_path = os.path.join(image_dir, images[i])
        vo.images.append(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE))
        if i == 0:
            cur_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ])  # the first pose is from gt
            res = starting_point_prediction(image_path)
            floor = res[-1]
            # res
            # [-0.992601101, 0.11838987, -0.0269609328, -0.0,
            # -0.121123877, -0.949930449, 0.288029769, 0.0,
            # 0.00848879594, 0.289164278, 0.957241851, 0.0]
            # r
            # [[-0.95724185  0.02696093  0.28802977], [-0.28916428 - 0.11838987 - 0.94993045],
            #  [0.0084888 - 0.9926011   0.12112388]]
            # r = np.array(
            #     [[9.57241851e-01, -2.88029769e-01, -2.69609328e-02], [2.89164278e-01, 9.49930449e-01, 1.18389870e-01],
            #      [-8.48879594e-03, -1.21123877e-01, 9.92601101e-01]])

            r = np.array([[res[12], -res[8], res[4]], [res[11], -res[7], res[3]], [-res[10], res[6], -res[2]]])
            r = r @ delta @ delta2
            # t = np.zeros((3,))
            t = np.array([res[5], res[9], res[13]])
            align_transformation = np.eye(4)
            align_transformation[:3:, :3] = r
            align_transformation[:3, 3] = t
            global_lat, global_lon = res[1], res[0]
        elif (i%800) == 0:
            relocalization = True
            res = starting_point_prediction(image_path)
            floor = res[-1]
        else:
            q1, q2 = vo.get_matches(i//10)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            # cur_pose = np.matmul(transf, np.linalg.inv(cur_pose))
            # print('transf: ', np.linalg.inv(transf))
        cur_pose2 = cur_pose.copy()
        cur_pose2[:3, 3] *= sc1
        a = cur_pose2.copy()
        cur_pose2 = align_transformation @ a
        euler = angel_trans.rotation2euler(cur_pose2[0:3, 0:3])
        quaternion = angel_trans.euler2quaternion(euler)
        gt = vo.gt_poses[int(i)]
        if relocalization:
            relocalization = False
            cur_pose2 = np.array([
                [-res[12], -res[4], res[8], res[5]],
                [-res[11], -res[3], res[7], res[9]],
                [res[10], res[2], -res[6], res[13]],
                [0, 0, 0, 1]
            ])
            # cur_pose2[0][3] = res[5]
            # cur_pose2[1][3] = res[9]
            # cur_pose2[2][3] = res[13]
            # euler = angel_trans.rotation2euler(cur_pose2[0:3, 0:3])
            # quaternion = angel_trans.euler2quaternion(euler)
            cur_pose = np.linalg.inv(align_transformation) @ cur_pose2
            cur_pose[:3,3] /= sc1

        lat, lon, alt = pm.enu2geodetic(cur_pose2[0, 3], cur_pose2[1, 3], cur_pose2[2, 3], global_lat, global_lon,
                                        global_alt)
        poses.append(cur_pose2[0:3, :].reshape(1, 12))
        # gt_path.append((gt_pose[0, 3], gt_pose[2, 3])) # gt pose
        # print('\ngt: ', (gt_pose[0, 3], gt_pose[2, 3], gt_pose[1,3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))  # current pose with x, y
        # print('predict: x, y, z ', (cur_pose[0, 3], cur_pose[1, 3], cur_pose[2, 3]))
        x, y, z = cur_pose2[0, 3], cur_pose2[1, 3], cur_pose2[2, 3]
        print(x, y, z)
        x_true, y_true, z_true = vo.gt_poses[int(i), 0, 3], vo.gt_poses[int(i), 1, 3], vo.gt_poses[int(i), 2, 3]
        print(x_true, y_true, z_true)
        draw_x, draw_y = int(18 * x) + 400, 400 - int(18 * y)
        true_x, true_y = int(18 * x_true) + 400, 400 - int(18 * y_true)
        cv2.circle(traj_img, (draw_x, draw_y), 1, (i * 255 / 4540, 255 - i * 255 / 4540, 0),
                   1)  # estimated from green to blue
        cv2.circle(traj_img, (true_x, true_y), 1, (0, 0, 255), 1)  # groundtruth in red
        # write text on traj_img
        cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
        text = "frame %f : x=%2f y=%2f z=%2f" % (i, x, y, z)
        cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        # show
        cv2.imshow('Trajectory', traj_img)
        print(lat, lon, quaternion, floor)
    # poses = np.concatenate(poses, axis=0)
    # np.savetxt("pose4_test.txt", poses, delimiter=' ', fmt='%1.8e')
    # plotting.visualize_paths_without_gt(estimated_path, "Visual Odometry",
    #                                     file_out=os.path.basename(data_dir) + method + ".html")
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
