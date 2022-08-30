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
import pymap3d as pm
import pandas as pd
from scale_confirm import qua2euler, qua2rm
from match_query import relocalize
from startingpoint_prediction import starting_point_prediction, starting_point_prediction_for_trial
import time
from time import perf_counter
from PIL import Image
import angel_trans


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

with_image_retrieval = True
with_start_point_prediction = True

longitude_scale = 0.4
latitude_scale = 0.6


class VisualOdometry():
    def __init__(self, method, matcher):
        # intrinsic
        self.K, self.P = self._load_calib('/media/yushichen/LENOVO_USB_HDD/projects/VisualOdometry/ipin_1/calib.txt')
        # self.gt_poses = self._load_poses(os.path.join(data_dir,"poses.txt"))
        # self.images = self._load_images(os.path.join(data_dir,"image_l"))
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

        # img3 = cv2.drawMatches(self.images[i-1], kp1, self.images[i],kp2, good ,None,**draw_params)
        # text = "frame %f" % (i)
        # cv2.putText(img3, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        # cv2.imshow("image", img3)
        # cv2.waitKey(1)

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
        if E is None:
            return None

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
        if E.data.contiguous == False:
            E = np.ascontiguousarray(E)
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


def camloc(vo, image_id, image_data, qua_pose, initial_longitude, initial_latitude, cur_pose, traj_img):
    # data_dir = "KITTI_sequence_1"  # Try KITTI_sequence_2 too
    image_id = int(image_id)
    ts0 = perf_counter()
    # print('System Start!')
    # method = 'superpoint'
    # matcher = 'superglue'
    # vo = VisualOdometry(method=method, matcher=matcher)
    # cur_pose = np.array([
    #     [1., 0., 0., 0.0],
    #     [0., 1. , 0., 0.0],
    #     [0., 0., 1., 0.0],
    #     [0.0, 0.0, 0.0, 1.0]
    # ]) # the first pose is from gt

    # estimated_path = []
    # for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
    # traj_img = np.zeros((traj_img_size, traj_img_size+200, 3), dtype=np.uint8)
    draw_scale = 0.5
    initial_rotation = 0.0
    relocalization = False
    vo.images.append(cv2.cvtColor(np.asarray(image_data), cv2.COLOR_RGB2GRAY))
    if image_id == 0:
        if with_start_point_prediction:
            # res = relocalize(image_path, db_descriptor_dir, db_gt_dir)
            time1 = time.time()
            res = starting_point_prediction_for_trial(image_data)
            time2 = time.time()
            # print('Strarting Point Prediction: ', time2 - time1)
            qua_pose = [res[3], res[4], res[5], res[6]]
            initial_longitude[0], initial_latitude[0] = res[0], res[1]
        yaw, pitch, roll = qua2euler(qua_pose[0], qua_pose[1], qua_pose[2], qua_pose[3])
        yaw_degree = yaw * 180 / np.pi
        # print('yaw: ', yaw_degree)
        rm = qua2rm(qua_pose[0], 0, qua_pose[3], 0)
        start_point = np.array([
            [rm[0][0], rm[0][1], rm[0][2], 0.0],
            [rm[1][0], rm[1][1], rm[1][2], 0.0],
            [rm[2][0], rm[2][1], rm[2][2], 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])  # the first pose is from gt
        for i in range(4):
            for j in range(4):
                cur_pose[i][j] = start_point[i][j]
    else:
        time5 = time.time()
        q1, q2 = vo.get_matches(image_id)
        print(len(q1), len(q2))
        if len(q1)>0 and len(q2)>0:
            transf = vo.get_pose(q1, q2)
            cur_point = np.matmul(cur_pose, np.linalg.inv(transf))
            for i in range(4):
                for j in range(4):
                    cur_pose[i][j] = cur_point[i][j]
        # else:
        #     # relocalization = True
        #     res = starting_point_prediction_for_trial(image_data)
        #     correct_longitude, correct_latitude = res[0], res[1]
        #     correct_w, correct_x, correct_y, correct_z = res[3], res[4], res[5], res[6]
        #     rm = qua2rm(correct_w, 0, correct_z, 0)
        #     x = (correct_longitude - initial_longitude[0]) * longitude_scale * (1e6)
        #     y = 0.0
        #     z = (correct_latitude - initial_latitude[0]) * latitude_scale * (1e6)
        #     start_point = np.array([
        #         [rm[0][0], rm[0][1], rm[0][2], x],
        #         [rm[1][0], rm[1][1], rm[1][2], y],
        #         [rm[2][0], rm[2][1], rm[2][2], z],
        #         [0.0, 0.0, 0.0, 1.0]
        #     ])
        #     for i in range(4):
        #         for j in range(4):
        #             cur_pose[i][j] = start_point[i][j]

        time6 = time.time()
        # print('Pose estimation: ', time6 - time5)

    Rm = cur_pose[0:3, 0:3]
    r3 = R.from_matrix(Rm)
    qua = r3.as_quat()  # x y z w
    # convert R to euler
    temp = R.from_matrix(Rm)
    euler = temp.as_euler('zxy', degrees=True)  # The second element is z
    # print('current euler: ', euler)
    if image_id == 0 or relocalization == True:
        # initial_qua = qua
        initial_rotation = euler[-1]
        relocalization = False
    # print('rotation difference: ', euler[-1] - initial_rotation)
    # detect big rotation
    if with_image_retrieval:
        time3 = time.time()
        # if initial_rotation >= -90 and initial_rotation <= 90:
        #     if (euler[-1] > initial_rotation - 90) and (euler[-1] <= initial_rotation + 90):
        #         print('ok')
        #     else:
        #         relocalization = True
        #         # res = relocalize(image_path, db_descriptor_dir, db_gt_dir)
        #         res = starting_point_prediction_for_trial(image_data)
        #         # exit(-1)
        # elif initial_rotation < -90 and initial_rotation >= -180:  # three
        #     base_rotation = initial_rotation + 180
        #     if (euler[-1] > base_rotation - 90) and (euler[-1] <= base_rotation + 90):
        #         relocalization = True
        #         # res = relocalize(image_path, db_descriptor_dir, db_gt_dir)
        #         res = starting_point_prediction_for_trial(image_data)
        #         # exit(-1)
        #     else:
        #         print('ok')
        # else:  # four
        #     base_rotation = initial_rotation - 180
        #     if (euler[-1] > base_rotation - 90) and (euler[-1] <= base_rotation + 90):
        #         relocalization = True
        #         # res = relocalize(image_path, db_descriptor_dir, db_gt_dir)
        #         res = starting_point_prediction_for_trial(image_data)
        #     else:
        #         print('ok')
        if image_id%80 == 0:
            relocalization = True
            res = starting_point_prediction_for_trial(image_data)
        if relocalization:
            correct_longitude, correct_latitude = res[0], res[1]
            correct_w, correct_x, correct_y, correct_z = res[3], res[4], res[5], res[6]
            rm = qua2rm(correct_w, 0, correct_z, 0)
            x = (correct_longitude - initial_longitude[0])*longitude_scale*(1e6)
            y = 0.0
            z = (correct_latitude - initial_latitude[0])*latitude_scale*(1e6)
            start_point = np.array([
                [rm[0][0], rm[0][1], rm[0][2], x],
                [rm[1][0], rm[1][1], rm[1][2], y],
                [rm[2][0], rm[2][1], rm[2][2], z],
                [0.0, 0.0, 0.0, 1.0]
            ])
            for i in range(4):
                for j in range(4):
                    cur_pose[i][j] = start_point[i][j]
        time4 = time.time()
        # print('Relocalization: ', time4 - time3)

    # estimated_path.append((cur_pose[0, 3], cur_pose[2, 3])) # current pose with x, y
    x, y, z = cur_pose[0, 3], cur_pose[2, 3], cur_pose[1, 3]

    draw_x, draw_y = 400 + int(draw_scale * x) , 400 - int(draw_scale * y)
    # true_x, true_y = int(draw_scale*x_true) + 400, 400 - int(draw_scale*y_true)
    cv2.circle(traj_img, (draw_x, draw_y), 1, (image_id * 255 / 4540, 255 - image_id * 255 / 4540, 0),1)  # estimated from green to blue
    cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
    pred_text = 'pred longitude: %6f  pred latitude: %6f ' % (x/3.62*(1e-6)+initial_longitude, y/5.*(1e-6)+initial_latitude)
    cv2.putText(traj_img, pred_text, (20, 120), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
    print(pred_text)
    # print('real quaterion: ', pose_df.iloc[i][4], pose_df.iloc[i][5], pose_df.iloc[i][6], pose_df.iloc[i][7])
    print('pred quaterion: ', qua[-1], qua[0], qua[1], qua[2])

    text = "frame %f : x=%2f y=%2f z=%2f" % (image_id, x, y, z)
    cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

    # show
    cv2.imshow('Trajectory', traj_img)
    # f.write(image_path[i] + ' ' + str(x) + ' ' + str(y) + ' ' + str(qua[-1]) + ' ' + str(qua[0])
    #         + ' ' + str(qua[1]) + ' ' + str(qua[2]) + '\n')

    # plotting.visualize_paths_without_gt(estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + method + ".html")
    # plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + "_retrieval.html")
    cv2.waitKey(1)
    lon = x/longitude_scale*(1e-6)+initial_longitude[0]
    lat = y/latitude_scale*(1e-6)+initial_latitude[0]
    floor = 3
    proc_time = perf_counter() - ts0
    q_w, q_x, q_y, q_z = qua[-1], qua[0], qua[1], qua[2]
    return {"lon": lon, "lat": lat, "floor": floor, "proc_time": proc_time, "q_w": q_w, "q_x": q_x, "q_y": q_y, "q_z": q_z}


def camloc2(vo, image_id, image_data, floor, cur_pose, traj_img, align_transformation,
             global_lat, global_lon, global_alt):
    # print('System Start!')
    image_id = int(image_id)
    ts0 = perf_counter()
    # data_dir = 'ipin_1'
    # method = 'superpoint'
    # matcher = 'superglue'
    # vo = VisualOdometry(data_dir, method=method, matcher=matcher)
    # image_path = os.listdir(path)
    # images = os.listdir(os.path.join(data_dir, 'image_l'))
    # images.sort()
    # pose_path = os.path.join(data_dir, 'pose_converted.txt')
    # pose_df = pd.read_csv(pose_path, sep=' ', header=None)
    # image_dir = os.path.join(data_dir, 'image_l')

    # play_trip(vo.images)  # Comment out to not play the trip
    gt_path = []
    # estimated_path = []
    # for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
    # traj_img_size = 800
    # traj_img = np.zeros((traj_img_size, traj_img_size + 200, 3), dtype=np.uint8)
    # half_traj_img_size = int(0.75 * traj_img_size)
    # draw_scale = 1
    draw_scale = 0.5
    # vo.gt_poses = np.array(vo.gt_poses)
    poses = []
    delta = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    delta2 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    # global_lat, global_lon, global_alt = 0, 0, 0
    scale = 5.5
    sc1 = 1 / scale
    relocalization = False
    # floor = 0.0
    vo.images.append(cv2.cvtColor(np.asarray(image_data), cv2.COLOR_RGB2GRAY))

    if image_id == 0:
        # cur_pose = np.array([
        #     [1.0, 0.0, 0.0, 0.0],
        #     [0.0, 1.0, 0.0, 0.0],
        #     [0.0, 0.0, 1.0, 0.0],
        #     [0.0, 0.0, 0.0, 1.0]
        # ])  # the first pose is from gt
        res = starting_point_prediction_for_trial(image_data)
        floor[0] = res[0][-1]

        r = np.array([[res[0][12], -res[0][8], res[0][4]], [res[0][11], -res[0][7], res[0][3]], [-res[0][10], res[0][6], -res[0][2]]])
        r = r @ delta @ delta2
        # t = np.zeros((3,))
        t = np.array([res[0][5], res[0][9], res[0][13]])
        # align_transformation = np.eye(4)
        # align_transformation[:3:, :3] = r
        # align_transformation[:3, 3] = t
        for i in range(3):
            for j in range(3):
                align_transformation[i][j] = r[i][j]
        align_transformation[0][3] = t[0]
        align_transformation[1][3] = t[1]
        align_transformation[2][3] = t[2]
        # global_lat[0], global_lon[0] = res[0][1], res[0][0]
    elif (image_id % 50) == 0:
        res = starting_point_prediction_for_trial(image_data)
        if res[1] > 150:
            floor[0] = res[0][-1]
            relocalization = True
        else:
            q1, q2 = vo.get_matches(image_id)
            print(len(q1), len(q2))
            if len(q1) > 5 and len(q2) > 5:
                transf = vo.get_pose(q1, q2)
                if transf is not None:
                    cur_point = np.matmul(cur_pose, np.linalg.inv(transf))
                    for i in range(4):
                        for j in range(4):
                            cur_pose[i][j] = cur_point[i][j]
            relocalization = False
    else:
        q1, q2 = vo.get_matches(image_id)
        print(len(q1), len(q2))
        if len(q1)>5 and len(q2)>5:
            transf = vo.get_pose(q1, q2)
            if transf is not None:
                cur_point = np.matmul(cur_pose, np.linalg.inv(transf))
                for i in range(4):
                    for j in range(4):
                        cur_pose[i][j] = cur_point[i][j]
        # cur_pose = np.matmul(transf, np.linalg.inv(cur_pose))
        # print('transf: ', np.linalg.inv(transf))
    cur_pose2 = cur_pose.copy()
    cur_pose2[:3, 3] *= sc1
    a = cur_pose2.copy()
    cur_pose2 = align_transformation @ a
    euler = angel_trans.rotation2euler(cur_pose2[0:3, 0:3])
    quaternion = angel_trans.euler2quaternion(euler)
    # gt = vo.gt_poses[int(i)]
    if relocalization:
        relocalization = False
        cur_pose2 = np.array([
            [-res[0][12], -res[0][4], res[0][8], res[0][5]],
            [-res[0][11], -res[0][3], res[0][7], res[0][9]],
            [res[0][10], res[0][2], -res[0][6], res[0][13]],
            [0, 0, 0, 1]
        ])
        # cur_pose2[0][3] = res[5]
        # cur_pose2[1][3] = res[9]
        # cur_pose2[2][3] = res[13]
        # euler = angel_trans.rotation2euler(cur_pose2[0:3, 0:3])
        # quaternion = angel_trans.euler2quaternion(euler)
        temp = np.linalg.inv(align_transformation) @ cur_pose2
        temp[:3, 3] /= sc1
        for i in range(4):
            for j in range(4):
                cur_pose[i][j] = temp[i][j]


    lat, lon, alt = pm.enu2geodetic(cur_pose2[0, 3], cur_pose2[1, 3], cur_pose2[2, 3], global_lat[0], global_lon[0],
                                    global_alt[0])
    # poses.append(cur_pose2[0:3, :].reshape(1, 12))
    x, y, z = cur_pose2[0, 3], cur_pose2[1, 3], cur_pose2[2, 3]
    # print(x, y, z)
    # x_true, y_true, z_true = vo.gt_poses[int(i), 0, 3], vo.gt_poses[int(i), 1, 3], vo.gt_poses[int(i), 2, 3]
    # print(x_true, y_true, z_true)
    draw_x, draw_y = int(18 * x) + 400, 400 - int(18 * y)
    # true_x, true_y = int(18 * x_true) + 400, 400 - int(18 * y_true)
    cv2.circle(traj_img, (draw_x, draw_y), 1, (image_id * 255 / 4540, 255 - image_id * 255 / 4540, 0),
               1)  # estimated from green to blue
    # cv2.circle(traj_img, (true_x, true_y), 1, (0, 0, 255), 1)  # groundtruth in red
    # write text on traj_img
    cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
    text = "frame %f : x=%2f y=%2f z=%2f" % (image_id, x, y, z)
    cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
    # show
    cv2.imshow('Trajectory', traj_img)
    print(lat, lon, quaternion, floor[0])
    cv2.waitKey(1)
    proc_time = perf_counter() - ts0
    q_w, q_x, q_y, q_z = quaternion[-1], quaternion[0], quaternion[1], quaternion[2]
    return {"lon": lon, "lat": lat, "floor": floor[0], "proc_time": proc_time, "q_w": q_w, "q_x": q_x, "q_y": q_y,
            "q_z": q_z}

if __name__ == "__main__":
    image_id = 0
    image_data = Image.open('/media/yushichen/LENOVO_USB_HDD/projects/VisualOdometry/ipin_1/image_l/frame_000007.png')
    camloc(image_id, image_data)
