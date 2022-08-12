import os
import numpy as np
import cv2

from lib.visualization import plotting
from lib.visualization.video import play_trip

from tqdm import tqdm


class VisualOdometry():
    def __init__(self, data_dir, method='orb'):
        # intrinsic
        self.K, self.P = self._load_calib(os.path.join(data_dir, 'calib.txt'))
        self.gt_poses = self._load_poses(os.path.join(data_dir,"poses.txt"))
        self.images = self._load_images(os.path.join(data_dir,"image_l"))
        self.orb = cv2.ORB_create(3000)
        self.sift = cv2.xfeatures2d.SIFT_create(3000)
        self.method = method
        #orb
        if self.method == 'orb':
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
        # sift
        elif self.method == 'sift':
            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

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
        # Find the keypoints and descriptors with ORB
        if self.method == 'orb':
            kp1, des1 = self.orb.detectAndCompute(self.images[i - 1], None)
            kp2, des2 = self.orb.detectAndCompute(self.images[i], None)
        elif self.method == 'sift':
            kp1, des1 = self.sift.detectAndCompute(self.images[i - 1], None)
            kp2, des2 = self.sift.detectAndCompute(self.images[i], None)
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

        draw_params = dict(matchColor = -1, # draw matches in green color
                 singlePointColor = None,
                 matchesMask = None, # draw only inliers
                 flags = 2)

        img3 = cv2.drawMatches(self.images[i], kp1, self.images[i-1],kp2, good ,None,**draw_params)
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
    data_dir = 'ipin_3'
    method = 'sift'
    vo = VisualOdometry(data_dir, method=method)
    image_path = os.listdir(os.path.join(data_dir, 'image_l'))


    # play_trip(vo.images)  # Comment out to not play the trip
    gt_path = []
    estimated_path = []
    # for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size+200, 3), dtype=np.uint8)
    half_traj_img_size = int(0.75 * traj_img_size)
    # draw_scale = 1
    draw_scale = 0.5
    for i in tqdm(range(len(image_path))):
        if i == 0:
            cur_pose = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]) # the first pose is from gt
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            print('transf: ', np.linalg.inv(transf))
        # gt_path.append((gt_pose[0, 3], gt_pose[2, 3])) # gt pose
        # print('\ngt: ', (gt_pose[0, 3], gt_pose[2, 3], gt_pose[1,3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3])) # current pose with x, y
        print('predict: x, y, z ',(cur_pose[0, 3], cur_pose[2, 3], cur_pose[1, 3]))
        x, y, z = cur_pose[0, 3], cur_pose[2, 3], cur_pose[1, 3]

        draw_x, draw_y = int(draw_scale * x) + 400, 400 - int(draw_scale * y)
        # true_x, true_y = int(draw_scale*x_true) + half_traj_img_size, half_traj_img_size - int(draw_scale*z_true)
        cv2.circle(traj_img, (draw_x, draw_y), 1, (i * 255 / 4540, 255 - i * 255 / 4540, 0),
                   1)  # estimated from green to blue
        # cv2.circle(traj_img, (true_x, true_y), 1,(0, 0, 255), 1)  # groundtruth in red
        # write text on traj_img
        cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
        text = "frame %f : x=%2f y=%2f z=%2f" % (i, x, y, z)
        cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        # show
        cv2.imshow('Trajectory', traj_img)

    plotting.visualize_paths_without_gt(estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + method + ".html")
    cv2.waitKey(1)

if __name__ == "__main__":
    main()
