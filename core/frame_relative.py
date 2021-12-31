import os
import cv2
import pickle

import torch
import tqdm
import numpy as np
import pydegensac
from core.test_camera import run_multi_stream, image_get, stop_flag
from core.kp_match import rord_matching
from core.kp_extract import init_model, extract


class FrameRelative(object):

    def __init__(self,
                 model_path,
                 max_frame_num,
                 url_list,
                 output_path="tmp/",
                 param_file="homography_param.pkl",
                 image_size=(1280, 960),
                 use_gpu=True,
                 debug=False):
        self.device = torch.device('cuda:0') if use_gpu else torch.device('cpu')
        self.model = init_model(model_path=model_path, device=self.device)
        self.url_list = url_list
        self.max_frame_num = max_frame_num
        self.output_path = output_path
        self.image_size = image_size
        self.param_file = param_file
        self.point_list_a = list()
        self.point_list_b = list()
        self.video_output_list = list()
        self.debug = debug

    def read_from_url_and_save(self):
        """
        multi process read video stream and save to specific path
        :return:
        """
        # prepare video saver and video output path
        os.makedirs(self.output_path, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        video_handler_list = list()
        for i in range(len(self.url_list)):
            output_name = os.path.join(self.output_path,
                                       "output_" + str(i) + ".avi")
            handler = cv2.VideoWriter(output_name, fourcc=fourcc, fps=20,
                                      frameSize=tuple(self.image_size))
            self.video_output_list.append(output_name)
            video_handler_list.append(handler)

        # start process
        queues, processes = run_multi_stream(self.url_list)

        # start read video stream and save video
        frame_count = 0
        while frame_count in tqdm.tqdm(range(self.max_frame_num)):
            frames = image_get(queues, self.url_list)
            for idx, frame in enumerate(frames):
                video_handler_list[idx].write(frame)
            frame_count += 1

        # stop process and release video saver
        stop_flag.value = True
        for handler in video_handler_list:
            handler.release()

        for process in processes:
            process.terminate()
        # stop process
        for process in processes:
            process.join()

    def get_homography_and_mask_all(self):
        """
        get homography and repeat part mask for all video view
        :return:
        """
        os.makedirs(self.output_path, exist_ok=True)
        param_file = os.path.join(self.output_path, self.param_file)

        self.video_output_list.append(self.video_output_list[0])
        pos = 0
        homography_dict = dict()
        while pos < len(self.video_output_list) - 1:
            video_path_a = self.video_output_list[pos]
            video_path_b = self.video_output_list[pos + 1]
            homo, mask_a, mask_b = self.read_video_and_get_homography(video_path_a, video_path_b)
            homography_dict["{}-{}".format(str(pos), str(pos + 1))] = homo
            cv2.imwrite(os.path.join(self.output_path, "mask_{}.png".format(pos)), mask_a)
            cv2.imwrite(os.path.join(self.output_path, "mask_{}.png".format(pos + 1)), mask_b)
            pos += 1

        print(homography_dict)
        with open(param_file, 'wb') as fp:
            pickle.dump(homography_dict, fp)

    def read_video_and_get_homography(self, video_path_a, video_path_b):
        """

        :param video_path_a:
        :param video_path_b:
        :return:
        """
        cap_a = cv2.VideoCapture(video_path_a)
        cap_b = cv2.VideoCapture(video_path_b)

        frame_count = 0
        while frame_count < self.max_frame_num and (cap_a.isOpened() and cap_b.isOpened()):
            ret_a, frame_a = cap_a.read()
            ret_b, frame_b = cap_b.read()
            if ret_a and ret_b:
                self.find_one_frame_match(frame_a, frame_b)
            else:
                break
            frame_count += 1
            print(frame_count)

        homo, inlier_kp_a, inlier_kp_b = self.get_homography_matrix()
        mask_a = self.get_mask_by_point(inlier_kp_a)
        mask_b = self.get_mask_by_point(inlier_kp_b)
        cap_a.release()
        cap_b.release()

        return homo, mask_a, mask_b

    def get_mask_by_point(self, points):
        """
        get binary mask image by draw point's polygon
        :param points: list([x,y])
        :return:
        """
        mask = np.zeros(shape=(self.image_size[::-1]), dtype=np.uint8)
        points = np.asarray([points], dtype=np.int32)
        hull = cv2.convexHull(points)
        cv2.fillPoly(mask, [hull], 255)
        return mask

    def find_one_frame_match(self, frame_a, frame_b):
        feat1 = extract(self.model, frame_a, device=self.device)
        feat2 = extract(self.model, frame_b, device=self.device)

        match_point_a, match_point_b, homo = rord_matching(feat1, feat2)

        if self.debug:
            placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(len(match_point_a))]
            draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(255, 0, 0), flags=0)
            match_cv_point_a = [cv2.KeyPoint(point[0], point[1], 1) for point in match_point_a]
            match_cv_point_b = [cv2.KeyPoint(point[0], point[1], 1) for point in match_point_b]
            image3 = cv2.drawMatches(frame_a, match_cv_point_a, frame_b,
                                     match_cv_point_b, placeholder_matches,
                                     None, **draw_params)
            cv2.imshow("debug", image3)
            cv2.waitKey(0)
        self.point_list_a.extend(match_point_a)
        self.point_list_b.extend(match_point_b)

    def get_homography_matrix(self):
        """
        get homography matrix for image rectifying
        :return:
        """
        assert len(self.point_list_a) > 0 and len(self.point_list_b) > 0
        assert len(self.point_list_a) == len(self.point_list_b)

        nd_points_a = np.asarray(self.point_list_a)
        nd_points_b = np.asarray(self.point_list_b)
        homo_b, inliers = pydegensac.findHomography(nd_points_a, nd_points_b,
                                                    10.0, 0.99, 10000)
        inlier_kp_a = [[point[0], point[1]] for point in nd_points_a[inliers]]
        inlier_kp_b = [[point[0], point[1]] for point in nd_points_b[inliers]]

        return homo_b, inlier_kp_a, inlier_kp_b
