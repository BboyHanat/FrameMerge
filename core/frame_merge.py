import os
import cv2
import pickle

import torch
import tqdm
import numpy as np
import pydegensac
from core.kp_match import rord_matching


class FrameMerge(object):

    def __init__(self,
                 max_frame_num,
                 url_list,
                 output_path="tmp/",
                 param_file="homography_param.pkl",
                 image_size=(1280, 960),
                 debug=False):

        self.url_list = url_list
        self.max_frame_num = max_frame_num
        self.output_path = output_path
        self.image_size = image_size
        self.param_file = param_file
        self.point_list_a = list()
        self.point_list_b = list()
        self.video_output_list = list()

    def get_homography_and_mask_all(self, video_output_list):
        """
        get homography and repeat part mask for all video view
        :return:
        """
        os.makedirs(self.output_path, exist_ok=True)
        param_file = os.path.join(self.output_path, self.param_file)

        pos = 0
        homography_dict = dict()
        while pos < len(video_output_list) - 1:
            video_path_a = video_output_list[pos]
            video_path_b = video_output_list[pos + 1]
            homo, mask_a, mask_b = self.read_video_and_get_homography(video_path_a, video_path_b)
            cv2.imwrite(os.path.join(self.output_path, "mask_{}.png".format(pos)), mask_a)
            cv2.imwrite(os.path.join(self.output_path, "mask_{}.png".format(pos + 1)), mask_b)
            merged_video_output = os.path.join(self.output_path, "merge-{}-{}.avi".format(pos, pos + 1))
            self.save_merged_video(video_path_a, video_path_b, homo,
                                   mask_a, mask_b, merged_video_output)
            homography_dict["{}-{}".format(str(pos), str(pos + 1))] = homo

            pos += 1

        print(homography_dict)
        with open(param_file, 'wb') as fp:
            pickle.dump(homography_dict, fp)

    def save_merged_video(self, video_path_a, video_path_b, homo_b, mask_a, mask_b, merged_video_output):
        cap_a = cv2.VideoCapture(video_path_a)
        cap_b = cv2.VideoCapture(video_path_b)

        new_w, new_h = self.compute_dst_size(homo_b, self.image_size[0], self.image_size[1])
        homo_b[0, 2] = max(homo_b[0, 2], 0)
        homo_b[1, 2] = max(homo_b[1, 2], 0)

        mask_b = self.perspective_transform(mask_b, homo_b, new_w, new_h)

        center_x_a, center_y_a = self.find_mask_center(mask_a)
        center_x_b, center_y_b = self.find_mask_center(mask_b)
        cv2.circle(mask_a, (center_x_a, center_y_a), radius=15, color=120, thickness=5)
        cv2.circle(mask_b, (center_x_b, center_y_b), radius=15, color=120, thickness=5)
        cv2.imwrite("debug1.png", mask_a)
        cv2.imwrite("debug2.png", mask_b)
        roi_x_min = center_x_b - center_x_a
        roi_y_min = center_y_b - center_y_a
        roi_x_max = center_x_b + (self.image_size[0] - center_x_a)
        roi_y_max = center_y_b + (self.image_size[1] - center_y_a)

        # compute padding and coord after shift
        pad_l = np.abs(min(0, roi_x_min))
        pad_t = np.abs(min(0, roi_y_min))
        roi_x_max = roi_x_max + pad_l
        roi_y_max = roi_y_max + pad_t
        pad_r = max(0, roi_x_max - new_w)
        pad_b = max(0, roi_y_max - new_h)
        roi_x_min = roi_x_min + pad_l
        roi_y_min = roi_y_min + pad_t

        frame_w = pad_l + new_w + pad_r
        frame_h = pad_t + new_h + pad_b
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
        handler = cv2.VideoWriter(merged_video_output, fourcc=fourcc, fps=20,
                                  frameSize=(frame_w, frame_h))
        frame_count = 0
        while frame_count < self.max_frame_num and (cap_a.isOpened() and cap_b.isOpened()):
            ret_a, frame_a = cap_a.read()
            ret_b, frame_b = cap_b.read()
            if ret_a and ret_b:
                frame_b = self.perspective_transform(frame_b, homo_b, new_w, new_h)
                frame_b = np.pad(frame_b, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                                 'constant', constant_values=(0, 0))
                cv2.imwrite("test-{}.png".format(str(frame_count)), frame_b)
                frame_b[roi_y_min: roi_y_max, roi_x_min: roi_x_max, :] = frame_a
                cv2.imshow("test", frame_b)
                cv2.waitKey(0)
                handler.write(frame_b)
                frame_count += 1
            else:
                break
        handler.release()

    @staticmethod
    def perspective_transform(image, homo, o_w, o_h):
        result_mask = cv2.warpPerspective(image, homo, dsize=(o_w, o_h),
                                          flags=cv2.WARP_FILL_OUTLIERS)
        return result_mask

    @staticmethod
    def compute_dst_size(homo, w, h):
        pts = np.asarray([[0, 0], [0, w], [h, w], [h, 0]],
                         dtype=np.float32).reshape(-1, 1, 2)
        pts_p = cv2.perspectiveTransform(np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2), homo)
        y_min = min(pts_p[:, 0, 0].min(), 0)
        x_min = min(pts_p[:, 0, 1].min(), 0)
        y_max = max(pts_p[:, 0, 0].max(), h)
        x_max = max(pts_p[:, 0, 1].max(), w)
        new_w = int(x_max - x_min)
        new_h = int(y_max - y_min)
        return new_w, new_h

    @staticmethod
    def find_mask_center(mask):
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(20, 20))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, hierarchy = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        max_cont = list()
        max_area = 0.0
        for cont in contours:
            area = cv2.contourArea(cont)
            if area > max_area:
                max_area = area
                max_cont.append(cont)
        m = cv2.moments(max_cont[-1])
        c_x = int(m["m10"] / m["m00"])
        c_y = int(m["m01"] / m["m00"])

        return c_x, c_y

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
                self.find_one_frame_match_by_hand(frame_a, frame_b)
            else:
                break
            frame_count += 1
            print(frame_count)

        p_matrix = self.get_homography_matrix()
        mask_a = self.get_mask_by_point(self.point_list_a)
        mask_b = self.get_mask_by_point(self.point_list_b)
        cap_a.release()
        cap_b.release()

        return p_matrix, mask_a, mask_b

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

    def find_one_frame_match_by_hand(self, frame_a, frame_b):
        frame_merge = np.hstack([frame_a, frame_b])
        h, w = frame_merge.shape[0:2]
        show_w = w // 2
        show_h = h // 2
        frame_merge = cv2.resize(frame_merge, (show_w, show_h))
        frame_a_point = list()
        frame_b_point = list()
        pt = list()

        def on_mouse(event, x, y):  # 标准鼠标交互函数

            if event == cv2.EVENT_MOUSEMOVE:
                if pt:
                    del pt[0]
                pt.append((x, y))

            if event == cv2.EVENT_LBUTTONUP:  # 当鼠标移动时
                if x > show_w / 2:
                    frame_b_point.append((x, y))
                else:
                    frame_a_point.append((x, y))

        cv2.namedWindow("get_kp")
        cv2.setMouseCallback("get_kp", on_mouse)
        while True:

            frame_merge_cp = cv2.copyTo(frame_merge, mask=None)
            if len(frame_a_point) - 1 == len(frame_b_point):
                pt1 = tuple(frame_a_point[-1])
                if pt:
                    cv2.line(frame_merge_cp, pt1, pt[0], color=(255, 0, 0), thickness=2)
            elif len(frame_a_point) == len(frame_b_point) - 1:
                pt1 = tuple(frame_b_point[-1])
                if pt:
                    cv2.line(frame_merge_cp, pt1, pt[0], color=(255, 0, 0), thickness=2)
            if len(frame_a_point) == len(frame_b_point) == 5:
                break
            cv2.imshow("get_kp", frame_merge_cp)
            cv2.waitKey(30)

        self.point_list_a.extend(frame_a_point)
        self.point_list_b.extend(frame_b_point)

    def get_homography_matrix(self):
        """
        get perspective matrix for image rectifying
        :return:
        """
        assert len(self.point_list_a) > 0 and len(self.point_list_b) > 0
        assert len(self.point_list_a) == len(self.point_list_b)

        nd_points_a = np.asarray(self.point_list_a)
        nd_points_b = np.asarray(self.point_list_b)
        p_matrix = cv2.getPerspectiveTransform(nd_points_b, nd_points_a)

        return p_matrix
