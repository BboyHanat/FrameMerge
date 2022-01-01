import cv2
import numpy as np


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


arr = [[5.61237389e+00, -1.74767574e-02, 1.19662641e+03],
       [3.47669615e+00, 2.32467510e+00, -2.32264821e+02],
       [3.66471397e-03, 3.82211492e-04, 1.24032707e+00]]

# a = cv2.KeyPoint((1,2))

homo = np.asarray(arr, dtype=np.float32)
cap = cv2.VideoCapture("rtsp://admin:dahua123456@192.168.99.149:554/cam/realmonitor?channel=1&subtype=0")
cap2 = cv2.VideoCapture("rtsp://admin:dahua123456@192.168.99.150:554/cam/realmonitor?channel=1&subtype=0")

while True:
    ret, image = cap.read()
    ret1, image1 = cap2.read()
    # mask = cv2.imread("tmp/mask_0.png", 0)
    if ret:
        # h, w = image.shape[0: 2]
        # new_w, new_h = compute_dst_size(homo=homo, w=w, h=w)
        # result = cv2.warpPerspective(image, homo, dsize=(new_w, new_h),
        #                              flags=cv2.WARP_FILL_OUTLIERS)
        # result_mask = cv2.warpPerspective(mask, homo, dsize=(new_w, new_h),
        #                                     flags=cv2.WARP_FILL_OUTLIERS)
        #
        # ret, result_mask = cv2.threshold(result_mask, 0, 1, cv2.THRESH_BINARY_INV)
        #
        # result = result * np.expand_dims(result_mask, -1)
        image1 = cv2.flip(image1, 0)
        out = np.hstack([image, image1])
        out = cv2.resize(out, (1280, 480))
        cv2.imshow("src1", out)
        # cv2.imshow("src2", image1)
        # cv2.imshow("dst", result)
        # cv2.imwrite("1.png", result)
        # cv2.imwrite("2.png", image1)
        cv2.waitKey(30)
    else:
        break
