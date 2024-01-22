from futstats.annotations.anns import Detection
import cv2
import numpy as np

# POINT2POINT2D = {
#     "0": (1020, 680),
#     "1": (2000, 60),
#     "2": (1700, 550),
#     "3": (1700, 820),
#     "4": (1700, 1050),
#     "5": (1020, 60),
#     "6": (1020, 520),
#     "7": (1020, 860),
#     "8": (1020, 1300),
#     "9": (360, 310),
#     "10": (360, 550),
#     "11": (360, 820),
#     "12": (2000, 310),
#     "13": (360, 1050),
#     "14": (160, 520),
#     "15": (160, 850),
#     "16": (60, 60),
#     "17": (60, 310),
#     "18": (60, 520),
#     "19": (60, 850),
#     "20": (60, 1050),
#     "21": (60, 1300),
#     "22": (2000, 520),
#     "23": (2000, 850),
#     "24": (2000, 1050),
#     "25": (2000, 1300),
#     "26": (1900, 520),
#     "27": (1900, 850),
#     "28": (1700, 310),
# }


# ckpt_best_dic

POINT2POINT2D = {
    "0": (2000, 60),
    "1": (1700, 550),
    "2": (1700, 820),
    "3": (1700, 1050),
    "4": (1020, 60),
    "5": (1020, 520),
    "6": (1020, 860),
    "7": (1020, 1300),
    "8": (360, 310),
    "9": (360, 550),
    "10": (360, 820),
    "11": (2000, 310),
    "12": (360, 1050),
    "13": (160, 520),
    "14": (160, 850),
    "15": (60, 60),
    "16": (60, 310),
    "17": (60, 520),
    "18": (60, 850),
    "19": (60, 1050),
    "20": (60, 1300),
    "21": (2000, 520),
    "22": (2000, 850),
    "23": (2000, 1050),
    "24": (2000, 1300),
    "25": (1900, 520),
    "26": (1900, 850),
    "27": (1700, 310),
    "28": (1020, 680),
}


def parse_detections(detections: list[Detection]):
    src_points = []
    dst_points = []

    for detection in detections:
        x2, y2 = detection.rect.bottom_right.int_xy_tuple
        x1, y1 = detection.rect.top_left.int_xy_tuple
        # get the center of the box
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        src_points.append(center)
        dst_points.append(POINT2POINT2D[str(detection.class_id)])

    return src_points, dst_points


def get_perspective_transform(src, dst):
    """Get the homography matrix between src and dst
    Arguments:
        src: np.array of shape (B,X,2) or (X,2), the X>3 original points per image
        dst: np.array of shape (B,X,2) or (X,2), the X>3 corresponding points per image
    Returns:
        M: np.array of shape (B,3,3) or (3,3), each homography per image
    Raises:
    """
    if len(src.shape) == 2:
        M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5)
    else:
        M = []
        for src_, dst_ in zip(src, dst):
            M.append(cv2.findHomography(src_, dst_, cv2.RANSAC, 5)[0])
        M = np.array(M)
    return M
