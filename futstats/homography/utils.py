from futstats.annotations.anns import Detection
import cv2
import numpy as np

POINT2POINT2D = {
    "0": (60, 1300),
    "1": (60, 60),
    "2": (60, 310),
    "3": (1020, 60),
    "4": (1020, 680),
    "5": (1020, 1300),
    "6": (2000, 60),
    "7": (2000, 310),
    "8": (1700, 310),
    "9": (2000, 520),
    "10": (1900, 520),
    "11": (2000, 850),
    "12": (1900, 850),
    "13": (360, 310),
    "14": (2000, 1050),
    "15": (1700, 1050),
    "16": (2000, 1300),
    "17": (360, 550),
    "18": (360, 820),
    "19": (1020, 520),
    "20": (1020, 860),
    "21": (1700, 550),
    "22": (1700, 820),
    "23": (60, 520),
    "24": (160, 520),
    "25": (60, 850),
    "26": (160, 850),
    "27": (60, 1050),
    "28": (360, 1050),
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
