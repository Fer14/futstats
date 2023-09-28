import sys
from sklearn.cluster import KMeans

sys.path.append("..")
from anns import *


POINT2POINT2D = {
    "1": (60, 60),
    "2": (60, 310),
    "3": (60, 520),
    "4": (60, 850),
    "5": (60, 1050),
    "6": (60, 1300),
    "7": (160, 520),
    "8": (160, 850),
    "9": (360, 310),
    "10": (360, 550),
    "11": (360, 820),
    "12": (360, 1050),
    "13": (1020, 60),
    "14": (1020, 520),
    "15": (1020, 860),
    "16": (1020, 1300),
    "cright": (1190, 680),
    "cleft": (850, 680),
}


def clean_detections(detections: List[Detection]):
    keypoints_found = {}

    centers_found = []

    for detection in detections:

        if detection.class_name == "c":
            centers_found.append(detection)
        else:
            if detection.class_name not in keypoints_found:
                keypoints_found[detection.class_name] = detection
            else:
                if (
                    keypoints_found[detection.class_name].confidence
                    < detection.confidence
                ):
                    keypoints_found[detection.class_name] = detection

    keypoints_found = {}

    centers_found = []

    for detection in detections:
        x2, y2 = detection.rect.bottom_right.int_xy_tuple
        x1, y1 = detection.rect.top_left.int_xy_tuple

        if detection.class_name == "c":
            centers_found.append(detection)
        else:
            if detection.class_name not in keypoints_found:
                keypoints_found[detection.class_name] = detection
            else:
                if (
                    keypoints_found[detection.class_name].confidence
                    < detection.confidence
                ):
                    keypoints_found[detection.class_name] = detection

    center_points = []

    for center in centers_found:
        x2, y2 = center.rect.bottom_right.int_xy_tuple
        x1, y1 = center.rect.top_left.int_xy_tuple

        # get center of x1,y1 x2,y2
        center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        center_points.append(center_point)

    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
    kmeans.fit(center_points)
    sides = {}

    if kmeans.cluster_centers_[0][0] < kmeans.cluster_centers_[1][0]:
        sides[0] = "left"
        sides[1] = "right"
    else:
        sides[0] = "right"
        sides[1] = "left"

    left_centers = []
    right_centers = []

    for center in centers_found:
        x2, y2 = center.rect.bottom_right.int_xy_tuple
        x1, y1 = center.rect.top_left.int_xy_tuple

        center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        if sides[kmeans.predict([center_point])[0]] == "left":
            left_centers.append(center)
        else:
            right_centers.append(center)

    higher_left_conf = -1
    higher_left_conf_center = None

    for left_center in left_centers:
        if left_center.confidence > higher_left_conf:
            higher_left_conf_center = left_center
            higher_left_conf = left_center.confidence

    keypoints_found["cleft"] = higher_left_conf_center

    higher_right_conf = -1
    higher_right_conf_center = None

    for right_center in right_centers:
        if right_center.confidence > higher_right_conf:
            higher_right_conf_center = right_center
            higher_right_conf = right_center.confidence

    keypoints_found["cright"] = higher_right_conf_center

    src_points = []
    dst_points = []

    for class_str, detection in keypoints_found.items():
        x2, y2 = detection.rect.bottom_right.int_xy_tuple
        x1, y1 = detection.rect.top_left.int_xy_tuple
        # get the center of the box
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        if class_str in POINT2POINT2D:
            src_points.append(center)
            dst_points.append(POINT2POINT2D[class_str])

    return src_points, dst_points
