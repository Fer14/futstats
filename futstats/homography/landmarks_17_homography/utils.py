from sklearn.cluster import KMeans

from futstats.annotations.anns import Detection


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


def clean_detections(detections: list[Detection]):
    keypoints_found = {}
    centers_found = []
    pred_detections = detections.copy()

    for detection in pred_detections:
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

    for detection in pred_detections:
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

    # if there are just 2 points and there are closer that a specific distance, just keep the one with more confidence
    if len(centers_found) == 2:
        x2, y2 = centers_found[0].rect.bottom_right.int_xy_tuple
        x1, y1 = centers_found[0].rect.top_left.int_xy_tuple

        x2_2, y2_2 = centers_found[1].rect.bottom_right.int_xy_tuple
        x1_2, y1_2 = centers_found[1].rect.top_left.int_xy_tuple

        center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        center_point_2 = (int((x1_2 + x2_2) / 2), int((y1_2 + y2_2) / 2))

        if (
            abs(center_point[0] - center_point_2[0]) < 50
            and abs(center_point[1] - center_point_2[1]) < 50
        ):
            if centers_found[0].confidence > centers_found[1].confidence:
                centers_found.pop(1)
            else:
                centers_found.pop(0)

        # if centers found is located to the right of the pred_detections with class name 14 and 15, rename the classname to cright

        # if "14" in keypoints_found and "15" in keypoints_found:

        #     if (
        #         centers_found[0].rect.top_left.int_xy_tuple[0]
        #         > keypoints_found["14"].rect.top_left.int_xy_tuple[0]
        #         and centers_found[0].rect.top_left.int_xy_tuple[0]
        #         > keypoints_found["15"].rect.top_left.int_xy_tuple[0]
        #     ):
        #         centers_found[0].class_name = "cright"
        #         keypoints_found["cright"] = centers_found[0]

        #     elif (
        #         centers_found[0].rect.top_left.int_xy_tuple[0]
        #         < keypoints_found["14"].rect.top_left.int_xy_tuple[0]
        #         and centers_found[0].rect.top_left.int_xy_tuple[0]
        #         < keypoints_found["15"].rect.top_left.int_xy_tuple[0]
        #     ):
        #         centers_found[0].class_name = "cleft"
        #         keypoints_found["cleft"] = centers_found[0]

    else:
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

        higher_left_conf_center.class_name = "cleft"
        keypoints_found["cleft"] = higher_left_conf_center

        higher_right_conf = -1
        higher_right_conf_center = None

        for right_center in right_centers:
            if right_center.confidence > higher_right_conf:
                higher_right_conf_center = right_center
                higher_right_conf = right_center.confidence

        higher_right_conf_center.class_name = "cright"
        keypoints_found["cright"] = higher_right_conf_center

    src_points = []
    dst_points = []

    clean_detections = []

    for class_str, detection in keypoints_found.items():
        clean_detections.append(detection)
        x2, y2 = detection.rect.bottom_right.int_xy_tuple
        x1, y1 = detection.rect.top_left.int_xy_tuple
        # get the center of the box
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        src_points.append(center)
        dst_points.append(POINT2POINT2D[class_str])

    return clean_detections, src_points, dst_points
