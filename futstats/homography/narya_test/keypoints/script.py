import logging
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
import cv2
import numpy as np

from tqdm import tqdm
from futstats.video.video_utils import VideoConfig, generate_frames, get_video_writer
import segmentation_models as sm


logging.disable(logging.INFO)


def _build_keypoint_preprocessing(input_shape, backbone):
    """Builds the preprocessing function for the Field Keypoint Detector Model."""
    sm_preprocessing = sm.get_preprocessing(backbone)

    def preprocessing(input_img, **kwargs):
        to_normalize = False if np.percentile(input_img, 98) > 1.0 else True

        if len(input_img.shape) == 4:
            print(
                "Only preprocessing single image, we will consider the first one of the batch"
            )
            image = input_img[0] * 255.0 if to_normalize else input_img[0] * 1.0
        else:
            image = input_img * 255.0 if to_normalize else input_img * 1.0

        image = cv2.resize(image, input_shape)
        image = sm_preprocessing(image)
        return image

    return preprocessing


INIT_HOMO_MAPPER = {
    0: [3, 3],
    1: [3, 66],
    2: [51, 65],
    3: [3, 117],
    4: [17, 117],
    5: [3, 203],
    6: [17, 203],
    7: [3, 255],
    8: [51, 254],
    9: [3, 317],
    10: [160, 3],
    11: [160, 160],
    12: [160, 317],
    13: [317, 3],
    14: [317, 66],
    15: [270, 66],
    16: [317, 118],
    17: [304, 118],
    18: [317, 203],
    19: [304, 203],
    20: [317, 255],
    21: [271, 255],
    22: [317, 317],
    23: [51, 128],
    24: [51, 193],
    25: [161, 118],
    26: [161, 203],
    27: [270, 128],
    28: [269, 192],
}


def _get_keypoints_from_mask(mask, treshold=0.9):
    """From a list of mask, compute the mapping of each keypoints to their location

    Arguments:
        mask: np.array of shape (nb_of_mask) x (mask_shape)
        treshold: Treshold of intensity to decide if a pixels is considered or not
    Returns:
        keypoints: Dict, mapping each keypoint id to its location
    Raises:

    """
    keypoints = {}
    indexes = np.argwhere(mask[:, :, :-1] > treshold)
    for indx in indexes:
        id_kp = indx[2]
        if id_kp in keypoints.keys():
            keypoints[id_kp][0].append(indx[0])
            keypoints[id_kp][1].append(indx[1])
        else:
            keypoints[id_kp] = [[indx[0]], [indx[1]]]

    for id_kp in keypoints.keys():
        mean_x = np.mean(np.array(keypoints[id_kp][0]))
        mean_y = np.mean(np.array(keypoints[id_kp][1]))
        keypoints[id_kp] = [mean_y, mean_x]
    return keypoints


def collinear(p0, p1, p2, epsilon=0.001):
    x1, y1 = p1[0] - p0[0], p1[1] - p0[1]
    x2, y2 = p2[0] - p0[0], p2[1] - p0[1]
    return abs(x1 * y2 - x2 * y1) < epsilon


class KeypointDetectorModel:
    """Class for Keras Models to predict the keypoint in an image. These keypoints can then be used to
    compute the homography.

    Arguments:
        backbone: String, the backbone we want to use
        model_choice: The model architecture. ('FPN','Unet','Linknet')
        num_classes: Integer, number of mask to compute (= number of keypoints)
        input_shape: Tuple, shape of the model's input
    Call arguments:
        input_img: a np.array of shape input_shape
    """

    def __init__(
        self,
        backbone="efficientnetb3",
        model_choice="FPN",
        num_classes=29,
        input_shape=(320, 320),
    ):
        self.input_shape = input_shape
        self.classes = [str(i) for i in range(num_classes)] + ["background"]
        self.backbone = backbone

        n_classes = len(self.classes)
        activation = "softmax"

        if model_choice == "FPN":
            self.model = sm.FPN(
                self.backbone,
                classes=n_classes,
                activation=activation,
                input_shape=(input_shape[0], input_shape[1], 3),
                encoder_weights="imagenet",
            )
        else:
            self.model = None
            print("{} is not used yet".format(model_choice))

        self.preprocessing = _build_keypoint_preprocessing(input_shape, backbone)

    def __call__(self, input_img):
        img = self.preprocessing(input_img)
        pr_mask = self.model.predict(np.array([img]))
        return pr_mask

    def load_weights(self, weights_path):
        try:
            self.model.load_weights(weights_path)
            print("Succesfully loaded weights from {}".format(weights_path))
        except:
            orig_weights = "from Imagenet"
            print(
                "Could not load weights from {}, weights will be loaded {}".format(
                    weights_path, orig_weights
                )
            )


import six


def get_perspective_transform_cv(src, dst):
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


def _points_from_mask(mask, treshold=0.9):
    """From a list of mask, compute src and dst points from the image and the 2D view of the image

    Arguments:
        mask: np.array of shape (nb_of_mask) x (mask_shape)
        treshold: Treshold of intensity to decide if a pixels is considered or not
    Returns:
        src_pts, dst_pts: Location of src and dst related points
    Raises:

    """
    list_ids = []
    src_pts, dst_pts = [], []
    available_keypoints = _get_keypoints_from_mask(mask, treshold)

    ## fer
    if 25 in available_keypoints and 26 in available_keypoints:
        p1 = available_keypoints[25]
        p2 = available_keypoints[26]

        midpoint_x = (p1[0] + p2[0]) // 2
        midpoint_y = (p1[1] + p2[1]) // 2

        distance_y = p2[1] - midpoint_y

        right_pointx = (midpoint_x + distance_y) * 1.3
        right_pointy = midpoint_y

        if right_pointx > 0 and right_pointx < 320:
            available_keypoints[29] = [right_pointx, right_pointy]
            INIT_HOMO_MAPPER[29] = [190, 160]

        left_pointx = (midpoint_x - distance_y) * 0.7
        left_pointy = midpoint_y

        if left_pointx > 0 and left_pointx < 320:
            available_keypoints[30] = [left_pointx, left_pointy]
            INIT_HOMO_MAPPER[30] = [130, 160]

        if 30 in INIT_HOMO_MAPPER and 29 in INIT_HOMO_MAPPER:
            if 11 in available_keypoints:
                del available_keypoints[11]
    ###

    for id_kp, v in six.iteritems(available_keypoints):
        src_pts.append(v)
        dst_pts.append(INIT_HOMO_MAPPER[id_kp])
        list_ids.append(id_kp)
    src, dst = np.array(src_pts), np.array(dst_pts)

    ### Final test : return nothing if 3 points are colinear and the src has just 4 points
    test_colinear = False
    if len(src) == 4:
        if (
            collinear(dst_pts[0], dst_pts[1], dst_pts[2])
            or collinear(dst_pts[0], dst_pts[1], dst_pts[3])
            or collinear(dst_pts[1], dst_pts[2], dst_pts[3])
        ):
            test_colinear = True
    src = np.array([]) if test_colinear else src
    dst = np.array([]) if test_colinear else dst

    return src, dst


if __name__ == "__main__":
    field = cv2.cvtColor(cv2.imread("../template.png"), cv2.COLOR_BGR2RGB)

    kp_model = KeypointDetectorModel(
        backbone="efficientnetb3",
        num_classes=29,
        input_shape=(320, 320),
    )

    kp_model.load_weights("./keypoint_detector.h5")

    video_writer_homography = get_video_writer(
        target_video_path="../../../output_video/ball_path_narya/keypoints/0a2d9b_0_warped2.mp4",
        video_config=VideoConfig(fps=30, width=field.shape[1], height=field.shape[0]),
    )

    video_writer_field = get_video_writer(
        target_video_path="../../../output_video/ball_path_narya/keypoints/0a2d9b_0_field2.mp4",
        video_config=VideoConfig(fps=30, width=320, height=320),
    )

    video_writer_template = get_video_writer(
        target_video_path="../../../output_video/ball_path_narya/keypoints/0a2d9b_0_template2.mp4",
        video_config=VideoConfig(fps=30, width=field.shape[1], height=field.shape[0]),
    )

    # get fresh video frame generator
    frame_iterator = iter(
        generate_frames(video_file="../../../../../clips/0a2d9b_0.mp4")
    )

    for iteration, frame in enumerate(tqdm(frame_iterator, total=750)):
        pr_mask = kp_model(frame)
        src_points, dst_points = _points_from_mask(pr_mask[0])
        if len(src_points) > 4:
            pred_homo_curr = get_perspective_transform_cv(src_points, dst_points)
            if pred_homo_curr != []:
                if pred_homo_curr is not None:
                    pred_homo = pred_homo_curr

            warped = cv2.warpPerspective(
                cv2.resize(frame, (320, 320)), pred_homo, dsize=(320, 320)
            )
            video_writer_homography.write(warped)
        img2 = frame.copy()
        img2 = cv2.resize(img2, (320, 320))
        template2 = field.copy()
        for s, d in zip(src_points, dst_points):
            img2 = cv2.circle(img2, (int(s[0]), int(s[1])), 10, (0, 0, 255), -1)
            template2 = cv2.circle(
                template2, (int(d[0]), int(d[1])), 2, (0, 0, 255), -1
            )
        video_writer_field.write(img2)
        video_writer_template.write(template2)


video_writer_homography.release()
video_writer_field.release()
video_writer_template.release()
