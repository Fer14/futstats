import logging
import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
import cv2
import numpy as np
from futstats.annotations.anns import (
    BallAnntator,
    Detection,
    LandmarkAnntator,
    FieldAnnotator,
    filter_detections_by_class,
)
from futstats.homography.utils import parse_detections, get_perspective_transform
from tqdm import tqdm
from futstats.video.video_utils import VideoConfig, generate_frames, get_video_writer
import segmentation_models as sm
import torch
import tensorflow as tf

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


def launch_ball_path_narya(
    yoloNas: bool,
    model,
    field_model: str,
    field_img_path: str,
    target_video_path: str,
    source_video_path: str,
    target_warped_video_path: str,
    target_ball_track_path: str,
):
    field = cv2.cvtColor(cv2.imread(field_img_path), cv2.COLOR_BGR2RGB)

    with tf.device("/CPU:0"):
        kp_model = KeypointDetectorModel(
            backbone="efficientnetb3",
            num_classes=29,
            input_shape=(320, 320),
        )

    kp_model.load_weights(field_model)

    video_writer_homography = get_video_writer(
        target_video_path=target_warped_video_path,
        video_config=VideoConfig(fps=30, width=field.shape[1], height=field.shape[0]),
    )

    video_writer = get_video_writer(
        target_video_path=target_video_path,
        video_config=VideoConfig(fps=30, width=1920, height=1080),
    )

    # get fresh video frame generator
    frame_iterator = iter(generate_frames(video_file=source_video_path))

    ball_marker_annotator = BallAnntator()
    landmarks_annotator = LandmarkAnntator()
    field_annotator = FieldAnnotator(
        field=field, field_height=320, field_width=320, field_y=50, field_x=1400
    )

    # loop over frames
    for iteration, frame in enumerate(tqdm(frame_iterator, total=750)):
        # run detector
        if yoloNas:
            results = list(model.predict(frame, conf=0.25))[0]
            detections = Detection.from_yoloNas(pred=results)
        else:
            results = model(frame, size=1280)
            detections = Detection.from_yolo5(
                pred=results.pred[0].cpu().numpy(), names=model.names
            )

        ball_detections = filter_detections_by_class(
            detections=detections, class_name="ball"
        )

        with tf.device("/CPU:0"):
            pr_mask = kp_model(frame)
        src_points, dst_points = _points_from_mask(pr_mask[0], treshold=0.25)

        if len(src_points) >= 4:
            curr_pred_homo = get_perspective_transform(src_points, dst_points)
            pred_homo = curr_pred_homo if curr_pred_homo is not None else pred_homo
            field_annotator.update(ball_detections, pred_homo, dsize=(320, 320))

        # annotate video frame
        annotated_image = frame.copy()

        for s, d in zip(src_points, dst_points):
            annotated_image = cv2.circle(
                annotated_image,
                (int(s[0] / 320 * 1920), int(s[1] / 320 * 1080)),
                10,
                (0, 0, 255),
                -1,
            )

        # Ball marker annotation
        annotated_image = ball_marker_annotator.annotate(
            image=annotated_image, detections=ball_detections
        )

        annotated_image = field_annotator.annotate(image=annotated_image)

        # save video frame
        video_writer.write(annotated_image)

        # if iteration % 10 == 0:
        if len(src_points) >= 4 and pred_homo is not None:
            video_writer_homography.write(
                cv2.warpPerspective(
                    cv2.resize(frame, (320, 320)),
                    pred_homo,
                    (320, 320),
                )
            )

    # close output video
    video_writer.release()
    video_writer_homography.release()
    # save field image
    cv2.imwrite(target_ball_track_path, field)


if __name__ == "__main__":
    model = torch.hub.load(
        "ultralytics/yolov5", "custom", "../../models/yoloV5model", device=0
    )
    launch_ball_path_narya(
        yoloNas=False,
        model=model,
        field_model="../homography/narya_test/keypoints/keypoint_detector.h5",
        field_img_path="../homography/narya_test/template.png",
        source_video_path="../../../clips/08fd33_0.mp4",
        target_video_path="../output_video/ball_path_narya/08fd33_0.mp4",
        target_warped_video_path="../output_video/ball_path_narya/08fd33_0_warped.mp4",
        target_ball_track_path="../output_video/ball_path_narya/08fd33_0_ball.png",
    )
