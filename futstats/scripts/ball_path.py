import logging

import cv2
import numpy as np
from annotations.anns import (
    BallAnntator,
    Detection,
    LandmarkAnntator,
    FieldAnnotator,
    filter_detections_by_class,
)
from homography.homography_utils import clean_detections
from homography.mask_utils import get_perspective_transform
from super_gradients.training import models
from tqdm import tqdm
from video.video_utils import VideoConfig, generate_frames, get_video_writer

logging.disable(logging.INFO)

CLASSES = [
    "1",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "c",
]
NUM_CLASES = len(CLASSES)


def launch_ball_homography(
    yoloNas: bool,
    model,
    field_model_path: str,
    field_img_path: str,
    target_video_path: str,
    source_video_path: str,
    target_warped_video_path: str,
    target_ball_track_path: str,
):

    field = cv2.cvtColor(cv2.imread(field_img_path), cv2.COLOR_BGR2RGB)

    field_model = models.get(
        "yolo_nas_l",
        num_classes=NUM_CLASES,
        checkpoint_path=field_model_path,
    )

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
        field=field, field_height=273, field_width=410, field_y=1005, field_x=100
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

            # filter detections by class
        ball_detections = filter_detections_by_class(
            detections=detections, class_name="ball"
        )

        field_results = list(field_model.predict(frame, conf=0.2))[0]
        field_detections = Detection.from_yoloNas(pred=field_results)

        clean_field_detections, src_points, dst_points = clean_detections(
            detections=field_detections
        )

        pred_homo = get_perspective_transform(
            np.array(src_points), np.array(dst_points)
        )

        field_annotator.update(ball_detections, pred_homo)

        # annotate video frame
        annotated_image = frame.copy()

        # Ball marker annotation
        annotated_image = ball_marker_annotator.annotate(
            image=annotated_image, detections=ball_detections
        )

        annotated_image = landmarks_annotator.annotate(
            image=annotated_image, detections=clean_field_detections
        )

        annotated_image = field_annotator.annotate(image=annotated_image)

        # save video frame
        video_writer.write(annotated_image)
        video_writer_homography.write(
            cv2.warpPerspective(frame, pred_homo, (field.shape[1], field.shape[0]))
        )

    # close output video
    video_writer.release()
    video_writer_homography.release()
    # save field image
    cv2.imwrite(target_ball_track_path, field)
