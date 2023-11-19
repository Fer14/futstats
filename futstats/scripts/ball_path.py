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
from homography.utils import parse_detections, get_perspective_transform
from super_gradients.training import models
from tqdm import tqdm
from video.video_utils import VideoConfig, generate_frames, get_video_writer

logging.disable(logging.INFO)

CLASSES = {
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
        "yolo_nas_s",
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

        ball_detections = filter_detections_by_class(
            detections=detections, class_name="ball"
        )

        field_results = list(field_model.predict(frame, conf=0.35))[0]
        field_detections = Detection.from_yoloNas(pred=field_results)

        src_points, dst_points = parse_detections(detections=field_detections)

        if len(src_points) >= 4:
            pred_homo = get_perspective_transform(
                np.array(src_points), np.array(dst_points)
            )
            if pred_homo is not None:
                ## sustituir lo de arriba por cv2.findHomography(src, dst, cv2.RANSAC, 5)
                field_annotator.update(ball_detections, pred_homo)

        # annotate video frame
        annotated_image = frame.copy()

        # Ball marker annotation
        annotated_image = ball_marker_annotator.annotate(
            image=annotated_image, detections=ball_detections
        )

        annotated_image = landmarks_annotator.annotate(
            image=annotated_image, detections=field_detections
        )

        annotated_image = field_annotator.annotate(image=annotated_image)

        # save video frame
        video_writer.write(annotated_image)
        if len(src_points) >= 4 and pred_homo is not None:
            video_writer_homography.write(
                cv2.warpPerspective(frame, pred_homo, (field.shape[1], field.shape[0]))
            )

    # close output video
    video_writer.release()
    video_writer_homography.release()
    # save field image
    cv2.imwrite(target_ball_track_path, field)
