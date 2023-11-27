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
from tracking.bytrack_utils import (
    BYTETrackerArgs,
    detections2boxes,
    match_detections_with_tracks,
)
from yolox.tracker.byte_tracker import BYTETracker, STrack

logging.disable(logging.INFO)


def launch_field_tracking(
    field_model_path: str,
    field_img_path: str,
    target_video_path: str,
    source_video_path: str,
):
    field = cv2.cvtColor(cv2.imread(field_img_path), cv2.COLOR_BGR2RGB)

    field_model = models.get(
        "yolo_nas_s",
        num_classes=29,
        checkpoint_path=field_model_path,
    )

    video_writer = get_video_writer(
        target_video_path=target_video_path,
        video_config=VideoConfig(fps=30, width=1920, height=1080),
    )

    byte_tracker = BYTETracker(BYTETrackerArgs())

    # get fresh video frame generator
    frame_iterator = iter(generate_frames(video_file=source_video_path))

    landmarks_annotator = LandmarkAnntator()

    # loop over frames
    for iteration, frame in enumerate(tqdm(frame_iterator, total=750)):
        field_results = list(field_model.predict(frame, conf=0.35))[0]
        field_detections = Detection.from_yoloNas(pred=field_results)

        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=field_detections),
            img_info=frame.shape,
            img_size=frame.shape,
        )

        tracked_detections = match_detections_with_tracks(
            detections=field_detections, tracks=tracks
        )

        annotated_image = frame.copy()
        annotated_image = landmarks_annotator.annotate(
            image=annotated_image, detections=tracked_detections
        )

        # save video frame
        video_writer.write(annotated_image)

    # close output video
    video_writer.release()
