import logging

import cv2
import numpy as np
from annotations.anns import (
    BallAnntator,
    Detection,
    LandmarkAnntator,
    FieldAnnotator,
    BoxAnntator,
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
    tracks2boxes,
)
from yolox.tracker.byte_tracker import BYTETracker, STrack

logging.disable(logging.INFO)


def launch_ball_tracking(
    yoloNas: bool,
    model,
    target_video_path: str,
    source_video_path: str,
):
    video_writer = get_video_writer(
        target_video_path=target_video_path,
        video_config=VideoConfig(fps=30, width=1920, height=1080),
    )

    byte_tracker = BYTETracker(BYTETrackerArgs())

    # get fresh video frame generator
    frame_iterator = iter(generate_frames(video_file=source_video_path))

    box_annotator = BoxAnntator()

    # loop over frames
    for iteration, frame in enumerate(tqdm(frame_iterator, total=750)):
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

        if ball_detections != []:
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections=ball_detections),
                img_info=frame.shape,
                img_size=frame.shape,
            )

        tracked_boxes = tracks2boxes(tracks)

        # tracked_detections = match_detections_with_tracks(
        #     detections=field_detections, tracks=tracks
        # )

        annotated_image = frame.copy()
        annotated_image = box_annotator.annotate(
            image=annotated_image, boxes=tracked_boxes
        )

        # annotated_image = landmarks_annotator.annotate(
        #     image=annotated_image, detections=tracked_detections
        # )

        # save video frame
        video_writer.write(annotated_image)

    # close output video
    video_writer.release()