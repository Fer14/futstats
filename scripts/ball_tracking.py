from tqdm import tqdm
from tqdm import tqdm
from video_utils import *
from anns import *
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from scripts.ball_posession import get_player_in_possession
from scripts.tracking import (
    detections2boxes,
    BYTETrackerArgs,
    match_detections_with_tracks,
    detections2norfairDetection,
)
from posession_utils import *
from norfair import Tracker, Video, draw_tracked_objects


def launch_ball_tracking(
    yoloNas: bool,
    model,
    target_video_path: str,
    source_video_path: str,
):

    # initiate video writer
    video_config = VideoConfig(fps=30, width=1920, height=1080)
    video_writer = get_video_writer(
        target_video_path=target_video_path, video_config=video_config
    )

    # get fresh video frame generator
    frame_iterator = iter(generate_frames(video_file=source_video_path))

    ball_marker_annotator = BallAnntator()
    ball_tracker_annotator = BallTraceAnntator()

    # initiate tracker
    tracker = Tracker(distance_function="euclidean", distance_threshold=20)

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

        tracked_objects = tracker.update(
            detections=detections2norfairDetection(ball_detections)
        )

        # annotate video frame
        annotated_image = frame.copy()

        # Ball marker annotation
        annotated_image = ball_marker_annotator.annotate(
            image=annotated_image, detections=ball_detections
        )

        draw_tracked_objects(annotated_image, tracked_objects)

        # annotated_image = ball_tracker_annotator.annotate(
        #     image=annotated_image, detections=ball_trackings
        # )

        # ball_trackings.append(ball_detections)

        # save video frame
        video_writer.write(annotated_image)

    # close output video
    video_writer.release()
