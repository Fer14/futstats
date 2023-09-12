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
)


def launch_all_together(
    yoloNas: bool,
    model,
    target_video_path: str,
    source_video_path: str,
    player_in_possession_proximity: int = 30,
):

    # initiate video writer
    video_config = VideoConfig(fps=30, width=1920, height=1080)
    video_writer = get_video_writer(
        target_video_path=target_video_path, video_config=video_config
    )

    # get fresh video frame generator
    frame_iterator = iter(generate_frames(video_file=source_video_path))

    # initiate annotators
    base_annotator = BaseAnnotator(
        colors=[BALL_COLOR, PLAYER_COLOR, PLAYER_COLOR, REFEREE_COLOR],
        thickness=THICKNESS,
    )

    player_goalkeeper_text_annotator = TextAnnotator(
        PLAYER_COLOR, text_color=Color(255, 255, 255), text_thickness=2
    )
    referee_text_annotator = TextAnnotator(
        REFEREE_COLOR, text_color=Color(0, 0, 0), text_thickness=2
    )

    ball_marker_annotator = MarkerAnntator(color=BALL_MARKER_FILL_COLOR)
    player_in_possession_marker_annotator = MarkerAnntator(
        color=PLAYER_MARKER_FILL_COLOR
    )

    # initiate tracker
    byte_tracker = BYTETracker(BYTETrackerArgs())

    # loop over frames
    for frame in tqdm(frame_iterator, total=750):

        # run detector
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
        referee_detections = filter_detections_by_class(
            detections=detections, class_name="referee"
        )
        goalkeeper_detections = filter_detections_by_class(
            detections=detections, class_name="goalkeeper"
        )
        player_detections = filter_detections_by_class(
            detections=detections, class_name="player"
        )

        player_goalkeeper_detections = player_detections + goalkeeper_detections
        tracked_detections = (
            player_detections + goalkeeper_detections + referee_detections
        )

        # calculate player in possession
        player_in_possession_detection = get_player_in_possession(
            player_detections=player_goalkeeper_detections,
            ball_detections=ball_detections,
            proximity=player_in_possession_proximity,
        )

        # track
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=tracked_detections),
            img_info=frame.shape,
            img_size=frame.shape,
        )
        tracked_detections = match_detections_with_tracks(
            detections=tracked_detections, tracks=tracks
        )

        tracked_referee_detections = filter_detections_by_class(
            detections=tracked_detections, class_name="referee"
        )
        tracked_goalkeeper_detections = filter_detections_by_class(
            detections=tracked_detections, class_name="goalkeeper"
        )
        tracked_player_detections = filter_detections_by_class(
            detections=tracked_detections, class_name="player"
        )

        # annotate video frame
        annotated_image = frame.copy()
        annotated_image = base_annotator.annotate(
            image=annotated_image, detections=tracked_detections
        )

        annotated_image = player_goalkeeper_text_annotator.annotate(
            image=annotated_image,
            detections=tracked_goalkeeper_detections + tracked_player_detections,
        )
        annotated_image = referee_text_annotator.annotate(
            image=annotated_image, detections=tracked_referee_detections
        )

        annotated_image = ball_marker_annotator.annotate(
            image=annotated_image, detections=ball_detections
        )
        annotated_image = player_in_possession_marker_annotator.annotate(
            image=annotated_image,
            detections=[player_in_possession_detection]
            if player_in_possession_detection
            else [],
        )

        # save video frame
        video_writer.write(annotated_image)

    # close output video
    video_writer.release()
