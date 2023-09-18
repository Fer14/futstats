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
from posession_utils import *


def update_stack_detections(
    detections_stack: Dict[int, List[Detection]],
    detections: List[Detection],
    len_detections_stack: int = 3,
) -> Dict[int, List[Detection]]:

    for detection in detections:
        if detection.tracker_id not in detections_stack:
            detections_stack[detection.tracker_id] = []
        if len(detections_stack[detection.tracker_id]) > len_detections_stack:
            detections_stack[detection.tracker_id].pop(0)
        detections_stack[detection.tracker_id].append(detection)
    return detections_stack


def launch_ensemble_team_posession_tracking(
    yoloNas: bool,
    model,
    target_video_path: str,
    source_video_path: str,
    player_in_possession_proximity: int = 45,
    len_detections_stack: int = 5,
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

    player_text_annotator = TextAnnotator(
        PLAYER_COLOR, text_color=Color(255, 255, 255), text_thickness=2
    )

    ball_marker_annotator = MarkerAnntator(color=BALL_MARKER_FILL_COLOR)
    player_in_possession_marker_annotator = MarkerAnntator(
        color=PLAYER_MARKER_FILL_COLOR
    )
    posession_pipeline = PosessionPipeline()
    color_in_posession = None
    team_posession_annotator = PosesionAnntator()
    posession = np.array([50, 50])
    posession_calculator = PosessionCalculator(posession)

    # initiate tracker
    byte_tracker = BYTETracker(BYTETrackerArgs())

    detections_stack = {}

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

        player_detections = filter_detections_by_class(
            detections=detections, class_name="player"
        )

        tracked_detections = player_detections

        # track
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=tracked_detections),
            img_info=frame.shape,
            img_size=frame.shape,
        )
        tracked_detections = match_detections_with_tracks(
            detections=tracked_detections, tracks=tracks
        )

        tracked_player_detections = filter_detections_by_class(
            detections=tracked_detections, class_name="player"
        )

        save_detections(image=frame, detections=tracked_player_detections)

        exit(0)

        # annotate video frame
        annotated_image = frame.copy()

        if iteration == 0:
            team_colors = posession_pipeline.get_team_colors(
                image=annotated_image,
                detections=player_detections,
            )

        tracked_player_detections_colored = (
            posession_pipeline.cluster_detection_ensembled(
                image=annotated_image,
                detections=tracked_player_detections,
                past_detections=detections_stack,
            )
        )

        detections_stack = update_stack_detections(
            detections_stack=detections_stack,
            detections=tracked_player_detections_colored,
            len_detections_stack=len_detections_stack,
        )

        # calculate player in possession
        player_in_possession_detection = get_player_in_possession(
            player_detections=tracked_player_detections_colored,
            ball_detections=ball_detections,
            proximity=player_in_possession_proximity,
        )

        color_in_posession = (
            player_in_possession_detection.color
            if player_in_possession_detection
            else color_in_posession
        )

        posession = posession_calculator.calculate_posession(
            color_in_posession=color_in_posession,
            team_colors=team_colors,
        )

        annotated_image = team_posession_annotator.annotate(
            image=annotated_image,
            team_colors=team_colors,
            posession=posession,
            color_in_posession=color_in_posession,
            player_in_posession=player_in_possession_detection,
        )

        ## Player text and circle annotation
        annotated_image = base_annotator.annotate(
            image=annotated_image, detections=tracked_player_detections_colored
        )
        annotated_image = player_text_annotator.annotate(
            image=annotated_image,
            detections=tracked_player_detections_colored,
        )
        # Ball marker annotation
        annotated_image = ball_marker_annotator.annotate(
            image=annotated_image, detections=ball_detections
        )
        # Player in possession marker annotation
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
