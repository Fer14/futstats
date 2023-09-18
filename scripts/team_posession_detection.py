from tqdm import tqdm
from video_utils import *
from anns import *
from .ball_posession import get_player_in_possession
from posession_utils import *


def launch_team_posession_detection(
    yoloNas: bool,
    model,
    target_video_path: str,
    source_video_path: str,
    player_in_possession_proximity: int = 30,  # distance in pixels from the player's bounding box where we consider the ball is in his possession
):

    # initiate video writer
    video_config = VideoConfig(fps=30, width=1920, height=1080)
    video_writer = get_video_writer(
        target_video_path=target_video_path, video_config=video_config
    )

    # get fresh video frame generator
    frame_iterator = iter(generate_frames(video_file=source_video_path))

    # initiate annotators
    ball_marker_annotator = MarkerAnntator(color=BALL_MARKER_FILL_COLOR)
    player_marker_annotator = RectTeamAnntator()
    player_posession_annotator = MarkerAnntator(color=PLAYER_MARKER_FILL_COLOR)
    team_posession_annotator = PosesionAnntator()
    posession_pipeline = PosessionPipeline()
    posession_calculator = PosessionCalculator(50, 50)
    color_in_posession = None

    # loop over frames
    for iteration, frame in enumerate(
        tqdm(frame_iterator, total=750, desc="Annotating video")
    ):

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

        player_detections = filter_detections_by_class(
            detections=detections, class_name="player"
        )

        player_in_possession_detection = get_player_in_possession(
            player_detections=player_detections,
            ball_detections=ball_detections,
            proximity=player_in_possession_proximity,
        )

        # annotate video frame
        annotated_image = frame.copy()

        if iteration == 0:
            team_colors = posession_pipeline.get_team_colors(
                image=annotated_image,
                detections=player_detections,
            )

        curr_color_in_posession = posession_pipeline.cluster_color(
            image=annotated_image,
            detection=[player_in_possession_detection]
            if player_in_possession_detection
            else [],
        )

        if curr_color_in_posession != None:
            color_in_posession = curr_color_in_posession
        else:
            if color_in_posession != None:
                color_in_posession = color_in_posession

        if color_in_posession != None:
            posession = posession_calculator.calculate_posession(
                color_in_posession=color_in_posession,
                team_colors=team_colors,
            )
        if iteration != 0 and color_in_posession != None:

            annotated_image = team_posession_annotator.annotate(
                image=annotated_image,
                team_colors=team_colors,
                posession=posession,
                color_in_posession=color_in_posession,
            )

        annotated_image = player_marker_annotator.annotate(
            image=annotated_image,
            detections=player_detections,
            colors=posession_pipeline.cluster_anns(
                image=annotated_image,
                detections=player_detections,
            ),
        )

        annotated_image = ball_marker_annotator.annotate(
            image=annotated_image, detections=ball_detections
        )
        annotated_image = player_posession_annotator.annotate(
            image=annotated_image,
            detections=[player_in_possession_detection]
            if player_in_possession_detection
            else [],
        )

        # save video frame
        video_writer.write(annotated_image)

    # close output video
    video_writer.release()
