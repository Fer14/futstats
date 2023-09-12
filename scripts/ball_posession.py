from tqdm import tqdm
from video_utils import *
from anns import *
import math

# resolves which player is currently in ball possession based on player-ball proximity
def get_player_in_possession(
    player_detections: List[Detection], ball_detections: List[Detection], proximity: int
) -> Optional[Detection]:
    if len(ball_detections) != 1:
        return None
    ball_detection = ball_detections[0]
    for player_detection in player_detections:
        left_distance, right_distance = calculate_feet_distance(
            player_detection.rect.bottom_right,
            player_detection.rect.bottom_left,
            ball_detection.rect.center,
        )
        if left_distance < proximity or right_distance < proximity:
            return player_detection

        # if player_detection.rect.pad(proximity).contains_point(
        #     point=ball_detection.rect.center
        # ):
        #     return player_detection


def euclidean_distance(point1: Point, point2: Point) -> float:
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def calculate_feet_distance(
    left_feet: Point, right_feet: Point, ball: Point
) -> Tuple[float, float]:
    distance_to_center_right = euclidean_distance(right_feet, ball)
    distance_to_center_left = euclidean_distance(left_feet, ball)
    return (distance_to_center_right, distance_to_center_left)


def launch_ball_posession(
    yoloNas: bool,
    model,
    target_video_path: str,
    source_video_path: str,
    player_in_possession_proximity: int = 20,  # distance in pixels from the player's bounding box where we consider the ball is in his possession
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
    player_marker_annotator = MarkerAnntator(color=PLAYER_MARKER_FILL_COLOR)

    # loop over frames
    for frame in tqdm(frame_iterator, total=750, desc="Annotating video"):

        # run detector
        if yoloNas:
            results = list(model.predict(frame, conf=0.25))[0]
            detections = Detection.from_yoloNas(pred=results)
        else:
            results = model(frame, size=1280)
            detections = Detection.from_yolo5(
                pred=results.pred[0].cpu().numpy(), names=model.names
            )

        # postprocess results
        ball_detections = filter_detections_by_class(
            detections=detections, class_name="ball"
        )
        goalkeeper_detections = filter_detections_by_class(
            detections=detections, class_name="goalkeeper"
        )
        player_detections = (
            filter_detections_by_class(detections=detections, class_name="player")
            + goalkeeper_detections
        )
        player_in_possession_detection = get_player_in_possession(
            player_detections=player_detections,
            ball_detections=ball_detections,
            proximity=player_in_possession_proximity,
        )

        # annotate video frame
        annotated_image = frame.copy()
        annotated_image = ball_marker_annotator.annotate(
            image=annotated_image, detections=ball_detections
        )
        annotated_image = player_marker_annotator.annotate(
            image=annotated_image,
            detections=[player_in_possession_detection]
            if player_in_possession_detection
            else [],
        )

        # save video frame
        video_writer.write(annotated_image)

    # close output video
    video_writer.release()
