from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from video_utils import *


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @property
    def int_xy_tuple(self) -> Tuple[int, int]:
        return int(self.x), int(self.y)


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @property
    def min_x(self) -> float:
        return self.x

    @property
    def min_y(self) -> float:
        return self.y

    @property
    def max_x(self) -> float:
        return self.x + self.width

    @property
    def max_y(self) -> float:
        return self.y + self.height

    @property
    def top_left(self) -> Point:
        return Point(x=self.x, y=self.y)

    @property
    def top_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y)

    @property
    def bottom_right(self) -> Point:
        return Point(x=self.x + self.width, y=self.y + self.height)

    @property
    def bottom_left(self) -> Point:
        return Point(x=self.x, y=self.y + self.height)

    @property
    def bottom_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height)

    @property
    def top_center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y)

    @property
    def center(self) -> Point:
        return Point(x=self.x + self.width / 2, y=self.y + self.height / 2)

    def pad(self, padding: float) -> Rect:
        return Rect(
            x=self.x - padding,
            y=self.y - padding,
            width=self.width + 2 * padding,
            height=self.height + 2 * padding,
        )

    def contains_point(self, point: Point) -> bool:
        return self.min_x < point.x < self.max_x and self.min_y < point.y < self.max_y


# detection utilities


@dataclass
class Detection:
    rect: Rect
    class_id: int
    class_name: str
    confidence: float
    tracker_id: Optional[int] = None
    color: Optional[Color] = None

    @classmethod
    def from_yolo5(cls, pred: np.ndarray, names: Dict[int, str]) -> List[Detection]:
        result = []
        for x_min, y_min, x_max, y_max, confidence, class_id in pred:
            class_id = int(class_id)
            result.append(
                Detection(
                    rect=Rect(
                        x=float(x_min),
                        y=float(y_min),
                        width=float(x_max - x_min),
                        height=float(y_max - y_min),
                    ),
                    class_id=class_id,
                    class_name=names[class_id],
                    confidence=float(confidence),
                )
            )
        return result

    @classmethod
    def from_yoloNas(cls, pred: np.ndarray) -> List[Detection]:
        boxes = pred.prediction.bboxes_xyxy
        labels = pred.prediction.labels
        confidence = pred.prediction.confidence
        names = pred.class_names

        result = []
        for box, label, confidence in zip(boxes, labels, confidence):
            result.append(
                Detection(
                    rect=Rect(
                        x=float(box[0]),
                        y=float(box[1]),
                        width=float(box[2] - box[0]),
                        height=float(box[3] - box[1]),
                    ),
                    class_id=int(label),
                    class_name=names[int(label)],
                    confidence=float(confidence),
                )
            )
        return result

    def get_color(self, image: np.ndarray) -> Color:
        x2, y2 = self.rect.bottom_right.int_xy_tuple
        x1, y1 = self.rect.top_left.int_xy_tuple
        crop = image[y1:y2, x1:x2, :]
        h = int(crop.shape[0] / 2)
        w = int(crop.shape[1] / 2)
        center_crop = crop[h - 5 : h + 5, w - 5 : w + 5, :]
        b, g, r = cv2.mean(center_crop)[:3]
        return Color(b=b, g=g, r=r)

    def get_hsv_value(self, image: np.ndarray) -> np.ndarray:
        """Returns the HSV value of an image.

        Args:
        image: The image to get the HSV value of.

        Returns:
        The HSV value of the image.
        """

        # Convert the image to HSV format.

        x2, y2 = self.rect.bottom_right.int_xy_tuple
        x1, y1 = self.rect.top_left.int_xy_tuple
        crop = image[y1:y2, x1:x2, :]

        # redimension the crop
        crop = cv2.resize(crop, (100, 100))

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Get the HSV values of the image.
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # Return the HSV values of the image.
        return np.array([h, s, v]).T


def filter_detections_by_class(
    detections: List[Detection], class_name: str
) -> List[Detection]:
    return [detection for detection in detections if detection.class_name == class_name]


# draw utilities


@dataclass(frozen=True)
class Color:
    r: int
    g: int
    b: int

    @property
    def bgr_tuple(self) -> Tuple[int, int, int]:
        return self.b, self.g, self.r

    @classmethod
    def from_hex_string(cls, hex_string: str) -> Color:
        r, g, b = tuple(int(hex_string[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))
        return Color(r=r, g=g, b=b)

    def to_array(self) -> np.ndarray:
        return np.array([self.r, self.g, self.b])


def draw_circle_in_box(image: np.ndarray, box_coordinates: np.ndarray, color: Color):

    center_x = int((box_coordinates[0] + box_coordinates[2]) / 2)
    center_y = int((box_coordinates[1] + box_coordinates[3]) / 2)

    # Get the radius of the circle.
    radius = int((box_coordinates[2] - box_coordinates[0]) / 2)

    # Draw the circle.
    cv2.circle(image, (center_x, center_y), radius, color.bgr_tuple, 4)

    return image


def draw_circle(image: np.ndarray, box_coordinates: np.ndarray, color: Color):

    center_x = int((box_coordinates[0] + box_coordinates[2]) / 2)
    center_y = int((box_coordinates[1] + box_coordinates[3]) / 2)

    # Draw the circle.
    cv2.circle(image, (center_x, center_y), 1, color.bgr_tuple, 1)

    return image


def draw_rect(
    image: np.ndarray, rect: Rect, color: Color, thickness: int = 2
) -> np.ndarray:
    cv2.rectangle(
        image,
        rect.top_left.int_xy_tuple,
        rect.bottom_right.int_xy_tuple,
        color.bgr_tuple,
        thickness,
    )
    return image


def draw_filled_rect(image: np.ndarray, rect: Rect, color: Color) -> np.ndarray:
    cv2.rectangle(
        image,
        rect.top_left.int_xy_tuple,
        rect.bottom_right.int_xy_tuple,
        color.bgr_tuple,
        -1,
    )
    return image


def draw_polygon(
    image: np.ndarray, countour: np.ndarray, color: Color, thickness: int = 2
) -> np.ndarray:
    cv2.drawContours(image, [countour], 0, color.bgr_tuple, thickness)
    return image


def draw_filled_polygon(
    image: np.ndarray, countour: np.ndarray, color: Color
) -> np.ndarray:
    cv2.drawContours(image, [countour], 0, color.bgr_tuple, -1)
    return image


def draw_text(
    image: np.ndarray, anchor: Point, text: str, color: Color, thickness: int = 2
) -> np.ndarray:
    cv2.putText(
        image,
        text,
        anchor.int_xy_tuple,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color.bgr_tuple,
        thickness,
        2,
        False,
    )
    return image


def draw_ellipse(
    image: np.ndarray, rect: Rect, color: Color, thickness: int = 2
) -> np.ndarray:
    cv2.ellipse(
        image,
        center=rect.bottom_center.int_xy_tuple,
        axes=(int(rect.width), int(0.35 * rect.width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color.bgr_tuple,
        thickness=thickness,
        lineType=cv2.LINE_4,
    )
    return image


# base annotator


@dataclass
class BaseAnnotator:
    colors: List[Color]
    thickness: int

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            annotated_image = draw_ellipse(
                image=image,
                rect=detection.rect,
                color=detection.color
                if detection.color
                else self.colors[detection.class_id],
                thickness=self.thickness,
            )
        return annotated_image


BALL_COLOR_HEX = "#FFFFFF"
BALL_COLOR = Color.from_hex_string(BALL_COLOR_HEX)

# red
GOALKEEPER_COLOR_HEX = "#850101"
GOALKEEPER_COLOR = Color.from_hex_string(GOALKEEPER_COLOR_HEX)

# green
PLAYER_COLOR_HEX = "#00D4BB"
PLAYER_COLOR = Color.from_hex_string(PLAYER_COLOR_HEX)

# yellow
REFEREE_COLOR_HEX = "#FFFF00"
REFEREE_COLOR = Color.from_hex_string(REFEREE_COLOR_HEX)

COLORS = [BALL_COLOR, GOALKEEPER_COLOR, PLAYER_COLOR, REFEREE_COLOR]
THICKNESS = 4

MARKER_CONTOUR_COLOR_HEX = "000000"
MARKER_CONTOUR_COLOR = Color.from_hex_string(MARKER_CONTOUR_COLOR_HEX)

# red
PLAYER_MARKER_FILL_COLOR_HEX = "FF0000"
PLAYER_MARKER_FILL_COLOR = Color.from_hex_string(PLAYER_MARKER_FILL_COLOR_HEX)

# green
BALL_MERKER_FILL_COLOR_HEX = "00FF00"
BALL_MARKER_FILL_COLOR = Color.from_hex_string(BALL_MERKER_FILL_COLOR_HEX)

MARKER_CONTOUR_THICKNESS = 2
MARKER_WIDTH = 20
MARKER_HEIGHT = 20
MARKER_MARGIN = 10

# distance in pixels from the player's bounding box where we consider the ball is in his possession


# calculates coordinates of possession marker
def calculate_marker(anchor: Point) -> np.ndarray:
    x, y = anchor.int_xy_tuple
    return np.array(
        [
            [x - MARKER_WIDTH // 2, y - MARKER_HEIGHT - MARKER_MARGIN],
            [x, y - MARKER_MARGIN],
            [x + MARKER_WIDTH // 2, y - MARKER_HEIGHT - MARKER_MARGIN],
        ]
    )


# draw single possession marker
def draw_marker(image: np.ndarray, anchor: Point, color: Color) -> np.ndarray:
    possession_marker_countour = calculate_marker(anchor=anchor)
    image = draw_filled_polygon(
        image=image, countour=possession_marker_countour, color=color
    )
    image = draw_polygon(
        image=image,
        countour=possession_marker_countour,
        color=MARKER_CONTOUR_COLOR,
        thickness=MARKER_CONTOUR_THICKNESS,
    )
    return image


@dataclass
class MarkerAnntator:

    color: Color

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            annotated_image = draw_marker(
                image=image, anchor=detection.rect.top_center, color=self.color
            )
        return annotated_image


@dataclass
class LandmarkAnntator:
    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            rect = detection.rect
            annotated_image = draw_rect(image, detection.rect, Color(255, 0, 0))
        return annotated_image


@dataclass
class BallAnntator:
    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            rect = detection.rect
            x2, y2 = rect.bottom_right.int_xy_tuple
            x1, y1 = rect.top_left.int_xy_tuple
            annotated_image = draw_circle_in_box(
                image, [x1, y1, x2, y2], Color.from_hex_string("#FFFFFF")
            )
        return annotated_image


@dataclass
class BallTraceAnntator:
    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            for inner_detection in detection:
                rect = inner_detection.rect
                x2, y2 = rect.bottom_right.int_xy_tuple
                x1, y1 = rect.top_left.int_xy_tuple
                annotated_image = draw_circle(
                    image, [x1, y1, x2, y2], Color.from_hex_string("#FFFFFF")
                )
        return annotated_image


@dataclass
class MarkerPosessionPlayerAnntator:
    def annotate(
        self, image: np.ndarray, detections: List[Detection], color: Color
    ) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            annotated_image = draw_marker(
                image=image, anchor=detection.rect.top_center, color=color
            )
        return annotated_image


@dataclass
class MarkerTeamAnntator:
    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            color = detection.get_color(image=annotated_image)
            annotated_image = draw_marker(
                image=image, anchor=detection.rect.top_center, color=color
            )
        return annotated_image


@dataclass
class RectTeamAnntator:
    def annotate(
        self,
        image: np.ndarray,
        detections: List[Detection],
        colors: List[Color],
    ) -> np.ndarray:
        annotated_image = image.copy()
        for color, detection in zip(colors, detections):
            # color = detection.get_color(image=annotated_image)
            annotated_image = draw_rect(
                image=annotated_image, rect=detection.rect, color=color, thickness=3
            )
        return annotated_image


@dataclass
class PosesionAnntator:
    def annotate(
        self,
        image: np.ndarray,
        team_colors: np.ndarray,
        posession: np.ndarray,
        color_in_posession: Color,
        player_in_posession: Detection,
    ) -> np.ndarray:

        annotated_image = image.copy()
        team1 = posession[0]
        team2 = posession[1]

        team1_width = team1 * 400 / (team1 + team2) if team1 + team2 > 0 else 0
        team2_width = team2 * 400 / (team1 + team2) if team1 + team2 > 0 else 0

        team1_score = int(team1 * 100)
        team2_score = int(team2 * 100)

        colors_team = {v: k for k, v in team_colors.items()}
        team_in_possesion = (
            colors_team[color_in_posession] if color_in_posession != None else None
        )

        annotated_image = draw_filled_rect(
            image=annotated_image,
            rect=Rect(
                x=20,
                y=15,
                width=400,
                height=50,
            ),
            color=Color(r=0, g=0, b=0),
        )
        annotated_image = draw_text(
            image=annotated_image,
            anchor=Point(x=30, y=35),
            text="Team:",
            color=Color(r=255, g=255, b=255),
        )
        annotated_image = (
            draw_text(
                image=annotated_image,
                anchor=Point(x=100, y=35),
                text=f"Team {team_in_possesion}",
                color=color_in_posession,
            )
            if color_in_posession != None
            else annotated_image
        )
        annotated_image = draw_text(
            image=annotated_image,
            anchor=Point(x=230, y=35),
            text="Player:",
            color=Color(r=255, g=255, b=255),
        )
        annotated_image = (
            draw_text(
                image=annotated_image,
                anchor=Point(x=310, y=35),
                text=f"Player {player_in_posession.tracker_id}",
                color=color_in_posession,
            )
            if player_in_posession != None and player_in_posession.tracker_id != None
            else annotated_image
        )
        annotated_image = draw_filled_rect(
            image=annotated_image,
            rect=Rect(
                x=20,
                y=40,
                width=team1_width,
                height=50,
            ),
            color=team_colors["1"],
        )
        annotated_image = draw_text(
            image=annotated_image,
            anchor=Point(x=(20 + team1_width) / 2, y=70),
            text=str(team1_score),
            color=Color(r=0, g=0, b=0),
        )
        annotated_image = draw_filled_rect(
            image=annotated_image,
            rect=Rect(
                x=20 + team1_width,
                y=40,
                width=team2_width,
                height=50,
            ),
            color=team_colors["2"],
        )
        annotated_image = draw_text(
            image=annotated_image,
            anchor=Point(x=(400 - team2_width) + team2_width / 2, y=70),
            text=str(team2_score),
            color=Color(r=0, g=0, b=0),
        )
        return annotated_image


# text annotator to display tracker_id
@dataclass
class TextAnnotator:
    background_color: Color
    text_color: Color
    text_thickness: int

    def annotate(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        annotated_image = image.copy()
        for detection in detections:
            # if tracker_id is not assigned skip annotation
            if detection.tracker_id is None:
                continue

            # calculate text dimensions
            size, _ = cv2.getTextSize(
                str(detection.tracker_id),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                thickness=self.text_thickness,
            )
            width, height = size

            # calculate text background position
            center_x, center_y = detection.rect.bottom_center.int_xy_tuple
            x = center_x - width // 2
            y = center_y - height // 2 + 10

            # draw background
            annotated_image = draw_filled_rect(
                image=annotated_image,
                rect=Rect(x=x, y=y, width=width, height=height).pad(padding=5),
                color=detection.color
                if detection.color != None
                else self.background_color,
            )

            # draw text
            annotated_image = draw_text(
                image=annotated_image,
                anchor=Point(x=x, y=y + height),
                text=str(detection.tracker_id),
                color=self.text_color,
                thickness=self.text_thickness,
            )
        return annotated_image
