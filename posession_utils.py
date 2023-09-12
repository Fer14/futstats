from typing import Dict, List
from anns import Detection, Color

import numpy as np
from sklearn.cluster import KMeans


class PosessionPipeline:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto")
        self.team_colors = {}

    def trainKmeans(self, colors: List[Color]):
        colors = np.array([c.to_array().reshape(-1, 1) for c in colors])
        colors = colors.reshape(colors.shape[:2])

        self.kmeans.fit(colors)

    def get_team_colors(
        self, image: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        colors = []
        for detection in detections:
            colors.append(detection.get_color(image))
        self.trainKmeans(colors)
        r1, g1, b1 = self.kmeans.cluster_centers_[0]
        r2, g2, b2 = self.kmeans.cluster_centers_[1]
        self.team_colors["1"] = Color(r=r1, g=g1, b=b1)
        self.team_colors["2"] = Color(r=r2, g=g2, b=b2)
        return self.team_colors

    def cluster_color(self, image: np.ndarray, detection: List[Detection]) -> Color:
        if detection == []:
            return None
        if isinstance(detection, list):
            detection = detection[0]
        color = detection.get_color(image)
        r, g, b = color.to_array()
        detection_color = np.array([r, g, b]).reshape(1, 3)
        detection_cluster = self.kmeans.predict(detection_color)
        r1, g1, b1 = self.kmeans.cluster_centers_[detection_cluster[0]]
        return Color(r=r1, g=g1, b=b1)

    def cluster_detection(
        self, image: np.ndarray, detections: List[Detection]
    ) -> List[Detection]:
        for detection in detections:
            color = self.cluster_color(image, detection)
            detection.color = color
        return detections

    def vote_color(self, colors: List[Color]) -> Color:
        ## get the most common color
        color = max(set(colors), key=colors.count)
        return color

    def cluster_detection_ensembled(
        self,
        image: np.ndarray,
        detections: List[Detection],
        past_detections=Dict[int, List[Detection]],
    ) -> List[Detection]:
        for detection in detections:
            colors = []
            colors.append(self.cluster_color(image, detection))
            old_detections = (
                past_detections[detection.tracker_id]
                if detection.tracker_id in past_detections
                else []
            )
            for old_detection in old_detections:
                colors.append(self.cluster_color(image, old_detection))
            color = self.vote_color(colors)
            detection.color = color
        return detections


class PosessionCalculator:
    def __init__(self, posession: np.ndarray):
        self.team1_posession = posession[0]
        self.team2_posession = posession[1]

    def calculate_posession(
        self, color_in_posession: Color, team_colors: Dict[str, Color]
    ) -> np.ndarray:
        """When the the is a color in posession, the posession is calculated
        else the posession is calculated based on the previous posession
        """
        if color_in_posession != None:
            if color_in_posession == team_colors["1"]:
                self.team1_posession += 1
            elif color_in_posession == team_colors["2"]:
                self.team2_posession += 1

        suma = (
            self.team1_posession + self.team2_posession
            if self.team1_posession + self.team2_posession != 0
            else 1
        )

        posession = self.team1_posession / suma, self.team2_posession / suma

        return np.array(posession)
