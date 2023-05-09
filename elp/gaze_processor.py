from typing import Dict
import collections

import imutils
import numpy as np
import cv2
# import tensorflow as tf
# import tensorflow_hub as hub
# from deepface import DeepFace
# from retinaface import RetinaFace
# import mediapipe as mp
from face_detection import RetinaFace

# Reference:
# https://towardsdatascience.com/real-time-head-pose-estimation-in-python-e52db1bc606a

# Constants:
MODEL_POINTS = np.array([
    (150.0, 170.0, -135.0),      # Right eye right corne
    (-150.0, 170.0, -135.0),     # Left eye left corner
    (0.0, 0.0, 0.0),             # Nose tip
    (150.0, -150.0, -125.0),      # Right mouth corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
])

# FW = 300
# FD = 135
# FH = 300
# MODEL_POINTS = np.array([
#     (0.0, 0.0, 0.0),        # Nose tip
#     (-FW/2, FH/4, -FD),     # Left eye left corner
#     (FW/2, FH/4, -FD),      # Right eye right corne
#     (-FW/2, -FH/2, -FD),    # Left Mouth corner
#     (FW/2, -FH/2, -FD)      # Right mouth corner
# ])

# Camera internals
H = 1080
W = 1920
FOCAL_LENGTH= W
center = (W/2, H/2)
camera_matrix = np.array(
    [[FOCAL_LENGTH, 0, center[0]],
    [0, FOCAL_LENGTH, center[1]],
    [0, 0, 1]], dtype = "double"
)
DIST_COEFFS = np.zeros((4,1)) # Assuming no lens distortion

class GazeResult:

    def __init__(self, **kwargs):
        self.container: Dict = kwargs
        self.img_size = 500

    def render(self, frame: np.ndarray):
        
        for face_id, face_data in self.container['results'].items():

            # Draw the rectangle
            rect = face_data['facial_area']
            x1, y1, x2, y2 = rect
            start_point = (int(x1), int(y1))
            end_point = (int(x2), int(y2))
            frame = cv2.rectangle(frame, start_point, end_point, (255,0,0), 2)
            frame = cv2.putText(frame, f"{face_data['score']:.2f}", start_point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

            # Draw the other landmarks
            for lm, color in {'right_eye': (0,0,255), 'left_eye': (0,255,0), 'nose':(255,0,0), 'mouth_right':(255,255,255), 'mouth_left':(255,255,0)}.items():
                raw = face_data['landmarks'][lm]
                frame = cv2.circle(frame, (int(raw[0]), int(raw[1])), 2, color, -1)

            # Draw the gaze vector
            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose
            t_vec, r_vec = face_data['pose']['T'], face_data['pose']["R"]
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), r_vec,t_vec, camera_matrix, DIST_COEFFS)
             
            p1 = ( int(face_data['landmarks']['nose'][0]), int(face_data['landmarks']['nose'][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            cv2.line(frame, p1, p2, (0,255,0), 2)

        return frame

class GazeProcessor():
    
    def __init__(self, start_time, face_confidence: float = 0.8):
        # Load model
        # STEP 2: Create an FaceDetector object.
        self.detector = RetinaFace()
        self.face_confidence = face_confidence

    def step(self, frame: np.ndarray, timestamp: float):

        output = self.detector(frame)

        results = collections.defaultdict(dict)

        for face_id, face_data in enumerate(output):
            bbox, image_points, score = face_data

            # Prun bad face detection
            if score < self.face_confidence:
                continue

            results[face_id]['facial_area'] = bbox
            results[face_id]['score'] = score
            results[face_id]['landmarks'] = {
                'right_eye': image_points[0],
                'left_eye': image_points[1],
                'nose': image_points[2],
                'mouth_right': image_points[3],
                'mouth_left': image_points[4],
            }

            (success, rotation_vector, translation_vector) = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, DIST_COEFFS, flags=cv2.SOLVEPNP_UPNP)

            # Store in face data
            results[face_id]['pose'] = {
                "success": success,
                "R": rotation_vector,
                "T": translation_vector
            }

        return GazeResult(results=results) 
