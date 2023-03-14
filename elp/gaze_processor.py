from typing import Dict

import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from deepface import DeepFace
from retinaface import RetinaFace

# Reference:
# https://towardsdatascience.com/real-time-head-pose-estimation-in-python-e52db1bc606a

# Constants:
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (-150.0, 170.0, -135.0),     # Left eye left corner
    (150.0, 170.0, -135.0),      # Right eye right corne
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

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

            # Draw the other landmarks
            for lm in ['right_eye', 'left_eye', 'nose', 'mouth_right', 'mouth_left']:
                raw = face_data['landmarks'][lm]
                frame = cv2.circle(frame, (int(raw[0]), int(raw[1])), 2, (0,0,255), -1)

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
    
    def __init__(self, start_time):
        # Load model
        ...
        

    def step(self, frame: np.ndarray, timestamp: float):

        # results = DeepFace.extract_faces(frame, detector_backend="retinaface")
        results = RetinaFace.detect_faces(frame)

        for face_id, face_data in results.items():
            image_points = np.array([
                face_data['landmarks']['nose'],
                face_data['landmarks']['left_eye'],
                face_data['landmarks']['right_eye'],
                face_data['landmarks']['mouth_left'],
                face_data['landmarks']['mouth_right'],
            ], dtype="double")

            (success, rotation_vector, translation_vector) = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, DIST_COEFFS, flags=cv2.SOLVEPNP_UPNP)

            # Store in face data
            results[face_id]['pose'] = {
                "success": success,
                "R": rotation_vector,
                "T": translation_vector
            }

        return GazeResult(results=results) 
