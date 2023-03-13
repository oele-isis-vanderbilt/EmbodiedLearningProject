import pathlib
import os

import pytest
import cv2
import pandas as pd

# Constants
GIT_ROOT = pathlib.Path(os.path.abspath(__file__)).parent.parent
DATA_ROOT = GIT_ROOT / 'data'
LOGS_FILEPATH = DATA_ROOT / 'Oct12TestWithGrads' / 'fish-only-logs.csv'
SCREEN_FILEPATH = DATA_ROOT / 'Oct12TestWithGrads' / 'STEP-01-Run-Fish-screen.mp4'
CAMERA_FILEPATH = DATA_ROOT / 'Oct12TestWithGrads' / 'STEP-01-Run-Fish-front.mp4'

LOG_OFFSET = 60 * 8 + 38.5

@pytest.fixture
def study_data():

    screen = cv2.VideoCapture(str(SCREEN_FILEPATH))
    camera= cv2.VideoCapture(str(CAMERA_FILEPATH))
    logs = pd.read_csv(str(LOGS_FILEPATH))

    # Change the logs to have second-based timestamps
    logs['timestamp'] = pd.to_datetime(logs['timestamp'], format="%H:%M:%S:%f")
    logs['timestamp'] = (logs['timestamp'] - logs['timestamp'][0]).dt.total_seconds()
    logs = logs[logs['timestamp'] >= LOG_OFFSET]
    logs['timestamp'] = logs['timestamp'] - LOG_OFFSET

    return {'screen': screen, 'logs': logs, 'camera': camera}
