import pathlib
import os
import logging

import pytest
import cv2
import pandas as pd

import elp

logger = logging.getLogger('elp')

# Constants
GIT_ROOT = pathlib.Path(os.path.abspath(__file__)).parent.parent
DATA_ROOT = GIT_ROOT / 'data'
LOGS_FILEPATH = DATA_ROOT / 'Oct12TestWithGrads' / 'fish-only-logs.csv'
SCREEN_FILEPATH = DATA_ROOT / 'Oct12TestWithGrads' / 'STEP-01-Run-Fish-screen.mp4'

# Test Configuration
STEP_SIZE = 5

# Basic asserts
assert LOGS_FILEPATH.exists()
assert SCREEN_FILEPATH.exists()

@pytest.fixture
def study_data():

    cap = cv2.VideoCapture(str(SCREEN_FILEPATH))
    logs = pd.read_csv(str(LOGS_FILEPATH))

    return {'screen': cap, 'logs': logs}


@pytest.fixture
def log_processor(study_data):
    return elp.LogProcessor(
        start_time=study_data['logs'].iloc[0].timestamp,
    )


def test_processing_logs(study_data, log_processor):

    # for i, row in study_data['logs'].iterrows():
    for i in range(0, len(study_data['logs']), STEP_SIZE):
        row = study_data['logs'].iloc[i]
        result = log_processor.step(row)
        output = result.render()

        cv2.imshow("output", output)
        cv2.waitKey(1)
