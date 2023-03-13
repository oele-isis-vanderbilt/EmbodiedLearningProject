import pathlib
import os
import logging
import pdb

import pytest
import cv2
import pandas as pd
import numpy as np
import imutils

import elp

logger = logging.getLogger('elp')

# Constants
GIT_ROOT = pathlib.Path(os.path.abspath(__file__)).parent.parent

# Test Configuration
STEP_SIZE = 0.5 # second
CROP = {'top': 179, 'left': 580, 'bottom': 73, 'right': 356}

@pytest.fixture
def gaze_processor(study_data):
    return elp.GazeProcessor(
        start_time=study_data['logs'].iloc[0].timestamp,
    )


def test_processing_gaze(study_data, gaze_processor):

    # Extract the study data
    screen = study_data['screen']
    camera = study_data['camera']

    # Determine the FPS and use that to compute a timestamp
    fps = screen.get(cv2.CAP_PROP_FPS)
    length = int(screen.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamp = 0
    result = None
    
    # Video writer to saving the output (demoing)
    # writer = cv2.VideoWriter(
    #     str(GIT_ROOT/'test'/'output'/f"gaze_processing.avi"),
    #     cv2.VideoWriter_fourcc(*'DIVX'),
    #     fps=fps,
    #     frameSize=[1700, 573]
    # )

    # Continue processing video
    for i in range(length):

        # Compute a timestamp
        timestamp += 1/fps

        # Get the data
        ret, frame = camera.read()

        result = gaze_processor.step(frame, timestamp)

        if result:
            frame = result.render(frame)

        cv2.imshow('output', frame)
        # writer.write(vis)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    # writer.release()
