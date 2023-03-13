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
def log_processor(study_data):
    return elp.LogProcessor(
        start_time=study_data['logs'].iloc[0].timestamp,
        corrections={'OFFSET': (-180,-260), 'AFFINE': (2.25,2)}
    )


def test_processing_logs(study_data, log_processor):

    # Extract the study data
    screen = study_data['screen']
    camera = study_data['camera']
    logs = study_data['logs']

    # Determine the FPS and use that to compute a timestamp
    fps = screen.get(cv2.CAP_PROP_FPS)
    length = int(screen.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamp = 0
    result = None
    
    # Video writer to saving the output (demoing)
    writer = cv2.VideoWriter(
        str(GIT_ROOT/'test'/'output'/f"log_processing.avi"),
        cv2.VideoWriter_fourcc(*'DIVX'),
        fps=fps,
        frameSize=[1700, 573]
    )

    # Continue processing video
    for i in range(length):

        # Compute a timestamp
        timestamp += 1/fps

        # Get the data
        ret, frame = screen.read()
        ret, camera_view = camera.read()

        # Crop the frame to only get the play area
        frame = frame[CROP['top']:-CROP['bottom'], CROP['left']:-CROP['right']]
        
        # Get a slice of the event logs
        if timestamp > log_processor.timestamp + STEP_SIZE:
            batch_df = logs[(logs['timestamp'] > log_processor.timestamp) & (logs['timestamp'] < log_processor.timestamp + STEP_SIZE)]
            result = log_processor.step(batch_df, timestamp)

        if result:
            frame = result.render(frame)

        vis = imutils.resize(elp.combine_frames(frame, imutils.resize(camera_view, height=frame.shape[0])), width=1700)


        cv2.imshow('output', vis)
        writer.write(vis)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    writer.release()
