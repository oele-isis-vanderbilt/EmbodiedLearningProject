import pathlib
import os
import logging
import pdb
import json

import torch
import pytest
import cv2
import pandas as pd
import numpy as np
import imutils

from l2cs import Pipeline, render

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

    # Change the starting time of the camera
    camera.set(cv2.CAP_PROP_POS_FRAMES, 24*60*5)

    # Determine the FPS and use that to compute a timestamp
    fps = screen.get(cv2.CAP_PROP_FPS)
    length = int(screen.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamp = 0
    result = None
    
    # Video writer to saving the output (demoing)
    writer = cv2.VideoWriter(
        str(GIT_ROOT/'test'/'output'/f"gaze_processing.avi"),
        cv2.VideoWriter_fourcc(*'DIVX'),
        fps=fps,
        # frameSize=[1920, 1080]
        frameSize=[1080, 607]
    )
    results = []

    gaze_pipeline = Pipeline(
        weights=GIT_ROOT / 'models' / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device=torch.device('cpu') # or 'gpu'
    )

    # Continue processing video
    for i in range(length):

        # Compute a timestamp
        timestamp += 1/fps

        # Get the data
        ret, frame = camera.read()

        if not ret:
            break

        frame = imutils.resize(frame, width=1080)

        results = gaze_pipeline.step(frame)
        frame = render(frame, results)

        cv2.imshow('output', frame)
        writer.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    writer.release()

    # with open(GIT_ROOT/'test'/'output'/'gaze.json', 'w') as f:
    #     json.dump(results, f)
