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
from dataclasses import asdict

from l2cs import Pipeline, render

import elp

logger = logging.getLogger('elp')

# Constants
GIT_ROOT = pathlib.Path(os.path.abspath(__file__)).parent.parent

# Test Configuration
STEP_SIZE = 0.5 # second
CROP = {'top': 179, 'left': 580, 'bottom': 73, 'right': 356}


def test_processing_gaze(study_data):

    # Extract the study data
    screen = study_data['screen']
    camera = study_data['camera']

    # Change the starting time of the camera
    # camera.set(cv2.CAP_PROP_POS_FRAMES, 24*60*4)

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

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    gaze_pipeline = Pipeline(
        weights=GIT_ROOT / 'models' / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device=device # or 'gpu'
    )
    
    # Write the game state
    gaze_file = GIT_ROOT/'test'/'output'/"gaze_logs.csv"
    if gaze_file.exists():
        os.remove(gaze_file)

    # Continue processing video
    for i in range(length):
    # for i in range(5):

        # Compute a timestamp
        timestamp += 1/fps

        # Get the data
        ret, frame = camera.read()

        if not ret:
            break

        frame = imutils.resize(frame, width=1080)

        results = gaze_pipeline.step(frame)
        frame = render(frame, results)

        gaze_df = pd.Series(asdict(results)).to_frame().T
        
        gaze_df.to_csv(
            str(gaze_file),
            mode='a',
            header=not gaze_file.exists(),
            index=False
        )

        cv2.imshow('output', frame)
        writer.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    writer.release()

    # with open(GIT_ROOT/'test'/'output'/'gaze.json', 'w') as f:
    #     json.dump(results, f)
