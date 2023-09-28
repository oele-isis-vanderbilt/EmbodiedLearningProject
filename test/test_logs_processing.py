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

    # Write the game state
    game_state_file = GIT_ROOT/'test'/'output'/"log_game_state.csv"
    prev_game_state = {}
    # if game_state_file.exists():
    #     os.remove(game_state_file)

    # Continue processing video
    for i in range(length):
    # for i in range(100):

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
            state = result.container['id_records']
            game_state_df = pd.Series({'frame_id': i ,'state': state, 'change': state != prev_game_state}).to_frame().T
            prev_game_state = result.container['id_records']
        else:
            game_state_df = pd.Series({'frame_id': i ,'state': prev_game_state, 'change': False}).to_frame().T

        game_state_df.to_csv(
            str(game_state_file),
            mode='a',
            header=not game_state_file.exists(),
            index=False
        )

        vis = imutils.resize(elp.combine_frames(frame, imutils.resize(camera_view, height=frame.shape[0])), width=1700)


        cv2.imshow('output', vis)
        writer.write(vis)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    writer.release()

def test_visualize_logs(study_data, log_processor):

    # Extract the study data
    screen = study_data['screen']
    camera = study_data['camera']

    # Determine the FPS and use that to compute a timestamp
    fps = screen.get(cv2.CAP_PROP_FPS)
    length = int(screen.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamp = 0
    result = None

    # Write the game state
    game_state_file = GIT_ROOT/'data'/'Oct12TestWithGrads'/"log_game_state.csv"
    game_state = pd.read_csv(game_state_file)

    # Continue processing video
    for i in range(length):
    # for i in range(100):

        # Compute a timestamp
        timestamp += 1/fps

        # Get the data
        ret, frame = screen.read()
        ret, camera_view = camera.read()

        # Crop the frame to only get the play area
        frame = frame[CROP['top']:-CROP['bottom'], CROP['left']:-CROP['right']]
        
        # Get a slice of the event logs
        id_records = eval(game_state.iloc[i].to_dict()['state'])

        result = elp.LogResult(
            id_records=id_records,
            timestamp=timestamp,
            corrections={'OFFSET': (-180,-260), 'AFFINE': (2.25,2)}
        )
        frame = result.render(frame)
        vis = imutils.resize(elp.combine_frames(frame, imutils.resize(camera_view, height=frame.shape[0])), width=1700)

        cv2.imshow('output', vis)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
