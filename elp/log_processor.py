from typing import Dict, Union, Optional, Any
import logging
import datetime

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger("elp")

class LogResult:

    def __init__(self, **kwargs):
        self.container: Dict = kwargs
        self.img_size = 500

    def apply_corrections(self, x, y, h, w):
        c = self.container['corrections']
        xx = int(((x+1)*h/2+c['OFFSET'][0])*c['AFFINE'][0])
        yy = int(((y+1)*w/2+c['OFFSET'][1])*c['AFFINE'][1])
        return (xx, yy)

    def render(self, img:Optional[np.ndarray] = None) -> np.ndarray:
        
        if type(img) == type(None):
            img = (255*np.ones((self.img_size,self.img_size,3))).astype(np.uint8)

        # Draw the information
        img = self.draw_position(img)
        img = self.draw_timestamp(img)

        return img

    def draw_position(self, img: np.ndarray) -> np.ndarray:

        h,w = img.shape[:2]

        # Draw the circles
        for id, id_data in self.container['id_records'].items():
            x = id_data['x']
            y = id_data['y']
            pt = self.apply_corrections(x,y,h,w)
            img = cv2.circle(img, pt, 20, (0,0,255), 2)
            img = cv2.putText(img, str(id), (pt[0]-17, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)

        return img

    def draw_timestamp(self, img: np.ndarray) -> np.ndarray:
        m, s = divmod(self.container['timestamp'], 60)
        h, m = divmod(m, 60)
        
        return cv2.putText(
            img,
            f'{int(h):02}:{int(m):02}:{s:.2f}',
            (0,25), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0,0,255), 
            2, 
            cv2.LINE_AA
        )

class LogProcessor:

    def __init__(self, start_time: float = 0, corrections: Dict[str, Any] = {'OFFSET': (0,0), 'AFFINE': (1,1)}):
        self.id_records: Dict = {}
        self.timestamp: float = start_time
        self.corrections = corrections

    def str_datetime_to_datetime(self, str_datetime: str):
        return datetime.datetime.strptime(str_datetime, "%H:%M:%S:%f")

    def update_record(self, id, log_record: pd.Series):
        self.id_records[id] = {'x': log_record.x, 'y': log_record.y, 'bpid': log_record.bpid}

    def step(self, log_record: Union[pd.Series, pd.DataFrame], timestamp: float):

        # Update the timestamp
        self.timestamp = timestamp

        # Processing single pd.Series
        if isinstance(log_record, pd.Series):
            self.update_record(log_record.id, log_record)
        else:
            # Break up by id and then select the latest record for each
            id_dfs = dict(tuple(log_record.groupby('id')))
            for id, df in id_dfs.items():
                self.update_record(id, df.iloc[-1])

        return LogResult(
            id_records=self.id_records.copy(),
            timestamp=self.timestamp,
            corrections=self.corrections
        )
