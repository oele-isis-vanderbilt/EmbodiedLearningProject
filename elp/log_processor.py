from typing import Dict, Union, Optional
import logging
import datetime

import cv2
import numpy as np

logger = logging.getLogger("elp")

class LogResult:

    def __init__(self, **kwargs):
        self.container: Dict = kwargs

    def render(self, img:Optional[np.ndarray] = None, img_size:int = 500) -> np.ndarray:
        # import pdb; pdb.set_trace()
        if img:
            ...
        else:
            img = (255*np.ones((img_size,img_size,3))).astype(np.uint8)

            for id, id_data in self.container['id_records'].items():
                x = id_data['x']
                y = id_data['y']
                pt = (-1*int(x*img_size), -1*int(y*img_size))
                img = cv2.circle(img, pt, 20, (0,0,255), 2)

            return img

class LogProcessor:

    def __init__(self, start_time:Union[str, datetime.datetime]):
        self.id_records: Dict = {}
        self.timestamp: float = 0

        if isinstance(start_time, str):
            self.start_time = self.str_datetime_to_datetime(start_time)
        else:
            self.start_time: datetime.datetime = start_time

    def str_datetime_to_datetime(self, str_datetime: str):
        return datetime.datetime.strptime(str_datetime, "%H:%M:%S:%f")

    def step(self, log_record):
        logger.debug(log_record)
        # import pdb; pdb.set_trace()

        # Update the ID record
        self.id_records[log_record.id] = {'x': log_record.x, 'y': log_record.y, 'bpid': log_record.bpid}
        self.timestamp = (self.str_datetime_to_datetime(log_record.timestamp) - self.start_time).total_seconds()

        return LogResult(
            id_records=self.id_records.copy(),
            timestamp=self.timestamp
        )
