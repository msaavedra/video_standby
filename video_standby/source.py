
import time
import statistics

import cv2

from .config import settings
from .errors import StreamError
from .logger import Logger


logger = Logger(settings)


class VideoSource:
    
    def __init__(self, name):
        self.name = name
        self.cap = cv2.VideoCapture(settings.sources[name].location)
    
    def _wait_for_device(self):
        for _ in range(5):
            if self.cap.isOpened():
                return
            else:
                logger.warn('Waiting for device...')
                time.sleep(1)
        else:
            raise StreamError(self.name, StreamError.SOURCE_UNAVAILABLE)
    
    def get_frame(self, destination_array=None):
        """Write the next frame data to the provided numpy array.

        If an array is not provided, write it to a new array.
        """
        self._wait_for_device()
        
        if destination_array:
            ret, frame = self.cap.read(destination_array)
        else:
            ret, frame = self.cap.read()
        if not ret:
            raise StreamError(self.name, StreamError.NO_RETURN_VALUE)
        frame_msec = int(round(self.cap.get(cv2.CAP_PROP_POS_MSEC), 4) * 1000)
        logger.info(f'Frame time:{frame_msec}')
        return frame
    
    def get_frame_rate(self, seconds=2):
        self._wait_for_device()
        
        # The frame rate is not available for all stream types.
        frame_rate = round(self.cap.get(cv2.CAP_PROP_FPS))
        if frame_rate:
            logger.info(f'Frame rate from cv2 properties: {frame_rate}')
            return frame_rate
        
        counts = []
        start_time = time.time()
        while True:
            for _ in range(10):
                frame_start_time = time.time()
                ret, frame = self.cap.read()
                if ret:
                    counts.append(time.time() - frame_start_time)
            
            elapsed = time.time() - start_time
            if elapsed > seconds:
                break
        
        # OpenCV's built in Buffering throws off the calculated rate. We
        # can't turn off buffering or even reliably get the buffer size
        # from most OpenCV backends , so lets remove short values from the
        # beginning.
        sd = statistics.stdev(counts)
        
        last = None
        for index, value in enumerate(counts):
            if last is None or value - last < sd:
                last = value
            else:
                slice_index = index
                break
        else:
            slice_index = 0
        
        counts = counts[slice_index:]
        
        frame_rate = round(1 / statistics.mean(counts))
        logger.info(f'Frame rate as calculated: {frame_rate}')
        
        return frame_rate
    
    def __enter__(self):
        logger.trace('Acquiring capture device.')
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        logger.trace('Releasing capture device.')
        self.cap.release()
