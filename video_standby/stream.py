
import itertools
from queue import Queue
from multiprocessing.sharedctypes import RawArray, RawValue

import cv2
import numpy as np

from video_standby.config import settings
from video_standby.logger import Logger


logger = Logger(settings)


class VideoStream(object):
    
    def __init__(self, name):
        self.name = name
        self.stream_settings = settings[name]
        self.command_queue = Queue()
        self.cap = cv2.VideoCapture(self.stream_settings['location'])
        
        _, sample_frame = self.cap.read()
        self.frame_shape = sample_frame.shape
        self.height, self.width, self.channels = self.frame_shape
        self.frame_dtype = sample_frame.dtype
        self.color_depth = self.frame_dtype.itemsize
        self.frame_size = sample_frame.nbytes
        del sample_frame
        
        self.buffer = VideoStreamBuffer(
            self.cap,
            self.stream_settings['buffer_frames'],
            self.frame_size,
            self.frame_shape,
            self.frame_dtype,
            )
    
    def start_recording(self):
        pass


class VideoStreamBuffer(object):
    """A circular buffer to hold frames of video data in shared memory.
    """
    
    block_size = 8192  # This is a default that will get adjusted dynamically.
    
    def __init__(self, cap, frame_count, frame_size, frame_shape, frame_dtype):
        self.cap = cap
        self.frame_count = frame_count
        self.writer_index_cycle = itertools.cycle(range(frame_count))
        self.frames = tuple([
            FrameSlot(frame_size, frame_shape, frame_dtype)
            for _ in range(frame_count)
            ])
    
    def write(self):
        index = next(self.writer_index_cycle)
        self.frames[index].write_from_cap(self.cap)
    
    def create_reader(self):
        yield from (f.copy_frame() for f in itertools.cycle(self.frames))


class FrameSlot(object):
    
    def __init__(self, frame_size, frame_shape, frame_dtype):
        self.seq_lock = RawValue('B')
        self.shared_array = RawArray('B', frame_size)
        self.frame = np.ndarray(
            frame_shape,
            dtype=frame_dtype,
            buffer=self.shared_array
            )
    
    def write_from_cap(self, cap):
        self.seq_lock.value += 1
        cap.read(self.frame)
        self.seq_lock.value += 1
    
    def copy_frame(self, allow_corruption=False):
        if allow_corruption is True:
            return self.frame.copy()
        
        while True:
            initial_sequence = self.seq_lock.value
            frame_copy = self.frame.copy()
            final_sequence = self.seq_lock.value
            
            if (initial_sequence == final_sequence) and (final_sequence & 1):
                return frame_copy
            else:
                logger.debug('Frame corruption detected, recopying!')
