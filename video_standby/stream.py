
from contextlib import contextmanager
import itertools
from queue import Queue
from multiprocessing import Condition, Lock, Process
from multiprocessing.sharedctypes import RawArray, RawValue, Value

import cv2
import numpy as np

from video_standby.config import settings
from video_standby.logger import Logger


logger = Logger(settings)

STREAM_STATUS_INITIALIZING = 0
STREAM_STATUS_STANDBY = 1
STREAM_STATUS_RECORDING = 2
STREAM_STATUS_CLOSED = 3

STREAM_STATUS_LABELS = {
    STREAM_STATUS_INITIALIZING: 'initializing',
    STREAM_STATUS_STANDBY: 'standby',
    STREAM_STATUS_RECORDING: 'recording',
    STREAM_STATUS_CLOSED: 'closed'
    }


class StreamCommandSequenceError(Exception):
    pass


class VideoStream(object):
    
    def __init__(self, name, start_immediately=True):
        self.name = name
        self.stream_settings = settings[name]
        self.command_queue = Queue()
        self.cap = cv2.VideoCapture(self.stream_settings['location'])
        
        self._status = Value('B', STREAM_STATUS_INITIALIZING)
        
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
        self.buffering_process = None
        self.recording_process = None
        self.recording_subprocess = None
        
        if start_immediately:
            self.start_buffering()
    
    @property
    def status(self):
        with self._status.get_lock():
            return self._status
    
    def start_buffering(self):
        with self._status.get_lock():
            if self._status.value == STREAM_STATUS_CLOSED:
                raise StreamCommandSequenceError('Stream is closed.')
            
            self._status.value = STREAM_STATUS_STANDBY
            
            # FIXME: closures can't be Process targets.
            def run():
                while self.status != STREAM_STATUS_CLOSED:
                    self.buffer.write()
            
            self.buffering_process = Process(target=run, args=())
            self.buffering_process.start()
    
    def start_recording(self):
        with self._status.get_lock():
            if self._status.value == STREAM_STATUS_CLOSED:
                raise StreamCommandSequenceError('Stream is closed.')
            
            if self._status.value == STREAM_STATUS_INITIALIZING:
                raise StreamCommandSequenceError('Stream is not buffered.')
            
            self._status.value = STREAM_STATUS_RECORDING

            # FIXME: closures can't be Process targets.
            def run():
                while self.status == STREAM_STATUS_RECORDING:
                    pass

            self.recording_process = Process(target=run, args=())
            self.recording_process.start()
    
    def stop_recording(self):
        if self.status != STREAM_STATUS_RECORDING:
            raise StreamCommandSequenceError('Stream is not recording.')
    
    def close(self):
        with self._status.get_lock():
            self._status.value = STREAM_STATUS_CLOSED
        
        if self.buffering_process:
            self.buffering_process.join()
        
        if self.recording_process:
            self.recording_process.join()
        
        self.cap.release()
    
    def __del__(self):
        self.close()

    def __str__(self):
        return '%s<%s> (%s)' % (
            self.__class__.__name__,
            self.name,
            STREAM_STATUS_LABELS[self.status],
            )


class VideoStreamBuffer(object):
    """A circular buffer to hold frames of video data in shared memory.
    """
    
    def __init__(self, cap, frame_count, frame_size, frame_shape, frame_dtype):
        self.cap = cap
        self.frame_count = frame_count
        self.frames = tuple([
            FrameWrapper(index, frame_size, frame_shape, frame_dtype)
            for index in range(frame_count)
            ])
        self.writer_frame_cycle = itertools.cycle(self.frames)
        self.write_boundary = Condition()
        self.last_write_position = RawValue('I', 0)
    
    def write(self):
        frame = next(self.writer_frame_cycle)
        frame.write_from_cap(self.cap)
        
        # Put up a boundary so that readers can't overtake
        with self.write_boundary:
            self.last_write_position.value = frame.index
            self.write_boundary.notify_all()
    
    def create_reader(self):
        with self.write_boundary:
            index = (self.last_write_position.value - (self.frame_count // 2))
            index = index % self.frame_count
        
        while True:
            # Wait if we're overtaking the writer
            with self.write_boundary:
                if index == self.last_write_position.value:
                    self.write_boundary.wait()
            
            frame = self.frames[index]
            yield frame.copy()
            index = (index + 1) % self.frame_count


class FrameWrapper(object):
    
    def __init__(self, index, frame_size, frame_shape, frame_dtype):
        self.index = index
        self.lock = SharedExclusiveLock()
        self.shared_array = RawArray('B', frame_size)
        self.frame = np.ndarray(
            frame_shape,
            dtype=frame_dtype,
            buffer=self.shared_array
            )
    
    def write_from_cap(self, cap):
        with self.lock.exclusive:
            cap.read(self.frame)
    
    def copy(self):
        with self.lock.shared:
            return self.frame.copy()


class SharedExclusiveLock(object):
    
    def __init__(self):
        self.exclusive_lock = Lock()
        self.shared_counter_lock = Lock()
        self.shared_counter = RawValue('B', 0)
        
    def acquire_shared(self):
        with self.shared_counter_lock:
            self.shared_counter.value += 1
            if self.shared_counter.value == 1:
                self.exclusive_lock.acquire()
    
    def release_shared(self):
        with self.shared_counter_lock:
            if self.shared_counter.value < 1:
                raise ValueError('Cannot decrement shared counter below zero.')
            self.shared_counter.value -= 1
            if self.shared_counter.value == 0:
                self.exclusive_lock.release()
    
    @property
    @contextmanager
    def shared(self):
        self.acquire_shared()
        try:
            yield self
        finally:
            self.release_shared()
    
    def acquire_exclusive(self):
        self.shared_counter_lock.acquire()
        self.exclusive_lock.acquire()
    
    def release_exclusive(self):
        self.exclusive_lock.release()
        self.shared_counter_lock.release()
    
    @property
    @contextmanager
    def exclusive(self):
        self.acquire_exclusive()
        try:
            yield self
        finally:
            self.release_exclusive()
