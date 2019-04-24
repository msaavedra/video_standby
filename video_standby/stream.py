
from contextlib import contextmanager
import itertools
from queue import Queue
from multiprocessing import Condition, Lock, Process
from multiprocessing.sharedctypes import RawArray, RawValue, Value
import os
import sys
import time

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


class VideoStream:
    
    def __init__(self, name, start_immediately=True):
        self._status = Value('B', STREAM_STATUS_INITIALIZING)
        self.buffering_process = None
        self.recording_process = None
        self.cap = None
        
        self.name = name
        logger.debug(str(settings['sources']))
        self.settings = settings['sources'][name]
        self.command_queue = Queue()
        
        # Load video properties from a sample frame
        temp_capture = self.create_capture_device()
        _, sample_frame = temp_capture.read()
        temp_capture.release()
        self.frame_shape = sample_frame.shape
        self.height, self.width, self.channels = self.frame_shape
        self.frame_dtype = sample_frame.dtype
        self.color_depth = self.frame_dtype.itemsize
        self.frame_size = sample_frame.nbytes
        del sample_frame
        
        # Set up a buffer
        self.frame_count = self.settings['buffer_frames']
        self.frames = tuple(
            RawArray('B', self.frame_size) for _ in range(self.frame_count)
            )
        self.frame_locks = tuple(
            SharedExclusiveLock() for _ in range(self.frame_count)
            )
        self.writer_index_cycle = itertools.cycle(range(self.frame_count))
        self.write_boundary = Condition()
        self.last_write_position = RawValue('I', 0)
        
        
        if start_immediately:
            self.start_buffering()
    
    def create_capture_device(self):
        return cv2.VideoCapture(self.settings['location'])
    
    @property
    def status(self):
        with self._status.get_lock():
            return self._status.value
    
    def start_buffering(self):
        with self._status.get_lock():
            if self._status.value == STREAM_STATUS_CLOSED:
                raise StreamCommandSequenceError('Stream is closed.')
            
            self._status.value = STREAM_STATUS_STANDBY
            self.buffering_process = Process(
                target=_buffer_stream,
                args=(self,),
                )
            self.buffering_process.start()
    
    def start_recording(self):
        with self._status.get_lock():
            if self._status.value == STREAM_STATUS_CLOSED:
                raise StreamCommandSequenceError('Stream is closed.')
            
            if self._status.value == STREAM_STATUS_INITIALIZING:
                raise StreamCommandSequenceError('Stream is not buffered.')
            
            self._status.value = STREAM_STATUS_RECORDING
            self.recording_process = Process(
                target=_record_stream,
                args=(self,),
                )
            self.recording_process.start()
    
    def stop_recording(self):
        if self.status != STREAM_STATUS_RECORDING:
            raise StreamCommandSequenceError('Stream is not recording.')

    def write(self):
        if self.cap is None:
            self.cap = self.create_capture_device()
        
        index = next(self.writer_index_cycle)
        lock = self.frame_locks[index]
        
        with lock.exclusive:
            numpy_frame = np.ndarray(
                self.frame_shape,
                dtype=self.frame_dtype,
                buffer=self.frames[index]
                )
            logger.debug(
                f'Created numpy array from shared memory at index {index}.'
                )
            if self.cap.isOpened():
                logger.debug('Video device is open, reading from it...')
                self.cap.read(numpy_frame)
                logger.debug('Captured frame data to numpy array.')
            else:
                logger.error('Video device closed unexpectedly!')
                self.close()
    
        # Put up a boundary so that readers can't overtake
        with self.write_boundary:
            self.last_write_position.value = index
            self.write_boundary.notify_all()
    
    def create_reader(self):
        with self.write_boundary:
            index = self.last_write_position.value - (self.frame_count // 2)
            index = index % self.frame_count
    
        while True:
            # Wait if we're overtaking the writer
            with self.write_boundary:
                if index == self.last_write_position.value:
                    self.write_boundary.wait()
            
            yield self.copy_frame(index)
            index = (index + 1) % self.frame_count
    
    def copy_frame(self, index):
        lock = self.frame_locks[index]
        with lock.shared:
            numpy_frame = np.ndarray(
                self.frame_shape,
                dtype=self.frame_dtype,
                buffer=self.frames[index]
                )
            return numpy_frame.copy()
    
    def close(self):
        with self._status.get_lock():
            if self._status.value == STREAM_STATUS_CLOSED:
                raise StreamCommandSequenceError('Stream is already closed.')
            
            self._status.value = STREAM_STATUS_CLOSED
        
        if self.buffering_process:
            self.buffering_process.join()
        
        if self.recording_process:
            self.recording_process.join()
        
        if self.cap:
            self.cap.release()
    
    def __del__(self):
        if self.status < STREAM_STATUS_CLOSED:
            self.close()

    def __str__(self):
        return '%s<%s> (%s)' % (
            self.__class__.__name__,
            self.name,
            STREAM_STATUS_LABELS[self.status],
            )


class SharedExclusiveLock:
    
    def __init__(self):
        self.exclusive_lock = Lock()
        self.shared_counter_lock = Lock()
        self.shared_counter = RawValue('B', 0)
        
    def acquire_shared(self):
        with self.shared_counter_lock:
            self.shared_counter.value += 1
            if self.shared_counter.value == 1:
                self.exclusive_lock.acquire()
        logger.debug('Acquired lock in shared mode.')
    
    def release_shared(self):
        with self.shared_counter_lock:
            if self.shared_counter.value < 1:
                raise ValueError('Cannot decrement shared counter below zero.')
            self.shared_counter.value -= 1
            if self.shared_counter.value == 0:
                self.exclusive_lock.release()
        logger.debug('Released lock in shared mode.')
    
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
        logger.debug('Acquired lock in exclusive mode.')
    
    def release_exclusive(self):
        self.exclusive_lock.release()
        self.shared_counter_lock.release()
        logger.debug('Released lock in exclusive mode.')
    
    @property
    @contextmanager
    def exclusive(self):
        self.acquire_exclusive()
        try:
            yield self
        finally:
            self.release_exclusive()


def _buffer_stream(stream):
    logger.info('Starting to buffer stream.')
    while stream.status != STREAM_STATUS_CLOSED:
        logger.debug('Writing frame to buffer...')
        stream.write()
        logger.debug('Buffer write completed.')
    logger.info('Finished buffering stream.')


def _record_stream(stream):
    logger.info('Starting to record stream.')
    reader = stream.create_reader()
    output_path = os.path.join(
        stream.settings['save_directory'],
        stream.name,
        '%s.mp4' % int(time.time())
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.debug(f'Writing video to {output_path}.')
    writer = cv2.VideoWriter(
        output_path,
        int(cv2.VideoWriter_fourcc(*'mp4v')),
        30.0,
        (stream.width, stream.height),
        )
    try:
        for frame in reader:
            writer.write(frame)
            if stream.status != STREAM_STATUS_RECORDING:
                break
    finally:
        writer.release()
        logger.info('Finished recording stream.')


def test():
    try:
        names = sys.argv[1:]
    except IndexError:
        names = []
    logger.info(f'Starting test of {names} cameras...')
    streams = [VideoStream(name) for name in names]
    time.sleep(30)
    for stream in streams:
        stream.start_recording()
    time.sleep(2)
    for stream in streams:
        stream.stop_recording()
        stream.close()
    logger.info('Finishing test...')


if __name__ == '__main__':
    test()
