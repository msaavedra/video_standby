
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
        
        self.name = name
        logger.debug(str(settings.sources))
        self.settings = settings.sources[name]
        self.command_queue = Queue()
        
        # Load stream properties from a sample frame
        with StreamSource(name) as source:
            sample_frame = source.get_frame()
        
        self.frame_shape = sample_frame.shape
        self.height, self.width, self.channels = self.frame_shape
        self.frame_dtype = sample_frame.dtype
        self.color_depth = self.frame_dtype.itemsize
        self.frame_size = sample_frame.nbytes
        del sample_frame
        
        # Set up a buffer
        self.frame_count = self.settings.buffer_frames
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
    
    @property
    def status(self):
        with self._status.get_lock():
            return self._status.value
    
    def start_buffering(self):
        with self._status.get_lock():
            if self._status.value == STREAM_STATUS_CLOSED:
                raise StreamCommandSequenceError('Stream is closed.')
            
            self.buffering_process = Process(
                target=buffer,
                args=(self,),
                )
            self.buffering_process.start()
            
            self._status.value = STREAM_STATUS_STANDBY
    
    def start_recording(self):
        with self._status.get_lock():
            if self._status.value == STREAM_STATUS_CLOSED:
                raise StreamCommandSequenceError('Stream is closed.')
            
            if self._status.value == STREAM_STATUS_INITIALIZING:
                raise StreamCommandSequenceError('Stream is not buffering.')
            
            self._status.value = STREAM_STATUS_RECORDING
            self.recording_process = Process(
                target=record,
                args=(self,),
                )
            self.recording_process.start()
    
    def stop_recording(self):
        with self._status.get_lock():
            if self._status.value != STREAM_STATUS_RECORDING:
                return
            
            self._status.value = STREAM_STATUS_STANDBY

    def write_to_buffer(self):
        """Write frames continuously to the buffer until closed.
        """
        logger.info(f'Starting to buffer {self.name} stream.')
    
        with StreamSource(self.name) as source:
            while self.status != STREAM_STATUS_CLOSED:
                index = next(self.writer_index_cycle)
                lock = self.frame_locks[index]
            
                with lock.exclusive:
                    numpy_frame = np.ndarray(
                        self.frame_shape,
                        dtype=self.frame_dtype,
                        buffer=self.frames[index]
                        )
                    try:
                        logger.trace('Capturing frame data to numpy array...')
                        source.write_frame(numpy_frame)
                        logger.trace('Finished capturing.')
                    except StreamError as e:
                        logger.error(str(e))
                        raise
                    except KeyboardInterrupt:
                        continue
            
                # Put up a boundary so that readers can't overtake
                with self.write_boundary:
                    self.last_write_position.value = index
                    self.write_boundary.notify_all()
    
        logger.info(f'Finished buffering {self.name} stream.')

    def write_to_file(self):
        """Copy frames out of the buffer and encode them into a video file.
        """
        logger.info('Starting to record stream.')
        reader = self.create_reader()
        output_path = os.path.join(
            self.settings.save_directory,
            '%s-%s.mp4' % (
                self.name,
                time.strftime('%Y%m%d%H%M%S', time.localtime()),
                )
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logger.info(f'Writing video to {output_path}.')
        writer = cv2.VideoWriter(
            output_path,
            int(cv2.VideoWriter_fourcc(*'mp4v')),
            30.0,
            (self.width, self.height),
            )
        try:
            for index, frame in reader:
                writer.write(frame)
                if self.status != STREAM_STATUS_RECORDING:
                    logger.info('Received record stop command.')
                    break
            remaining_frames = self.get_distance_to_write_boundary(
                index) - 1
            logger.debug(f'{remaining_frames} remaining frames in buffer.')
            for i in range(remaining_frames):
                index, frame = next(reader)
                writer.write(frame)
                logger.debug(f'Got {i + 1} of {remaining_frames} from {index}')
        finally:
            writer.release()
            logger.info('Finished recording stream.')

    def create_reader(self, skip_frames=0):
        with self.write_boundary:
            index = self.last_write_position.value - (self.frame_count * .75)
            index = int(index) % self.frame_count
    
        while True:
            # Wait if we're overtaking the writer
            with self.write_boundary:
                if index == self.last_write_position.value:
                    self.write_boundary.wait()
            
            # Yield the frame if we haven't been instructed to skip it.
            if index % (skip_frames + 1) == 0:
                yield index, self.copy_frame(index)
            
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
    
    def get_distance_to_write_boundary(self, index):
        with self.write_boundary:
            write_position = self.last_write_position.value
        
        if index <= write_position:
                distance = write_position - index
        else:
            distance = write_position + self.frame_count - index
        
        logger.debug(
            f'Index-{index}, '
            f'Boundary-{write_position}, '
            f'Size-{self.frame_count}, ',
            f'Distance-{distance}'
            )
        
        return distance
    
    def close(self):
        with self._status.get_lock():
            if self._status.value == STREAM_STATUS_CLOSED:
                return
            
            self._status.value = STREAM_STATUS_CLOSED
        
        try:
            if self.buffering_process:
                self.buffering_process.join()
            
            if self.recording_process:
                self.recording_process.join()
        except AssertionError as e:
            if 'can only join a child process' not in str(e):
                raise
    
    def __del__(self):
        self.close()

    def __str__(self):
        return '%s<%s> (%s)' % (
            self.__class__.__name__,
            self.name,
            STREAM_STATUS_LABELS[self.status],
            )


class StreamSource:
    
    def __init__(self, name):
        self.cap = cv2.VideoCapture(settings.sources[name].location)
    
    def write_frame(self, destination_array):
        """Write the next frame data to the provided numpy array.
        """
        if not self.cap.isOpened():
            raise StreamError('Stream is closed')
        
        ret, frame = self.cap.read(destination_array)
        if not ret:
            raise StreamError('Could not read data from stream.')
    
    def get_frame(self):
        """Read the next frame data and return it as a new numpy array.
        """
        if not self.cap.isOpened():
            raise StreamError('Stream is closed')
    
        ret, frame = self.cap.read()
        if not ret:
            raise StreamError('Could not read data from stream.')
        
        return frame
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        logger.debug('Releasing capture device')
        self.cap.release()


class StreamError(Exception):
    pass


class SharedExclusiveLock:
    
    def __init__(self):
        self.exclusive_lock = Lock()
        self.shared_counter_lock = Lock()
        self.shared_counter = RawValue('B', 0)
        
    def acquire_shared(self):
        logger.trace('Attempting to acquire lock in shared mode...')
        with self.shared_counter_lock:
            self.shared_counter.value += 1
            if self.shared_counter.value == 1:
                self.exclusive_lock.acquire()
        logger.trace('Acquired lock in shared mode.')
    
    def release_shared(self):
        with self.shared_counter_lock:
            if self.shared_counter.value < 1:
                raise ValueError('Cannot decrement shared counter below zero.')
            self.shared_counter.value -= 1
            if self.shared_counter.value == 0:
                self.exclusive_lock.release()
        logger.trace('Released lock in shared mode.')
    
    @property
    @contextmanager
    def shared(self):
        self.acquire_shared()
        try:
            yield self
        finally:
            self.release_shared()
    
    def acquire_exclusive(self):
        logger.trace('Attempting to acquire lock in exclusive mode...')
        self.shared_counter_lock.acquire()
        self.exclusive_lock.acquire()
        logger.trace('Acquired lock in exclusive mode.')
    
    def release_exclusive(self):
        self.exclusive_lock.release()
        self.shared_counter_lock.release()
        logger.trace('Released lock in exclusive mode.')
    
    @property
    @contextmanager
    def exclusive(self):
        self.acquire_exclusive()
        try:
            yield self
        finally:
            self.release_exclusive()


def buffer(stream):
    stream.write_to_buffer()


def record(stream):
    stream.write_to_file()


def test():
    import random
    
    def record_stream(stream):
        for _ in range(random.randint(1,5)):
            time.sleep(random.uniform(.1, 45.0))
            logger.debug(f'Starting recording test of {stream.name} camera.')
            stream.start_recording()
            time.sleep(random.uniform(2.5, 30.0))
            stream.stop_recording()
            logger.debug(f'Finished test of {stream.name} camera.')
    
    try:
        names = sys.argv[1:]
    except IndexError:
        names = []
    
    streams = [VideoStream(name) for name in names]
    
    import threading
    threads = []
    for stream in streams:
        thread = threading.Thread(target=record_stream, args=(stream,))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    
    for stream in streams:
        stream.close()

if __name__ == '__main__':
    test()
