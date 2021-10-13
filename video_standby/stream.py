
from multiprocessing import Process, Queue
from multiprocessing.sharedctypes import RawArray, RawValue, Value
import os
import sys
import time

import cv2
import numpy as np

from .buffer import Buffer
from .config import settings
from .errors import StreamError
from .lock import SharedExclusiveLock
from .logger import Logger
from .source import VideoSource


logger = Logger(settings)


class VideoStream(Process):
    STATUS_INITIALIZING = 0
    STATUS_BUFFERING = 1
    STATUS_RECORDING = 2
    STATUS_CLOSED = 3
    
    STREAM_STATUS_LABELS = {
        STATUS_INITIALIZING: 'initializing',
        STATUS_BUFFERING: 'buffering',
        STATUS_RECORDING: 'recording',
        STATUS_CLOSED: 'closed'
        }
    
    def __init__(self, name, settings):
        super().__init__()
        
        self._status = Value('B', self.STATUS_CLOSED)
        self.name = name
        self.command_queue = Queue()
        self.properties = StreamProperties(name)
    
    @property
    def status(self):
        with self._status.get_lock():
            return self._status.value
    
    def run(self):
        with self._status.get_lock():
            self._status.value = self.STATUS_INITIALIZING
        
            self.properties.update(self.name)
        
            buffering_process = Buffer(self)
            buffering_process.start()
            self._status.value = self.STATUS_BUFFERING
    
    def start_buffering(self):
        pass
    
    def start_recording(self):
        with self._status.get_lock():
            if self._status.value == self.STATUS_CLOSED:
                raise StreamError(
                    self.name,
                    StreamError.BAD_COMMAND_SEQUENCE,
                    "can't record a closed stream"
                    )
            
            if self._status.value == self.STATUS_INITIALIZING:
                raise StreamError(
                    self.name,
                    StreamError.BAD_COMMAND_SEQUENCE,
                    "can't record a stream that is not buffered"
                    )
            
            self._status.value = self.STATUS_RECORDING
            self.recording_process = Process(
                target=VideoStream.record,
                args=(self,),
                )
            self.recording_process.start()
    
    def stop_recording(self):
        with self._status.get_lock():
            if self._status.value != self.STATUS_RECORDING:
                return
            
            self._status.value = self.STATUS_BUFFERING
    
    def write_to_file(self):
        """Copy frames out of the buffer and encode them into a video file.
        """
        props = self.properties.to_dict()
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
            props['frame_rate'],
            (props['width'], props['height']),
            )
        try:
            index = None
            for index, frame in reader:
                writer.write(frame)
                if self.status != self.STATUS_RECORDING:
                    logger.info('Received record stop command.')
                    break
            
            if index is None:
                remaining_frames = 0
            else:
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
    
    # A static method suitable for use as a subprocess target.
    record = staticmethod(write_to_file)
    
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
            f'Size-{self.frame_count}, '
            f'Distance-{distance}'
            )
        
        return distance
    
    def close(self):
        with self._status.get_lock():
            if self._status.value == self.STATUS_CLOSED:
                return
            
            self._status.value = self.STATUS_CLOSED
        
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
            self.STREAM_STATUS_LABELS[self.status],
            )


class StreamProperties:
    
    def __init__(self):
        self._lock = SharedExclusiveLock()
        self._frame_shape = RawArray('I', 3)  # 3-array of unsigned ints
        self._dtype_char = RawValue('c')  # char
        self._frame_size = RawValue('L')  # unsigned long
        self._frame_rate = RawValue('I')
    
    def _get_frame_shape(self):
        return tuple(self._frame_shape)
    
    def _get_frame_dtype(self):
        return np.typeDict[self._dtype_char.value.decode('ascii')]()
    
    def _get_frame_size(self):
        return self._frame_size.value
    
    def _get_frame_rate(self):
        return self._frame_rate.value
    
    def update(self, source_name):
        with VideoSource(source_name) as source:
            frame_rate = source.get_frame_rate()
            sample_frame = source.get_frame()
        
        with self._lock.exclusive:
            for i, value in enumerate(sample_frame.shape):
                self._frame_shape[i] = value
            self._dtype_char.value = sample_frame.dtype.char.encode('ascii')
            self._frame_size.value = sample_frame.nbytes
            self._frame_rate.value = frame_rate
    
    def to_dict(self):
        with self._lock.shared:
            frame_dtype = self._get_frame_dtype()
            height, width, channels = self._get_frame_shape()
            return {
                'frame_size': self._get_frame_size(),
                'frame_rate': self._get_frame_rate(),
                'height': height,
                'width': width,
                'color_channels': channels,
                'color_depth': frame_dtype.itemsize,
                }
    
    @property
    def frame_shape(self):
        with self._lock.shared:
            return tuple(self._frame_shape)
    
    @property
    def frame_dtype(self):
        with self._lock.shared:
            return self._get_frame_dtype()
    
    @property
    def color_depth(self):
        return self.frame_dtype.itemsize
    
    @property
    def frame_size(self):
        with self._lock.shared:
            return self._get_frame_size()
    
    @property
    def frame_rate(self):
        with self._lock.shared:
            return self._get_frame_rate()

'''
def buffer(stream):
    stream.write_to_buffer()


def record(stream):
    stream.write_to_file()
'''

#@profiled
def output_hls(stream):
    import subprocess
    import select
    height, width, channels = stream.properties.frame_shape
    height //= 2
    width //= 2
    frame_rate = str(stream.properties.frame_rate)
    reader = stream.create_reader(skip_frames=0)
    args = (
        'ffmpeg',
        '-y',
        '-hide_banner',
        '-f', 'rawvideo',
        '-video_size', f'{width}x{height}',
        '-pixel_format', 'bgr24',
        '-i', '-',
        '-r', frame_rate,
        '-framerate', frame_rate,
        '-an',
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'veryfast',
        '-tune', 'zerolatency',
        '-pix_fmt', 'yuv420p',
        '-f', 'hls',
        '-hls_time', '2',
        '-hls_list_size', '5',
        '-hls_wrap', '10',
        '/home/mike/video_standby/stream.m3u8',
        )
    with open('/home/mike/video_standby/ffmpeg.log', 'w') as ffmpeg_stderr:
        ffmpeg = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stderr=ffmpeg_stderr,
            close_fds=True,
            )
        
        try:
            count = 0
            start = time.time()
            for index, frame in reader:
                frame = cv2.resize(frame, (width, height))
                while True:
                    try:
                        ffmpeg.stdin.write(frame.tostring())
                    except BrokenPipeError:
                        logger.error('Broken pipe to ffmpeg.')
                        raise
                    else:
                        break
                        
                count += 1
                if count >= 2500:
                    elapsed = time.time() - start
                    logger.info(
                        f'Wrote {count} frames in {elapsed} seconds:'
                        f' {int(count / elapsed)} frames per second.'
                        )
                    break
        finally:
            ffmpeg.stdin.close()
            ffmpeg.wait()
            logger.info(f'ffmpeg returncode: {ffmpeg.returncode}')


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
    
    stream = streams[0]
    stream.start_buffering()
    logger.debug('Starting to create HLS files...')
    output_hls(stream)
    
    return
    
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
