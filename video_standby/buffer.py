
import itertools
from multiprocessing import Condition, Process
from multiprocessing.sharedctypes import RawArray, RawValue
import time

import numpy as np

from .config import settings
from .errors import StreamError
from .lock import SharedExclusiveLock
from .logger import Logger
from .source import VideoSource

logger = Logger(settings)


class Buffer(Process):
    
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.properties = stream.properties
        self.settings = stream.settings
        
        properties_dict = self.properties.to_dict()
        self.frame_count = (settings.buffer_time * self.properties.frame_rate)
        self.frames = tuple(
            BufferedFrame(properties_dict) for _ in range(self.frame_count)
            )
        
        self.write_boundary = Condition()
        self.last_write_position = RawValue('I', 0)
    
    def run(self):
        """Write frames continuously to the buffer until closed.
        """
        logger.info(f'Starting to buffer {self.stream.name} stream.')
        
        writer_index_cycle = itertools.cycle(range(self.frame_count))
        with VideoSource(self.stream.name) as source:
            while self.stream.status != self.stream.STATUS_CLOSED:
                index = next(writer_index_cycle)
                frame = self.frames[index]
                
                with frame.lock.exclusive:
                    try:
                        logger.trace('Capturing frame data...')
                        source.get_frame(frame.np_representation)
                        logger.trace('Finished capturing.')
                    except StreamError as e:
                        if e.code == StreamError.SOURCE_UNAVAILABLE:
                            logger.warning(
                                f'{e.stream_name} - stream is unavailable.'
                                )
                            time.sleep(5)
                        else:
                            logger.error(str(e))
                            raise
                    except KeyboardInterrupt:
                        continue
                
                # Put up a boundary so that readers can't overtake
                with self.write_boundary:
                    self.last_write_position.value = index
                    self.write_boundary.notify_all()
        
        logger.info(f'Finished buffering {self.name} stream.')
    
    def create_reader(self, skip_frames=0):
        with self.write_boundary:
            index = self.last_write_position.value - (self.frame_count * .75)
            index = int(index) % self.frame_count
        
        while True:
            # Wait if we're overtaking the writer
            with self.write_boundary:
                if index == self.last_write_position.value:
                    logger.trace('Reader overtook writer. Waiting...')
                    self.write_boundary.wait()
            
            # Yield a copy of the  frame if we're not supposed to skip it.
            if index % (skip_frames + 1) == 0:
                yield index, self.frames[index].copy()
            
            index = (index + 1) % self.frame_count


class BufferedFrame:
    
    def __init__(self, properties):
        self._data = RawArray('B', properties['frame_size'])
        self._lock = SharedExclusiveLock()
        self.frame_shape = properties['frame_shape']
        self.frame_dtype = properties['frame_dtype']
    
    @property
    def np_representation(self):
        return np.ndarray(
            self.frame_shape,
            dtype=self.frame_dtype,
            buffer=self._data,
            )
    
    def copy(self):
        with self._lock.shared:
            return self.np_representation.copy()
