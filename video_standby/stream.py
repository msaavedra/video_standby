
import collections
from queue import Queue
import select
import time

from video_standby.config import settings
from video_standby.logger import Logger


logger = Logger(settings)


class VideoStream(object):
    
    def __init__(self, name):
        self.name = name
        self.stream_settings = settings[name]
        self.command_queue = Queue()
        self.buffer = self.buffer_stream()
    
    def buffer_stream(self):
        stream = None  # TODO - create this
        duration = self.stream_settings['buffer_duration']
        return VideoStreamBuffer(stream, duration)


class VideoStreamBuffer(object):
    """A circular buffer to hold some video data until we write to a file.
    """
    
    buffer_size = 8192
    
    def __init__(self, stream, block_secs):
        self.stream = stream
        self.block_secs = block_secs
        self.blocks = collections.deque(maxlen=2)
        
        # We'll fill these in with derived values once we start running
        self.block_size = 0
    
    def write(self):
        """
        """
        # Only do anything if we've got data waiting in the input stream
        if select.select([self.stream], [], [], 0)[0]:
            if self.block_size == 0:
                # If we haven't written before, we won't have a block size, so
                # we'll see how much we can read in block_secs to get an idea
                # how big our chunks should be
                chunk = ''
                start_time = time.time()
                while time.time() - start_time < self.block_secs:
                    chunk += self.stream.read(self.buffer_size)
                self.block_size = len(chunk)
                logger.debug('Set block size to %s' % self.block_size)
            else:
                chunk = self.stream.read(self.block_size)
            
            self.blocks.append(chunk)
    
    def read(self):
        if len(self.blocks) > 1:
            return self.blocks.popleft()
        else:
            return ''
