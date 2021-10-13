
import atexit
import json
import multiprocessing
from multiprocessing.util import _exit_function
import queue
import threading

import falcon
from falcon.routing.converters import BaseConverter
from gunicorn.app.base import BaseApplication
import zmq

from .config import settings
from .errors import StreamError
from .logger import Logger
from .stream import VideoStream, StreamGroup



logger = Logger(settings)


class Daemon:
    
    def __init__(self):
        self._streams = {}
    
    def run(self):
        while True:
            updated = settings.load_updates()
            if updated:
                self.streams.update(settings)

    def __init__(self, settings=None):
        if settings:
            self._stream_map = {
                k: VideoStream(k, v) for (k, v) in settings.sources.items()
                }
        else:
            self._stream_map = {}
    
    def update(self, settings):
        for name, stream_settings in settings.sources.items():
    
    def __getitem__(self, name):
        return self._stream_map[name]
    
    def __iter__(self):
        yield from self._stream_map.values()


class StreamManager:
    
    def __init__(self, name):
        self.stream = VideoStream.start_new(name)


class VideoStandbyAPI(falcon.API):
    
    def __init__(self, video_streams, *args, **kwargs):
        self.video_streams = video_streams
        super().__init__(*args, **kwargs)


class WebDaemon(BaseApplication):
    
    def __init__(self, video_streams):
        self.api = VideoStandbyAPI(video_streams)
        self.api.router_options.converters['stream'] = StreamRouteConverter
        self.api.add_route(
            '/streams/{stream_name}',
            resources.StreamResource(video_streams),
            )
        self.api.add_route(
            '/streams/{stream_name}/command',
            resources.StreamCommandResource(video_streams),
            )
        super().__init__(self)
    
    def init(self, parser, opts, args):
        pass
    
    def load_config(self):
        self.cfg.set(
            'bind',
            f'{settings.globals.hostname}:{settings.globals.port}',
            )
        self.cfg.set('workers', settings.globals.handler_count)
        self.cfg.set('post_worker_init', self.on_post_worker_init)
    
    def load(self):
        return self.api
    
    def on_post_worker_init(self, worker):
        atexit.unregister(_exit_function)


class StreamRouteConverter(BaseConverter):
    
    streams = {}
    streams_lock = threading.Lock()
    
    def __init__(self):
        logger.info(type(settings.sources))
        with self.streams_lock:
            for source in settings.sources:
                if source not in self.streams:
                    try:
                        self.streams[source] = VideoStream(source)
                    except StreamError:
                        logger.error(f'Stream {source} not found.')
    
    def convert(self, value):
        try:
            return self.streams[value]
        except KeyError:
            return None
