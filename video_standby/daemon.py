
import atexit
import json
from multiprocessing.util import _exit_function
import threading

import falcon
from falcon.routing.converters import BaseConverter
from gunicorn.app.base import BaseApplication

from video_standby.config import settings
from video_standby.logger import Logger
from video_standby.stream import VideoStream, STREAM_STATUS_LABELS


logger = Logger(settings)


class VideoStandbyAPI(falcon.API):
    
    def __init__(self):
        pass


class Daemon(BaseApplication):
    
    def __init__(self):
        self.api = falcon.API()
        self.api.router_options.converters['stream'] = StreamRouteConverter
        self.api.add_route(
            '/streams/{stream:stream}',
            StreamResource(),
            )
        self.api.add_route(
            '/streams/{stream:stream}/command',
            StreamCommandResource(),
            )
        super().__init__(self)
    
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
                    self.streams[source] = VideoStream(source)
    
    def convert(self, value):
        try:
            return self.streams[value]
        except KeyError:
            return None


class StreamResource:
    
    def on_get(self, req, resp, stream):
        resp.body = json.dumps({
            'name': stream.name,
            'height': stream.height,
            'width': stream.width,
            'channels': stream.channels,
            'color_depth': stream.color_depth,
            'status': STREAM_STATUS_LABELS[stream.status]
            })
        resp.content_type = 'application/json'


class StreamCommandResource:
    
    def on_post(self, req, resp, stream):
        command = req.bounded_stream.read().decode()
        logger.info(f'Command: {command}')
        if command == 'start_recording':
            stream.start_recording()
        elif command == 'stop_recording':
            stream.stop_recording()
        else:
            resp.status = falcon.HTTP_409
            return
        
        resp.status = falcon.HTTP_200
        resp.body = 'success'
