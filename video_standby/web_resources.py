
import json

import falcon

from video_standby.config import settings
from video_standby.logger import Logger


logger = Logger(settings)


class _StreamBase:
    
    def __init__(self, video_streams):
        self.video_streams = video_streams
    
    def _get_stream_by_name(self, name):
        return self.video_streams[name]


class StreamResource(_StreamBase):
    
    def on_get(self, req, resp, stream_name):
        stream = self._get_stream_by_name(stream_name)
        resp.body = json.dumps({
            'name': stream.name,
            'height': stream.height,
            'width': stream.width,
            'channels': stream.channels,
            'color_depth': stream.color_depth,
            'status': STREAM_STATUS_LABELS[stream.status]
            })
        resp.content_type = 'application/json'


class StreamCommandResource(_StreamBase):
    
    def on_post(self, req, resp, stream_name):
        stream = self._get_stream_by_name(stream_name)
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
