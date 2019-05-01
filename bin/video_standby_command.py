#!/usr/bin/env python3

import sys

import requests

from video_standby.config import settings
from video_standby.logger import Logger


logger = Logger(settings)


def send_command(stream, command):
    host = settings.globals.hostname
    port = settings.globals.port
    url = f'http://{host}:{port}/streams/{stream}/command'
    logger.info(f'URL: {url}')
    response = requests.post(url, data=command)
    if response.status_code > 399:
        sys.stderr.write('Sending command failed: %s: %s' % (
            response.status_code,
            response.text,
            ))


if __name__ == '__main__':
    stream = sys.argv[1]
    command = sys.argv[2]
    send_command(stream, command)