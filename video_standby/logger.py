
import logging
import sys


def Logger(settings):
    """A factory function to return the project logger.
    """
    logger = logging.getLogger('video_standby')
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(getattr(logging, settings['log_level']))
        handler.setFormatter(logging.Formatter(settings['log_format']))
        logger.addHandler(handler)
    
    return logger
