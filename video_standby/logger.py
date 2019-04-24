
import logging
import sys


def Logger(settings):
    """A factory function to return the project logger.
    """
    logger = logging.getLogger('video_standby')
    level_name = settings['global']['log_level']
    try:
        level = getattr(logging, level_name)
    except AttributeError:
        sys.stderr.write(f'Invalid log level {level_name}.')
        level_name = settings.default_global_settings.get('log_level', 'INFO')
        level = getattr(logging, level_name)
        sys.stderr.write(f'Using log level {level_name} instead.')
    
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(settings['global']['log_format']))
        logger.addHandler(handler)
    
    logger.info(f'Created {level_name} logger.')
    return logger
