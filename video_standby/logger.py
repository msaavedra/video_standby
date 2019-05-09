
import logging
import sys


TRACE = logging.DEBUG - 1
logging.addLevelName(TRACE, 'TRACE')


def Logger(settings):
    """A factory function to return the project logger.
    """
    logger = logging.getLogger('video_standby')
    level_name = settings['globals']['log_level']
    if level_name == 'TRACE':
        level = TRACE
    else:
        try:
            level = getattr(logging, level_name)
        except AttributeError:
            
            sys.stderr.write(f'Invalid log level {level_name}.\n')
            level_name = settings.default_global_settings.get(
                'log_level',
                'INFO'
                )
            level = getattr(logging, level_name)
            sys.stderr.write(
                f'Using log level {level_name} instead.\n'
                )
    
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter(settings['globals']['log_format'])
            )
        logger.addHandler(handler)
    
    logger.info(f'Created {level_name} logger.')
    return logger


def trace(logger, message, *args, **kwargs):
    if logger.isEnabledFor(TRACE):
        logger._log(TRACE, message, args, **kwargs)

logging.Logger.trace = trace