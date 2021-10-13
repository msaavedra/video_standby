
from types import MappingProxyType


class StreamError(Exception):
    SOURCE_UNAVAILABLE = 1
    NO_RETURN_VALUE = 2
    BAD_COMMAND_SEQUENCE = 3
    
    code_defaults = MappingProxyType({
        SOURCE_UNAVAILABLE: 'stream source device is unavailable',
        NO_RETURN_VALUE: 'could not read data from stream',
        BAD_COMMAND_SEQUENCE: 'order of command sequence is incorrect',
        })
    
    def __init__(self, stream_name, code, message=None):
        self.stream_name = stream_name
        self.code = code
        
        # Always get the default message, since it also validates the code.
        default_message = self.code_defaults[code]
        if not message:
            message = default_message
        
        message = f'<{stream_name}> - {message}'
        if not message.endswith('.'):
            message += '.'
        
        super().__init__(message)
