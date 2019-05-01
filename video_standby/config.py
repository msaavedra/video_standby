
import os
import sys

import yaml

if os.name == 'posix':
    _default_settings_base = '/etc'
    _default_ipc_base = '/tmp'
elif os.name == 'nt':
    _default_settings_base = os.environ('APPDATA')
    _default_ipc_base = os.environ('APPDATA')
else:
    _default_settings_base = os.getcwd()
    _default_ipc_base = os.getcwd()

DEFAULT_SETTINGS_PATH = os.path.join(
    _default_settings_base,
    'video_standby',
    'settings.yaml',
    )
DEFAULT_IPC_PATH = os.path.join(
    _default_ipc_base,
    'video_standby',
    'ipc.sock'
    )


class Settings(object):
    
    default_global_settings = {
        'hostname': 'localhost',
        'port': 8080,
        'handler_count': 2,
        'log_level': 'INFO',
        'log_format': (
            '%(asctime)s: '
            'pid:%(process)d '
            '%(module)s.%(name)s:%(lineno)s'
            ' - %(message)s'
            )
        }
    default_source_settings = {
        'location': 'rtsp://localhost/',
        'codec': 'h264',
        'save_directory': os.path.join(
            os.path.expanduser('~'),
            'video_standby'
            ),
        'buffer_frames': 60,
        }
    
    def __init__(self, path):
        self.path = path
        self._settings = self._load()
        self.mtime = 0
    
    def __getitem__(self, key):
        return self._settings[key]
    
    def __getattr__(self, name):
        return self._settings[name]
    
    def reload_on_update(self):
        return False  # FIXME: add reloading support
    
    def _load(self):
        try:
            with open(self.path, 'r') as f:
                file_settings = yaml.safe_load(f.read())
        except Exception as e:
            sys.stderr.write_to_buffer('Failed opening settings file %s. Reason: %s' % (
                self.path,
                e,
                ))
            file_settings = {}
        
        global_settings = self._deep_merge(
            self.default_global_settings,
            file_settings.get('globals', {})
            )
        sources = {
            name: self._deep_merge(self.default_source_settings, source)
            for (name, source) in file_settings.get('sources', {}).items()
            }
        return SettingsNamespace({
            'globals': global_settings,
            'sources': sources
            })
    
    def _deep_merge(self, base, updates):
        base = base.copy()
        
        for key, value in updates.items():
            if key not in base:
                continue
            if isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
    
        return SettingsNamespace(base)


class SettingsNamespace(dict):
    """A simple read-only namespace to hold settings.
    
    Example:
        >>> settings = SettingsNamespace({'test': 'hello'})
        >>> print(settings.test)
        hello
    """
    def __init__(self, source_dict=None):
        super().__init__(source_dict or {})
    
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__} object has no attribute '{name}'",
                )
    
    def __setattr__(self, name, value):
        raise NotImplementedError()
    
    def __setitem__(self, key, value):
        raise NotImplementedError()


settings = Settings(
    os.environ.get('VIDEO_STANDBY_SETTINGS_PATH', DEFAULT_SETTINGS_PATH)
    )
