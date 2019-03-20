
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
        'ipc_path': DEFAULT_IPC_PATH,
        'log_level': 'INFO',
        'log_format': (
            '%(asctime)s: '
            '%(module)s.%(name)s:%(lineno)s'
            ' - %(message)s'
            )
        }
    default_source_settings = {}
    
    def __init__(self, path):
        self.path = path
        self._settings = self._load()
        self.mtime = 0
    
    def __getitem__(self, key):
        return self._settings[key]
    
    def reload_on_update(self):
        return False  # FIXME: add reloading support
    
    def _load(self):
        try:
            with open(self.path, 'r') as f:
                file_settings = yaml.load(f.read())
        except Exception as e:
            sys.stderr.write('Failed opening settings file %s. Reason: %s' % (
                self.path,
                e,
                ))
            file_settings = {}
        
        global_settings = self._deep_merge(
            self.default_global_settings,
            file_settings.get('global', {})
            )
        sources = {
            name: self._deep_merge(self.default_source_settings, source)
            for (name, source) in file_settings.get('sources', {}).items()
            }
        return {
            'globals': global_settings,
            'sources': sources
            }
    
    def _deep_merge(self, base, updates):
        base = base.copy()
        
        for key, value in updates.items():
            if key not in base:
                continue
            if isinstance(value, dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
    
        return base
    

settings = Settings(
    sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SETTINGS_PATH
    )
