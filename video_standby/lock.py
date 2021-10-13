
from contextlib import contextmanager
from multiprocessing import Lock
from multiprocessing.sharedctypes import RawValue

from .config import settings
from .logger import Logger


logger = Logger(settings)


class SharedExclusiveLock:
    
    def __init__(self):
        self.exclusive_lock = Lock()
        self.shared_counter_lock = Lock()
        self.shared_counter = RawValue('B', 0)
    
    def acquire_shared(self):
        logger.trace('Attempting to acquire lock in shared mode...')
        with self.shared_counter_lock:
            self.shared_counter.value += 1
            if self.shared_counter.value == 1:
                self.exclusive_lock.acquire()
        logger.trace('Acquired lock in shared mode.')
    
    def release_shared(self):
        with self.shared_counter_lock:
            if self.shared_counter.value < 1:
                raise ValueError('Cannot decrement shared counter below zero.')
            self.shared_counter.value -= 1
            if self.shared_counter.value == 0:
                self.exclusive_lock.release()
        logger.trace('Released lock in shared mode.')
    
    @property
    @contextmanager
    def shared(self):
        self.acquire_shared()
        try:
            yield self
        finally:
            self.release_shared()
    
    def acquire_exclusive(self):
        logger.trace('Attempting to acquire lock in exclusive mode...')
        self.shared_counter_lock.acquire()
        self.exclusive_lock.acquire()
        logger.trace('Acquired lock in exclusive mode.')
    
    def release_exclusive(self):
        self.exclusive_lock.release()
        self.shared_counter_lock.release()
        logger.trace('Released lock in exclusive mode.')
    
    @property
    @contextmanager
    def exclusive(self):
        self.acquire_exclusive()
        try:
            yield self
        finally:
            self.release_exclusive()
