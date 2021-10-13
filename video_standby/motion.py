
import sys
import time
from types import MappingProxyType

import cv2
import numpy as np

from video_standby.config import settings
from video_standby.stream import VideoStream
from video_standby.logger import Logger


logger = Logger(settings)


class MotionDetector:
    SENSITIVITY_LOW = 0.04
    SENSITIVITY_MEDIUM = 0.03
    SENSITIVITY_HIGH = 0.02
    sensitivity_labels = MappingProxyType({
        'LOW': SENSITIVITY_LOW,
        'MEDIUM': SENSITIVITY_MEDIUM,
        'HIGH': SENSITIVITY_HIGH,
        })
    
    def __init__(self, width, height, mask, sensitivity, min_duration):
        self.sensitivity = self.sensitivity_labels[sensitivity]
        self.min_duration = min_duration
        self.width = width
        self.height = height
        self.mask = self.process_mask(mask)
        self.area = width * height
        self.min_object_area = int(self.area * self.sensitivity)
        self.skip_frames = max((1, int(10000 * (self.sensitivity ** 2))))
        self.width_dilation = self.get_dilation(self.width, self.sensitivity)
        self.height_dilation = self.get_dilation(self.height, self.sensitivity)
        self.dilation_kernel = np.ones(
            (self.width_dilation, self.height_dilation),
            np.uint8,
            )
    
    @staticmethod
    def get_dilation(length, sensitivity):
        return 2 * (int(length * (.08 - sensitivity)) // 2) + 1
    
    def process_mask(self, mask):
        return mask
        height, width, channels = mask.shape
        if height != self.height or width != self.width:
            mask = cv2.resize(mask, (self.width, self.height))
        
        return mask
    
    def preprocess_frame(self, frame, mask=None):
        frame = cv2.resize(frame, (self.width, self.height))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        return frame
    
    def get_first_frame(self, reader):
        while True:
            # Try getting a sample frame until we have a non-black frame.
            comp_frame = next(reader)[1]
            if comp_frame.any():
                break
        return self.preprocess_frame(comp_frame)
    
    def detect(self, stream, motion_started_handler, motion_ended_handler):
        reader = stream.create_reader(skip_frames=self.skip_frames)
        comp_frame = self.get_first_frame(reader)
        
        recording = False
        latest_motion = 0
        diff = None
        try:
            for index, frame in reader:
                frame = self.preprocess_frame(frame)
                new_diff = cv2.absdiff(frame, comp_frame)
                if diff is None:
                    diff = new_diff
                else:
                    cv2.addWeighted(new_diff, 1.0, diff, .5, .0, diff)
            
                mono_diff = cv2.threshold(diff, 48, 255, cv2.THRESH_BINARY)[1]
                cv2.dilate(
                    mono_diff,
                    self.dilation_kernel,
                    iterations=1,
                    dst=mono_diff
                    )
                contours, _ = cv2.findContours(
                    mono_diff,
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_SIMPLE,
                    )
                
                for contour in contours:
                    if cv2.contourArea(contour) > self.min_object_area:
                        latest_motion = time.time()
                        break
                
                time_since_motion = time.time() - latest_motion
                if time_since_motion < self.min_duration:
                    if not recording:
                        recording = True
                        motion_started_handler()
                elif recording:
                    recording = False
                    motion_ended_handler()
            
                comp_frame = frame
        except KeyboardInterrupt:
            pass

'''
def process_frame(frame):
    frame = cv2.resize(frame, DIMENSIONS)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame


def detect_motion(name):
    writer = cv2.VideoWriter(
        '/home/mike/video_standby/test.mp4',
        int(cv2.VideoWriter_fourcc(*'mp4v')),
        30 / (SKIP_FRAMES + 1),
        DIMENSIONS,
        )
    stream = VideoStream(name)
    reader = stream.create_reader(skip_frames=SKIP_FRAMES)
    while True:
        # Try getting a sample frame until we have something that is not black.
        comp_frame = next(reader)[1]
        if comp_frame.any():
            break
    comp_frame = process_frame(comp_frame)
    
    motion_recently = False
    recording = False
    latest_motion = 0
    diff = None
    try:
        for index, frame in reader:
            frame = process_frame(frame)
            new_diff = cv2.absdiff(frame, comp_frame)
            if diff is None:
                diff = new_diff
            else:
                cv2.addWeighted(new_diff, 1.0, diff, .5, .0, diff)
            
            mono_diff = cv2.threshold(diff, 48, 255, cv2.THRESH_BINARY)[1]
            cv2.dilate(
                mono_diff, DILATION_KERNEL, iterations=1, dst=mono_diff
                )
            contours, _ = cv2.findContours(
                mono_diff,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
                )
            
            large_contours = []
            small_contours = []
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                if contour_area > MIN_OBJECT_AREA:
                    latest_motion = time.time()
                    large_contours.append(contour)
                else:
                    small_contours.append(contour)
            
            motion_recently = time.time() - latest_motion < MIN_RECORD_TIME
            
            output_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            if motion_recently:
                if not recording:
                    recording = True
                    print('Start recording')
                cv2.putText(
                    output_frame,
                    'RECORDING',
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                    )
            elif recording is True:
                recording = False
                print('Stop recording')
            
            if large_contours:
                print(f'Detected motion at index {index}.')
                cv2.drawContours(
                    output_frame,
                    large_contours,
                    -1,
                    (255,0,0),
                    3,
                    )

            if small_contours:
                cv2.drawContours(
                    output_frame,
                    small_contours,
                    -1,
                    (0, 255, 0),
                    1,
                    )
            
            comp_frame = frame
            writer.write(output_frame)
    except KeyboardInterrupt:
        pass
    
    print('Releasing writer')
    writer.release()
'''

if __name__ == '__main__':
    name = sys.argv[1]
    stream = VideoStream(name)
    motion_settings = stream.settings.motion_detection
    detector = MotionDetector(
        640,
        480,
        None,
        motion_settings.sensitivity,
        motion_settings.min_recording_time,
        )
    
    def handle_motion_started():
        logger.info('Motion detected! Start recording!')
        
    def handle_motion_ended():
        logger.info(
            f'No motion detected for {motion_settings.min_recording_time} '
            'seconds. Stop!'
            )
    
    detector.detect(stream, handle_motion_started, handle_motion_ended)
