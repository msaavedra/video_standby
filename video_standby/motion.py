
import sys
import time

import cv2
import numpy as np

from video_standby.config import settings
from video_standby.stream import VideoStream

WIDTH = 640
HEIGHT = 480
DIMENSIONS = (WIDTH, HEIGHT)
AREA = WIDTH * HEIGHT
MIN_RECORD_TIME = 5

SENSITIVITY = 'low'
if SENSITIVITY == 'low':
    BASE_PERCENTAGE = 0.04
elif SENSITIVITY == 'medium':
    BASE_PERCENTAGE = 0.03
elif SENSITIVITY == 'high':
    BASE_PERCENTAGE = 0.02
else:
    raise ValueError('Invalid sensitivity')

MIN_OBJECT_AREA = int(AREA * BASE_PERCENTAGE)
WIDTH_DILATION = 2 * (int(WIDTH * (.08 - BASE_PERCENTAGE)) // 2) + 1
HEIGHT_DILATION = 2 * (int(HEIGHT * (.08 - BASE_PERCENTAGE)) // 2) + 1
DILATION_KERNEL = np.ones((WIDTH_DILATION, HEIGHT_DILATION), np.uint8)
SKIP_FRAMES = max((1, int(10000 * (BASE_PERCENTAGE ** 2))))


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


if __name__ == '__main__':
    name = sys.argv[1]
    detect_motion(name)
