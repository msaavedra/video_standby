
import sys
import time

import cv2
import numpy as np

from video_standby.config import settings
from video_standby.stream import VideoStream

WIDTH = 480
HEIGHT = 320
DIMENSIONS = (WIDTH, HEIGHT)
AREA = WIDTH * HEIGHT

DILATION_KERNEL = np.ones((23,23), np.uint8)

def process_frame(frame):
    frame = cv2.resize(frame, DIMENSIONS)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame


def detect_motion(name):
    writer = cv2.VideoWriter(
        '/home/mike/video_standby/test.mp4',
        int(cv2.VideoWriter_fourcc(*'mp4v')),
        3.0,
        DIMENSIONS,
        )
    stream = VideoStream(name)
    reader = stream.create_reader(skip_frames=9)
    while True:
        # Try getting a sample frame until we have something that is not black.
        comp_frame = next(reader)[1]
        if comp_frame.any():
            break
    comp_frame = process_frame(comp_frame)
    
    recording = False
    latest_motion = 0
    diff = None
    try:
        for index, frame in reader:
            frame = process_frame(frame)
            new_diff = cv2.absdiff(frame, comp_frame)
            #cv2.threshold(new_diff, 64, 255, cv2.THRESH_BINARY, new_diff)
            #new_diff = cv2.dilate(new_diff, None, iterations=8)
            #cv2.merge((new_diff, new_diff, new_diff, new_diff))
            if diff is None:
                diff = new_diff
            else:
                cv2.addWeighted(new_diff, 1.0, diff, .5, .0, diff)
            
            thresh_diff = cv2.threshold(diff, 48, 255, cv2.THRESH_BINARY)[1]
            cv2.dilate(thresh_diff, DILATION_KERNEL, iterations=1, dst=thresh_diff)
            contours, _ = cv2.findContours(
                thresh_diff,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE,
                )
            
            contours_to_draw = []
            contours_too_small = []
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                contour_percentage = min(((contour_area / AREA), 1.0))
                if contour_percentage > .025:
                    print('Found large area of motion')
                    latest_motion = time.time()
                    contours_to_draw.append(contour)
                else:
                    contours_too_small.append(contour)

            output_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            if (time.time() - latest_motion) < 15:
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
            
            if contours_to_draw:
                print('drawing contours')
                cv2.drawContours(
                    output_frame,
                    contours_to_draw,
                    -1,
                    (255,0,0),
                    3,
                    )

            if contours_too_small:
                print('drawing small contours')
                cv2.drawContours(
                    output_frame,
                    contours_too_small,
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
