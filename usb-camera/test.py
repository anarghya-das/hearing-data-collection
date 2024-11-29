import platform
import cv2
import threading
from datetime import datetime
import time


def video_recording(cap, out, lsl_out=None, stop_event=None):
    print("Video Started")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Get the current timestamp
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")
            if lsl_out:
                lsl_out.push_sample([timestamp])
            # Define the text properties
            colour = (0, 255, 0)  # Green color for the text
            origin = (10, 30)  # Top left corner
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 1
            thickness = 2
            # Add the timestamp to the frame
            cv2.putText(frame, timestamp, origin,
                        font, scale, colour, thickness)

            cv2.imshow('frame', frame)
            # Write the frame to the output file
            out.write(frame)

            if stop_event and stop_event.is_set():
                print('Stopping Video')
                break

        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


backend = None
if platform.system() == 'Windows':
    backend = cv2.CAP_DSHOW
# Open a connection to the camera (0 is usually the built-in camera, change if you have multiple cameras)
cap = cv2.VideoCapture(0, backend)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('/Users/anarghya/Developer/research/hearing-experiment/output.mp4',
                      fourcc, fps, (frame_width, frame_height))

stop_video = threading.Event()
video_thread = threading.Thread(
    target=video_recording, args=(cap, out, None, stop_video))
video_thread.start()

time.sleep(10)
stop_video.set()
video_thread.join()
print("Video Stopped")
