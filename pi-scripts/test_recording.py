import time
from utils import initialize_camera, apply_gray_and_time, start_camera, stop_camera
from pprint import *
import sys

recording_duration = 5
try:
    recording_duration = int(sys.argv[1])
except IndexError:
    print('No duration provided, starting 5 seconds recording!')

camera, _ = initialize_camera(
    (640, 480), 60, callback_fn=apply_gray_and_time, lsl=False)
pprint(camera.sensor_modes)
start_camera(camera)
time.sleep(recording_duration)
stop_camera(camera)
