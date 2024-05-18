import time
from utils import initialize_camera, apply_gray_and_time, start_camera, stop_camera
import queue


def stream_video(resolution, frame_rate, callback=None, message_queue=None):
    camera, _ = initialize_camera(
        resolution, frame_rate, callback_fn=callback, lsl=False)
    start_camera(camera, output_type='stream',
                 stream_cmd="-f rtp udp://<ip-addr>:9000")
    camera_stopped = False
    try:
        while True:
            try:
                time.sleep(1 / frame_rate)
                if message_queue:
                    message = message_queue.get(block=False)
                    if message is None:
                        stop_camera(camera)
                        camera_stopped = True
                        break
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("User ended the recording.")
    finally:
        if not camera_stopped:
            stop_camera(camera)


if __name__ == "__main__":
    resolution = (1920, 1080)
    frame_rate = 30
    stream_video(resolution, frame_rate, apply_gray_and_time)

# sudo ip link set wlan0 down
# sudo ip link set wlan0 up
# 