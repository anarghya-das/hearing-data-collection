from picamera2 import Picamera2
import cv2
from picamera2 import MappedArray, Picamera2
from picamera2.encoders import H264Encoder
from pylsl import StreamInfo, StreamOutlet
import uuid
from datetime import datetime
from threading import Thread
import socket
import queue
import time


def apply_timestamp(request):
    colour = (0, 0, 0)  # Black color for the text
    origin = (0, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5  # Reduced size of the text
    thickness = 1  # Reduced thickness
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")

    with MappedArray(request, "main") as m:
        # Convert the BGR image to grayscale directly
        gray_image = cv2.cvtColor(m.array, cv2.COLOR_BGR2GRAY)
        # Add the timestamp directly to the grayscale image
        cv2.putText(gray_image, timestamp, origin,
                    font, scale, colour, thickness)
        # Copy the modified grayscale image back to the original array as a 3-channel grayscale image
        m.array[:, :, 0] = gray_image
        m.array[:, :, 1] = gray_image
        m.array[:, :, 2] = gray_image


def create_stream():
    info = StreamInfo(name="VideoFrameStream", type='videostream',
                      channel_format='float32', channel_count=1, source_id=str(uuid.uuid4()))
    outlet = StreamOutlet(info)
    return outlet


def create_camera(callback):
    frame_rate = 60
    frame_val = int(1000000/frame_rate)
    print(f"Frame rate val: {frame_val}")
    picam2 = Picamera2()
    camera_config = picam2.create_video_configuration(main={"size": (
        640, 480), "format": "BGR888"}, controls={"FrameDurationLimits": (frame_val, frame_val)})
    picam2.configure(camera_config)
    picam2.pre_callback = callback
    return picam2


def start_camera(picam2):
    now = datetime.now()
    filename = now.strftime("%Y-%m-%d_%H-%M")
    name = f'{filename}.h264'
    encoder = H264Encoder()  # Adjust bitrate as needed
    picam2.start_recording(encoder, name)
    print('Recording started...')
    return True


def stop_camera(picam2):
    picam2.stop_recording()
    print('Recording stopped...')


def start_recording(q, timestamp_func):
    outlet = create_stream()
    camera = create_camera(timestamp_func)

    frameCounter = 0
    started = start_camera(camera)
    # frame_rate = 60

    while True:
        try:
            # time.sleep(1 / frame_rate)
            message = q.get(block=False)
            if message is None:
                if started:
                    stop_camera(camera)
                break

            _ = camera.capture_metadata()
            frameCounter += 1
            outlet.push_sample([frameCounter])

        except queue.Empty:
            continue


def handle_client(client_socket, timestamp_func):
    q = queue.Queue()
    camera_thread = Thread(target=start_recording, args=(q, timestamp_func))
    thread_started = False

    while True:
        message = client_socket.recv(1024).decode('utf-8')
        if message.lower() == "start":
            print("Received 'start' message")
            camera_thread.start()
            thread_started = True
        elif message.lower() == "stop":
            print("Received 'stop' message")
            if thread_started:
                q.put(None)
            break

    client_socket.close()
    camera_thread.join()


if __name__ == "__main__":
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_socket.bind(("", 65432))  # binds it to the hostname pi1
    server_socket.listen()

    try:
        while True:
            print(f"Waiting for experiment to start...")
            client_socket, address = server_socket.accept()
            print(f"Connection from {address} has been established!")
            handle_client(client_socket, apply_timestamp)

    except KeyboardInterrupt as k:
        print('User terminated.')
    finally:
        server_socket.close()
        print('socket closed')
