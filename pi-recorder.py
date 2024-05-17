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


def get_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Connect to a non-routable IP address
        s.connect(('10.254.254.254', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


def apply_timestamp(request):
    colour = (0, 255, 0)
    origin = (0, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 2
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    with MappedArray(request, "main") as m:
        cv2.putText(m.array, timestamp, origin, font, scale, colour, thickness)


def create_stream():
    info = StreamInfo(name="VideoFrameStream", type='videostream',
                      channel_format='float32', channel_count=1, source_id=str(uuid.uuid4()))
    outlet = StreamOutlet(info)
    return outlet


def create_camera(callback):
    picam2 = Picamera2()
    camera_config = picam2.create_video_configuration(
        main={"size": (1920, 1080)})
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

    while True:
        try:
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
    ip_addr = get_ip_address()
    print(f"IP address of Pi: {ip_addr}")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_socket.bind(('0.0.0.0', 65432))
    server_socket.listen()
    print(f"Waiting for experiment to start...")

    try:
        while True:
            client_socket, address = server_socket.accept()
            print(f"Connection from {address} has been established!")
            handle_client(client_socket, apply_timestamp)

    except KeyboardInterrupt as k:
        print('User terminated.')
    finally:
        server_socket.close()
        print('socket closed')
