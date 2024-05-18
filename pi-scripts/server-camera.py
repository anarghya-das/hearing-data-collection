from threading import Thread
import socket
import queue
import time
from utils import initialize_camera, start_camera, apply_gray_and_time, stop_camera
from stream import stream_video


def start_recording(resolution, frame_rate, timestamp_func, q):
    camera, stream_outlet = initialize_camera(
        resolution, frame_rate, timestamp_func)
    started = start_camera(camera, outlet=stream_outlet)

    while True:
        try:
            time.sleep(1 / frame_rate)
            message = q.get(block=False)
            if message is None:
                if started:
                    stop_camera(camera)
                break

        except queue.Empty:
            continue


def handle_client(client_socket, resolution, frame_rate, timestamp_func):
    q = queue.Queue()
    camera_thread = Thread(target=start_recording, args=(
        resolution, frame_rate, timestamp_func, q))
    streaming_thread = Thread(target=stream_video, args=(
        resolution, frame_rate, timestamp_func, q))

    thread_camera_started = False
    thread_streaming_started = False
    try:
        while True:
            message = client_socket.recv(1024).decode('utf-8')
            if message.lower() == "start":
                print("Received 'start' message")
                camera_thread.start()
                thread_camera_started = True
            elif message.lower() == "stream":
                print("Received 'stream' message")
                streaming_thread.start()
                thread_streaming_started = True
            elif message.lower() == "stop":
                print("Received 'stop' message")
                if thread_camera_started or thread_streaming_started:
                    q.put(None)
                break
        if thread_streaming_started:
            streaming_thread.join()
        if thread_camera_started:
            camera_thread.join()
    except Exception as e:
        print(f"Some error while handling client request: {e}")
    finally:
        client_socket.close()


if __name__ == "__main__":
    resolution = (1920, 1080)
    frame_rate = 30

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_socket.bind(("", 65432))  # binds it to the hostname pi1
    server_socket.listen()

    try:
     # TODO add while loop to listen to more connections, problems in closing the picamera
        print(f"Waiting for experiment to start...")
        client_socket, address = server_socket.accept()
        print(f"Connection from {address} has been established!")
        handle_client(client_socket, resolution,
                      frame_rate, apply_gray_and_time)
    except KeyboardInterrupt as k:
        print('User terminated.')
    finally:
        server_socket.close()
        print('socket closed')
