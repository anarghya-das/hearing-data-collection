import cv2
import signal
import sys
import platform
import threading
from pylsl import StreamInfo, StreamOutlet
from datetime import datetime
import serial
import csv
import random
import time
import argparse


class VideoRecorder:
    def __init__(self, cam_id=0, output_path='output.avi', default_fps=30, display_video=False, enable_lsl=False):
        print("VideoRecorder start.")
        self.cam_id = cam_id
        self.output_path = output_path
        self.display_video = display_video
        self.enable_lsl = enable_lsl
        self.cap = None
        self.out = None
        self.video_outlet = None
        self.default_fps = default_fps
        self.stop_event = threading.Event()
        print("VideoRecorder initialized successfully.")

    def signal_handler(self, sig, frame):
        print('Termination signal received. Releasing resources...')
        self.stop()
        sys.exit(0)

    def stop(self):
        self.stop_event.set()

    def release_resources(self):
        if self.cap:
            self.cap.release()
        if self.out:
            self.out.release()
        cv2.destroyAllWindows()

    def record_video(self):
        backend = cv2.CAP_DSHOW if platform.system() == 'Windows' else None
        self.cap = cv2.VideoCapture(self.cam_id, backend)

        if not self.cap.isOpened():
            print("Error opening video stream")
            return

        fps = self.cap.get(cv2.CAP_PROP_FPS) or self.default_fps
        print(f"fps: {fps}")
        if self.enable_lsl:
            info = StreamInfo('VideoStream', 'Video', 1, fps,
                              'float32', 'videouid34234')
            self.video_outlet = StreamOutlet(info)

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(
            self.output_path, fourcc, fps, (frame_width, frame_height))

        frame_number = 0
        try:
            while self.cap.isOpened() and not self.stop_event.is_set():
                ret, frame = self.cap.read()
                if ret:
                    now = datetime.now()
                    timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")
                    if self.video_outlet:
                        self.video_outlet.push_sample([frame_number])

                    colour = (0, 255, 0)
                    origin = (10, 30)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 1
                    thickness = 2

                    cv2.putText(frame, timestamp, origin,
                                font, scale, colour, thickness)
                    self.out.write(frame)
                    frame_number += 1
                    if self.display_video:
                        cv2.imshow('frame', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("No frame received.")
                    break
        finally:
            self.release_resources()


class PPGRecorder:
    def __init__(self, port, baud_rate=115200, enable_lsl=True, simulate_data=False):
        self.port = port
        self.baud_rate = baud_rate
        self.enable_lsl = enable_lsl
        self.simulate_data = simulate_data
        self.ser = None
        self.stop_event = threading.Event()
        self.sample_rate = 50

        if self.enable_lsl:
            info = StreamInfo('PPGStream', 'PPG', 1, self.sample_rate,
                              'float32', 'ppguid34234')
            self.ppg_outlet = StreamOutlet(info)

    def signal_handler(self, sig, frame):
        print('Termination signal received. Releasing resources...')
        self.stop()
        sys.exit(0)

    def start(self):
        if self.simulate_data:
            self.simulate_ppg_data()
        else:
            try:
                self.ser = serial.Serial(self.port, self.baud_rate, timeout=1)
                time.sleep(2)  # Wait for the connection to be established
                print(f"Successfully connected to {self.port}")

                self.record_ppg()
            except serial.SerialException as e:
                print(f"Serial error: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

    def stop(self):
        self.stop_event.set()

    def record_ppg(self):
        with open('pulse_data.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Datetime', 'Timestamp', 'Signal'])

            print("Logging data. Press 'q' to stop.")
            while not self.stop_event.is_set():
                signal = self.read_signal()
                if signal:
                    timestamp = time.time()
                    datetime_str = datetime.fromtimestamp(
                        timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')
                    csvwriter.writerow([datetime_str, timestamp, signal])

                    if self.enable_lsl:
                        self.ppg_outlet.push_sample([float(signal)])
                else:
                    print("Missed a signal")

    def read_signal(self):
        try:
            signal = self.ser.readline().decode('utf-8').strip()
            split_signal = signal.split(" ")
            value = None
            if len(split_signal) == 2:
                value = split_signal[1]
            return value
        except Exception as e:
            print(f"Error reading signal: {e}")
            return None

    def simulate_ppg_data(self):
        print("Simulating PPG data. Press 'q' to stop.")
        while not self.stop_event.is_set():
            signal = random.uniform(0.0, 1.0)  # Simulate a random PPG signal

            if self.enable_lsl:
                self.ppg_outlet.push_sample([signal])

            time.sleep(0.02)  # Simulate a 50 Hz signal

    def release_resources(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"Closed connection to {self.port}")


def main():
    parser = argparse.ArgumentParser(description="Record video, PPG, or both.")
    parser.add_argument('--enable-lsl', action='store_true', help="Enable LSL")
    parser.add_argument('--recording-time', type=int, default=5,
                        help="Recording time in seconds")
    parser.add_argument(
        '--record-video', action='store_true', help="Record video")
    parser.add_argument('--cam-id', type=int, default=1, help="Camera ID")
    parser.add_argument('--display-video', action='store_true',
                        help="Display video while recording") # display video doesn't work on mac for some reason since it is running on a different thread

    parser.add_argument('--record-ppg', action='store_true', help="Record PPG")
    # Additional arguments for PPG recording
    parser.add_argument('--ppg-port', type=str,
                        default="COM5", help="PPG device port")
    parser.add_argument('--simulate-ppg', action='store_true',
                        help="Simulate PPG data")

    args = parser.parse_args()
    video_recorder = None
    video_process = None
    ppg_recorder = None
    ppg_thread = None

    if args.record_video:
        video_recorder = VideoRecorder(
            cam_id=args.cam_id, output_path='video.avi', display_video=args.display_video, enable_lsl=args.enable_lsl)
        signal.signal(signal.SIGINT, video_recorder.signal_handler)
        signal.signal(signal.SIGTERM, video_recorder.signal_handler)

        video_process = threading.Thread(
            target=video_recorder.record_video)
        video_process.start()

    if args.record_ppg:
        ppg_recorder = PPGRecorder(
            port=args.ppg_port, enable_lsl=args.enable_lsl, simulate_data=args.simulate_ppg)
        signal.signal(signal.SIGINT, ppg_recorder.signal_handler)
        signal.signal(signal.SIGTERM, ppg_recorder.signal_handler)

        ppg_thread = threading.Thread(target=ppg_recorder.start)
        ppg_thread.start()

    # Example of stopping the recorder after 20 seconds
    try:
        time.sleep(args.recording_time)
        if args.record_video:
            video_recorder.stop()
        if args.record_ppg:
            ppg_recorder.stop()
    except KeyboardInterrupt:
        if args.record_video:
            video_recorder.stop()
        if args.record_ppg:
            ppg_recorder.stop()


if __name__ == "__main__":
    main()
