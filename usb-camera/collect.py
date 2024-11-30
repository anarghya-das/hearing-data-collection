import cv2
import signal
import sys
import platform
from multiprocessing import Process, Event
from pylsl import StreamInfo, StreamOutlet
from datetime import datetime


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
        self.stop_event = Event()
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

        fps = self.cap.get(cv2.CAP_PROP_FPS) or self.default_fps
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
                    break
        finally:
            self.release_resources()


if __name__ == "__main__":
    recorder = VideoRecorder(
        cam_id=1, output_path='video.avi', display_video=False, enable_lsl=True)
    signal.signal(signal.SIGINT, recorder.signal_handler)
    signal.signal(signal.SIGTERM, recorder.signal_handler)

    # Create and start a process to run the record_video method
    recording_process = Process(target=recorder.record_video)
    recording_process.start()

    # Example of stopping the recorder after 10 seconds
    try:
        import time
        time.sleep(5)
        recorder.stop()
        # recording_process.join(timeout=10)
    except KeyboardInterrupt:
        recorder.stop()
        recording_process.join()
