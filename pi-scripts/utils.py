from picamera2.outputs import FfmpegOutput
from picamera2 import MappedArray, Picamera2
from pylsl import StreamInfo, StreamOutlet
from datetime import datetime
from picamera2.encoders import H264Encoder
import cv2
import os


class TimestampCollector(FfmpegOutput):
    def __init__(self, filename, lsl_out=None, timestamp_file=None):
        super().__init__(output_filename=filename, pts=timestamp_file)
        self.lsl_outlet = lsl_out

    def outputframe(self, frame, keyframe=True, timestamp=None):
        if self.recording and self.ffmpeg:
            # Handle the case where the FFmpeg prcoess has gone away for reasons of its own.
            try:
                self.ffmpeg.stdin.write(frame)
                self.ffmpeg.stdin.flush()  # forces every frame to get timestamped individually
            except Exception as e:  # presumably a BrokenPipeError? should we check explicitly?
                self.ffmpeg = None
                if self.error_callback:
                    self.error_callback(e)
            else:
                if timestamp and self.lsl_outlet:
                    self.lsl_outlet.push_sample([timestamp])
                self.outputtimestamp(timestamp)


def initialize_camera(resolution, frame_rate, callback_fn=None, lsl=True, stream_id="pi-vid"):
    picam2 = Picamera2()
    video_config = create_config(picam2, resolution, frame_rate)
    picam2.configure(video_config)
    picam2.pre_callback = callback_fn
    outlet = None
    if lsl:
        outlet = create_stream(frame_rate, stream_id)
    return picam2, outlet


def video_output(filename, outlet=None, encoder_bitrate=10000000):
    encoder = H264Encoder(bitrate=encoder_bitrate)
    output = TimestampCollector(
        f'{filename}.mp4', lsl_out=outlet, timestamp_file=None)
    return encoder, output


def start_camera(picam2, outlet=None, filename=None, output_type='file',
                 stream_cmd="-f rtp udp://<ip-addr>:<port>", root='recordings'):
    try:
        if not filename:
            os.makedirs(root, exist_ok=True)
            now = datetime.now()
            filename = now.strftime("%Y-%m-%d_%H-%M")
        filename = os.path.join(root, filename)
        encoder, output = None, None
        if output_type == 'file':
            encoder, output = video_output(filename, outlet)
        elif output_type == 'stream':
            encoder, output = H264Encoder(), FfmpegOutput(stream_cmd)
        picam2.start_recording(encoder, output)
        print('Recording started...')
        return True
    except Exception as e:
        print(f"Exception in starting recording: {e}")
        return False


def stop_camera(picam2):
    picam2.stop_recording()
    print('Recording stopped...')


def apply_gray_and_time(request):
    colour = (0, 255, 0)  # Black color for the text
    origin = (0, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1
    thickness = 2
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


def create_config(camera, resolution=(640, 480), frame_rate=60):
    frame_val = int(1000000/frame_rate)
    camera_config = camera.create_video_configuration(
        main={"size": resolution, "format": "BGR888"},
        controls={"FrameDurationLimits": (frame_val, frame_val)})
    return camera_config


def create_stream(sampling_rate, stream_id):
    info = StreamInfo(name="VideoFrameStream", type='videostream', nominal_srate=sampling_rate,
                      channel_format='float32', channel_count=1, source_id=stream_id)
    outlet = StreamOutlet(info)
    return outlet
