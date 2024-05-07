from picamera2 import Picamera2
import time
import cv2
from picamera2 import MappedArray, Picamera2
from picamera2.encoders import H264Encoder
from pylsl import StreamInfo, StreamOutlet
import uuid
from datetime import datetime

colour = (0, 255, 0)
origin = (0, 30)
font = cv2.FONT_HERSHEY_SIMPLEX
scale = 1
thickness = 2


def apply_timestamp(request):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    with MappedArray(request, "main") as m:
        gray = cv2.cvtColor(m.array, cv2.COLOR_BGR2GRAY)
        cv2.putText(gray, timestamp, origin, font, scale, colour, thickness)


info = StreamInfo(name="VideoFrameStream", type='videostream',
                  channel_format='float32', channel_count=1, source_id=str(uuid.uuid4()))
outlet = StreamOutlet(info)

picam2 = Picamera2()
camera_config = picam2.create_video_configuration(main={"size": (1920, 1080)})
picam2.configure(camera_config)
picam2.pre_callback = apply_timestamp

encoder = H264Encoder()  # Adjust bitrate as needed
picam2.start_encoder(encoder, "final.h264")

frameCounter = 1

picam2.start()
print('Recording started...')

try:
    while True:
        metadata = picam2.capture_metadata()
        frameCounter += 1
        outlet.push_sample([frameCounter])
except KeyboardInterrupt:
    print("Interrupted by user")
finally:
    picam2.stop()
    picam2.close()
    print('Saved output video to:', "final.h264")

# time_split = frame_number/frame_rate
# ffmpeg -i Camera0_20220405_1130.avi -ss 10 -to 20 -c copy output_segment.avi
