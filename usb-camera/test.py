import platform
import cv2
from datetime import datetime
from pylsl import StreamInfo, StreamOutlet

backend = None
if platform.system() == 'Windows':
    backend = cv2.CAP_DSHOW
# Open a connection to the camera (0 is usually the built-in camera, change if you have multiple cameras)
cap = cv2.VideoCapture(1, backend)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

video_info = StreamInfo(name="VideoTimeStream", type='videostream', nominal_srate=fps,
                        channel_format='string', channel_count=1, source_id='usb-vid')

lsl_out = StreamOutlet(video_info)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4',
                      fourcc, fps, (frame_width, frame_height))

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

        # cv2.imshow('frame', frame)
        # Write the frame to the output file
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
out.release()
# cv2.destroyAllWindows()
