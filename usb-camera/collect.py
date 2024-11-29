import cv2
import time
from datetime import datetime
import platform

backend = None
if platform.system() == 'Windows':
    backend = cv2.CAP_DSHOW
# Open a connection to the camera (0 is usually the built-in camera, change if you have multiple cameras)
cap = cv2.VideoCapture(0, backend)

# Get the frame rate
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frame rate: {fps} FPS")

# Define the codec and create VideoWriter object
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_width, frame_height)
# Alternatively, you can use 'avc1' or 'H264'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, fps, (frame_width, frame_height))

# Calculate the expected frame duration
expected_frame_duration = 1.0 / fps

previous_time = time.time()
timestamps = []
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Get the current timestamp
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        timestamps.append(timestamp)
        # Define the text properties
        colour = (0, 255, 0)  # Green color for the text
        origin = (10, 30)  # Top left corner
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        thickness = 2

        # Add the timestamp to the frame
        cv2.putText(frame, timestamp, origin, font, scale, colour, thickness)

        # Write the frame to the output file
        out.write(frame)

        # Display the frame
        cv2.imshow('frame', frame)

        # Check the actual frame duration
        current_time = time.time()
        actual_frame_duration = current_time - previous_time
        previous_time = current_time

        # Calculate the actual FPS
        actual_fps = 1.0 / actual_frame_duration if actual_frame_duration > 0 else 0

        # Print the actual frame duration and FPS, and compare with expected duration
        print(
            f"Actual frame duration: {actual_frame_duration:.4f} seconds (Expected: {expected_frame_duration:.4f} seconds)")
        print(f"Actual FPS: {actual_fps:.2f} FPS (Expected: {fps:.2f} FPS)")

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
