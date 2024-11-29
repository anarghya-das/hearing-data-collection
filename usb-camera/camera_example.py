import cv2
from datetime import datetime


def record_and_display_specific_frames(output_video_path):
    # IMPORTANT: cv2.CAP_DSHOW is required
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps,
                          (frame_width, frame_height))

    print("Recording started. ")

    with open(output_video_path.replace(".avi", "_timestamps.txt"), "w") as timestamp_file:
        frame_count = 0
        # specific_frames = [500, 1000, 2000,10000]
        captured_frames = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            timestamp_file.write(f"{timestamp}\n")
            # text only for verification
            # cv2.putText(frame, timestamp, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            out.write(frame)

            # if frame_count in specific_frames:
            #     captured_frames[frame_count] = (frame.copy(), timestamp)
            #     print(f"Frame {frame_count}: {timestamp}")

            cv2.imshow('Recording', frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q') or frame_count >= 1000:
                break

            # if frame_count > max(specific_frames):
            #     break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Recording stopped.")

    # for frame_num, (frame, timestamp) in captured_frames.items():
    #     print(f"Displaying Frame {frame_num} with timestamp: {timestamp}")
    #     cv2.putText(frame, timestamp, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    #     cv2.imshow(f"Frame {frame_num}", frame)
    #     cv2.waitKey(0)

    # cv2.destroyAllWindows()


record_and_display_specific_frames("camera_recording_with_timestamps.avi")
