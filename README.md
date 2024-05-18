# Hearing Study Data Collection Setup

## System Setup 

### FFMPEG 

- Make sure [FFMPEG](https://ffmpeg.org/) is installed on both Raspberry Pi and the recording computer

### GStreamer

- Install [GStreamer](https://gstreamer.freedesktop.org/download/) to stream video from Raspberry Pi to the recording computer
- For Windows, make sure to select complete installation (not typical) when installing
  
> Useful Commands
> - Streaming from Pi (update with hostname or ip-address of the computer): 
> ``` python stream.py ```
> - Capturing from the computer (Replace <port_num> with actual port number): 
> ``` gst-launch-1.0 -v udpsrc port=<port_num> caps="application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264" ! rtpjitterbuffer ! rtph264depay ! avdec_h264 ! videoconvert ! fpsdisplaysink video-sink=autovideosink ```
> - Converting .h264 to .mp4:
> ``` ffmpeg -i <input_file>.h264 -c copy <output_file>.mp4 ```
> - Converting .mjpeg to .mp4:
> ``` ffmpeg -i <input_file>.mjpeg -pix_fmt yuv420p -c:v libx264 -crf 20 -an <output_file>.mp4 ```
> - Splitting a video file by times:
> ``` ffmpeg -i <input_file>.h264 -ss <start_time> -to <end_time> -c:v libx264 -preset fast -crf 18 ```
>> - `-crf` flag ensures the quality (0-51, 0 - lossless, 23 - default, 51 - worst) 


### Raspberry Pi 

- Create a virtual environment with `--system-site-packages` flag
- Activate the virtual environment
- Install [pylsl](https://github.com/chkothe/pylsl/tree/master)
  - [Examples](https://github.com/chkothe/pylsl/tree/master/examples)
- Install [picamera2](https://github.com/raspberrypi/picamera2)
  - Follow the installation instructions under the `Installation using pip` section
- Copy and run the `server-camera.py` script on the Raspberry Pi 
- Once the experiment is finished the video is saved to the `recordings` folder with the current date and time. Access this folder using a file manager and move the file to the `input` folder for further analysis.

### PsychoPy
- [Sound Latency](https://psychopy.org/api/sound/playback.html)
  
### Lab Recorder 
- [Installation](https://github.com/labstreaminglayer/App-LabRecorder)
  
## Steps

- Run Lab Recorder
- Run PsychoPy, open `passive.psyexp` file
- Run the `pi-recorder.py` script on the raspberry pi

## Data Processing 

- Move the raw video (.mp4) from Raspberry Pi to the `input` folder
- Move the experiment data file (ends with `.xdf`) to the `exp_data` folder
- The processing script will load and segment the data based on the markers. It will then save the segmented video files and create an output folder
