# Hearing Study Data Collection Setup

## System Setup 

### FFMPEG 

- Make sure [FFMPEG](https://ffmpeg.org/) is installed on both Raspberry Pi and the recording computer
  
> Useful Commands
> - Streaming from Pi: 
> ``` libcamera-vid -t 0 --inline --listen -o tcp://<ip_address>:<port> --width 800 --height 450 --framerate 30 ```
> - Capturing from the computer: 
> ``` ffplay tcp://<ip_address>:<port> -fflags nobuffer -flags low_delay -framedrop ```
> - Converting .h264 to .mp4:
> ``` ffmpeg -i <input_file>.h264 -c copy <output_file>.mp4 ```
> - Splitting a video file by times:
> ``` ffmpeg -i <input_file>.h264 -ss <start_time> -to <end_time> -c:v libx264 -preset fast -crf 18 ```
    >>>>> `-crf` flag ensures the quality (0-51, 0 - lossless, 23 - default, 51 - worst) 


### Raspberry Pi 

- Create a virtual environment with `--system-site-packages` flag
- Activate the virtual environment
- Install [picamera2](https://github.com/raspberrypi/picamera2)
  - Follow the installation instructions under the `Installation using pip` section
- Copy and run the `pi-recorder.py` script to the raspberry pi 
- Once the experiment is finished the video is saved to `final.h264` file. Move this to the `input` folder for further analysis.

### PsychoPy
- TODO
  
### Lab Recorder 
- TODO 
  
## Steps

- Run Lab Recorder
- Run PsychoPy, open `passive.psyexp` file
- Run the `pi-recorder.py` script on the raspberry pi

## Data Processing 

- Move the raw video from raspberry pi to the `input` folder
- Move the experiment data file (ends with `.xdf`) to the `exp_data` folder
- The processing script will load and segment the data based on the markers. It will then save the segmented video files and create an output folder
