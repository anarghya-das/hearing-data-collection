# Hearing Study Data Collection Setup

Experiment and configuration files for PsychoPy incorporating the Lab Streaming Layer (LSL) for data collection.

## Pre-Requisites 
- Setup the Recording Environment by following [this guide](https://docs.google.com/document/d/1NA2v7Z6gLFAqDksrsyBf3V2RNZ6RxAdAVVEvcNDk-yA/edit?usp=sharing)
- Read through the [data collection protocol](https://docs.google.com/document/d/1ouoUjMdvXaoEwp-7u0hbgPy1gG8JQRklimvL0BNJeHc/edit?usp=sharing) for this experiment
- [Updated Setup](https://docs.google.com/document/d/1rhVqhTrDCe6JzxTdRydFx3ycqmDQtdOv-aFe-VhdEWo/edit?usp=sharing)

## Repository Structure

### Experiment Configurations
- `config.yaml` file contains the various configurations for accommodating different modalities in the experiment and handling the timings for various routines in the experiment.
- It currently supports three configurations: `eeg`, `fnirs`, `test`
- Each of the configurations has the following attributes:

```yaml
  sound_files: ["babycry", "chewing" ... ] # List of sound file names to include from the sounds folder
  baseline_duration: 15 # Duration of the baseline routine (in seconds)
  stimuli_repeat: 5 # Number of times each sound stimulus will repeat
  stimulus_dration: 4 # Duration of the stimulus routine (in seconds)
  response_duration: 5 # Duration of the response routine following the stimulus (in seconds)
  rest_range: [6, 8] # Duration range of the rest routine following the response (in seconds)
```
- Check the `config.yaml` file for the actual values for each configuration
  
### PsychoPy Configuration
- Experiment files:
  - `hearing.pyexp`: Experiment file for the fNIRS/EEG modality experiments
  - `pupil.pyexp`: Experiment file for pupil dilation and fNIRS/EEG modality experiments
- Before running the PsychoPy experiment, it expects a few inputs:
  - `participant`: A unique ID (number) for the participant. By default, it generates a random number which can be changed.
  - `run`: Run number for the same participant for the same experiment if multiple recordings are needed (usually 1)
  - `config`: The name of the experimental configuration value defined above.


<details>
  <summary>Raspberry Pi Related Setup</summary>

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
- After the experiment, the video will be saved to the `recordings` folder with the current date and time. Access this folder using a file manager and move the file to the `input` folder for further analysis.

## Steps

- Run the `pi-recorder.py` script on the raspberry pi

## Data Processing 

- Move the raw video (.mp4) from Raspberry Pi to the `input` folder
- Move the experiment data file (ends with `.xdf`) to the `exp_data` folder
- The processing script will load and segment the data based on the markers. It will then save the segmented video files and create an output folder

</details>
