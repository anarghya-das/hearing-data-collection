#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on December 06, 2024, at 13:26
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# Run 'Before Experiment' code from code_init
import os
import random
from pylsl import StreamInfo, StreamOutlet
import socket
from psychopy import sound, core
import yaml
import cv2
import signal
import sys
import threading
from datetime import datetime
import platform
import threading
import serial
import csv
import random
import time
import atexit
import pandas as pd

started_recording = False
original_quit = core.quit
lsl_socket, recorder, ppg_recorder, recording_thread, ppg_thread = None, None, None, None, None

def cleanup():
    global started_recording, lsl_socket, recorder, ppg_recorder, recording_thread, ppg_thread
    print("At exit...")
    if started_recording:
        print("Performing cleanup tasks...")
        if recorder:
            recorder.stop()
            recording_thread.join()
        if ppg_recorder:
            ppg_recorder.stop()
            ppg_thread.join()
        started_recording = False

def custom_quit():
    cleanup()  # Your cleanup function
    original_quit()

def get_audio_files(folder):
    return [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.wav') or file.endswith('.mp3')]

class VideoRecorder:
    def __init__(self, cam_id=0, output_name='output', default_fps=30, display_video=False, enable_lsl=False):
        print("VideoRecorder start.")
        self.cam_id = cam_id
        self.output_path = os.path.join(os.getcwd(),"input", f"{output_name}.avi")
        self.display_video = display_video
        self.enable_lsl = enable_lsl
        self.cap = None
        self.out = None
        self.video_outlet = None
        self.default_fps = default_fps
        self.stop_event = threading.Event()
        if self.enable_lsl:
            info = StreamInfo('VideoStream', 'Video', 1, self.default_fps,
                              'float32', 'videouid34234')
            self.video_outlet = StreamOutlet(info)
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
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.out = cv2.VideoWriter(
            self.output_path, fourcc, self.default_fps, (frame_width, frame_height))

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

class PPGRecorder:
    def __init__(self, port, baud_rate=9600, enable_lsl=True, simulate_data=False):
        self.port = port
        self.baud_rate = baud_rate
        self.enable_lsl = enable_lsl
        self.simulate_data = simulate_data
        self.ser = None
        self.stop_event = threading.Event()
        self.sample_rate = 50 # TODO update with actual sampling rate

        if self.enable_lsl:
            info = StreamInfo('PPGStream', 'PPG', 1, self.sample_rate,
                              'float32', 'ppguid34234')
            self.ppg_outlet = StreamOutlet(info)

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
                    print("No signal received.")

    def read_signal(self):
        try:
            signal = self.ser.readline().decode('utf-8').strip()
            return signal
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

core.quit = custom_quit
atexit.register(cleanup)
marker_info = StreamInfo('HearingMarkerStream', 'Markers',
                         1, 0, 'string', 'hearingid2023')

config_file = open('config.yaml')
yaml_config = yaml.safe_load(config_file)
marker_outlet = StreamOutlet(marker_info)

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'hearing'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'run': '1',
    'config': 'hearing',
    'enable_video': 'true',
    'enable_ppg': 'true',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_winSize = [1512, 982]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s/%s_%s_%s' % (expInfo['participant'],expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='D:\\projects\\hearing-data-collection\\hearing.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('exp')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=True, allowStencil=False,
            monitor='testMonitor', color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [-1.0000, -1.0000, -1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    # create speaker 'hlt_sound'
    deviceManager.addDevice(
        deviceName='hlt_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'let_sound'
    deviceManager.addDevice(
        deviceName='let_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # create speaker 'current_sound'
    deviceManager.addDevice(
        deviceName='current_sound',
        deviceClass='psychopy.hardware.speaker.SpeakerDevice',
        index=-1
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome" ---
    # Run 'Begin Experiment' code from code_init
    config_type = expInfo['config']
    enable_video = expInfo['enable_video'] == 'true'
    enable_ppg = expInfo['enable_ppg'] == 'true'
    exp_config = yaml_config[config_type]
    
    stim_root = os.path.join(os.getcwd(),"stimuli") 
    # Pupil Muscular Test (PMT) variables
    pmt_prestim_duration = exp_config['pmt_prestim']
    pmt_stim_duration = exp_config['pmt_stim']
    pmt_poststim_duration = exp_config['pmt_poststim']
    pmt_trials = exp_config['pmt_trials']
    # Hearing Loudness Test (HLT) variables
    hlt_prestim_duration = exp_config['hlt_prestim']
    hlt_stim_duration = exp_config['hlt_stim']
    hlt_poststim_duration = exp_config['hlt_poststim']
    hlt_repeats = exp_config['hlt_trials']
    hlt_folder = os.path.join(stim_root,"hlt")
    hlt_stims = get_audio_files(hlt_folder)
    hlt_stims = hlt_stims * hlt_repeats
    hlt_trials = len(hlt_stims)
    hlt_idx = 0
    # Listening Effort Test (LET) variables
    let_prestim_duration = exp_config['let_prestim']
    let_stim_duration = exp_config['let_stim']
    let_poststim_duration = exp_config['let_poststim']
    let_repeats = exp_config['let_trials']
    let_folder = os.path.join(stim_root,"let")
    let_stims = get_audio_files(let_folder)
    let_stims = let_stims * let_repeats
    let_trials = len(let_stims)
    let_idx = 0 
    #Aversion Sound Test (AST) variables
    ast_prestim_duration = exp_config['ast_prestim']
    ast_stim_duration = exp_config['ast_stim']
    ast_poststim_duration = exp_config['ast_poststim']
    ast_repeats = exp_config['ast_trials']
    ast_folder = os.path.join(stim_root,"ast")
    ast_stims = get_audio_files(ast_folder)
    ast_stims = ast_stims * ast_repeats
    ast_trials = len(ast_stims)
    random.shuffle(ast_stims)
    ast_idx = 0
    
    root_dir = os.path.join(os.getcwd(),"exp_data") 
    participant_id = expInfo['participant']
    run = expInfo['run']
    task = yaml_config['task']
    
    print("creating additional modalitiy recorders: PPG and Video")
    global started_recording, lsl_socket, recorder, ppg_recorder, recording_thread, ppg_thread
    if enable_video:
        recorder = VideoRecorder(cam_id=1, output_name=participant_id, display_video=False, enable_lsl=True)
        recording_thread = threading.Thread(target=recorder.record_video)
    if enable_ppg:
        ppg_recorder = PPGRecorder(port="COM4", enable_lsl=True, simulate_data=True)
        ppg_thread = threading.Thread(target=ppg_recorder.start)
    
    lsl_socket = socket.create_connection(("localhost", 22345))
    if lsl_socket is not None:
        started_recording = True
        if recording_thread:
            recording_thread.start()
        if ppg_thread:
            ppg_thread.start()
        template_str = os.path.join("sub-%p","sub-%p_task-%b_run-%n.xdf")
        config_str = f'filename {{root:{root_dir}}} {{template:{template_str}}} {{run:{run}}} {{participant:{participant_id}}} {{task:{task}}}\n'
        print(config_str)
    
        lsl_socket.sendall(b"update\n")
        lsl_socket.sendall(b"select all\n")
        lsl_socket.sendall(config_str.encode('utf-8'))
        lsl_socket.sendall(b"start\n")
       
        core.wait(5)
    
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Welcome!\n\nPlease focus only on the cross area on the screen.\n\nUse your right hand to hold the mouse, as you will need it shortly.\n\nIn experiment 1, the screen color will change. No response is required from you.\n\nWhen you are ready, click the left mouse button to start the experiment.\n',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    mouse = event.Mouse(win=win)
    x, y = [None, None]
    mouse.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "pmt_prestim" ---
    black_screen = visual.Rect(
        win=win, name='black_screen',
        width=(2, 1)[0], height=(2, 1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    cross_7 = visual.ShapeStim(
        win=win, name='cross_7', vertices='cross',
        size=(0.3, 0.3),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "pmt_stim" ---
    gray_screen = visual.Rect(
        win=win, name='gray_screen',
        width=(2, 1)[0], height=(2, 1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[0.0039, 0.0039, 0.0039], fillColor=[0.0039, 0.0039, 0.0039],
        opacity=None, depth=-1.0, interpolate=True)
    cross_8 = visual.ShapeStim(
        win=win, name='cross_8', vertices='cross',
        size=(0.3, 0.3),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "pmt_poststim" ---
    black_screen_2 = visual.Rect(
        win=win, name='black_screen_2',
        width=(2, 1)[0], height=(2, 1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    cross_9 = visual.ShapeStim(
        win=win, name='cross_9', vertices='cross',
        size=(0.3, 0.3),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "hlt_welcome" ---
    text = visual.TextStim(win=win, name='text',
        text="In Experiment 2, \n\nWe will play a pure-tone sound, gradually increasing the loudness.\n\nFor each loudness level, please rate it as follows:\n\n0: Can't hear\n1: Clearly audible\n2: Too loud\n\nUse the left mouse button to select your rating.",
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    mouse_2 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_2.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "hlt_prestim" ---
    black_screen_3 = visual.Rect(
        win=win, name='black_screen_3',
        width=(2, 1)[0], height=(2, 1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    cross_2 = visual.ShapeStim(
        win=win, name='cross_2', vertices='cross',
        size=(0.3, 0.3),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "hlt_stim" ---
    hlt_sound = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='hlt_sound',    name='hlt_sound'
    )
    hlt_sound.setVolume(1.0)
    black_screen_4 = visual.Rect(
        win=win, name='black_screen_4',
        width=(2, 1)[0], height=(2, 1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    cross_5 = visual.ShapeStim(
        win=win, name='cross_5', vertices='cross',
        size=(0.3, 0.3),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    
    # --- Initialize components for Routine "hlt_poststim" ---
    black_screen_5 = visual.Rect(
        win=win, name='black_screen_5',
        width=(2, 1)[0], height=(2, 1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    cross_6 = visual.ShapeStim(
        win=win, name='cross_6', vertices='cross',
        size=(0.3, 0.3),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "hlt_response" ---
    slider_4 = visual.Slider(win=win, name='slider_4',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=(0, 1, 2),ticks=None, granularity=1,
        style='radio', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    
    # --- Initialize components for Routine "let_welcome" ---
    text_6 = visual.TextStim(win=win, name='text_6',
        text='In Experiment 3, \n\nA voice will speak random numbers from 0 to 20.\n\nEach time you hear a number, click the left mouse button to select the corresponding number on the screen.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    mouse_3 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_3.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "let_prestim" ---
    black_screen_6 = visual.Rect(
        win=win, name='black_screen_6',
        width=(2, 1)[0], height=(2, 1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    cross_10 = visual.ShapeStim(
        win=win, name='cross_10', vertices='cross',
        size=(0.3, 0.3),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "let_stim" ---
    let_sound = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='let_sound',    name='let_sound'
    )
    let_sound.setVolume(1.0)
    black_screen_7 = visual.Rect(
        win=win, name='black_screen_7',
        width=(2, 1)[0], height=(2, 1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-2.0, interpolate=True)
    cross_11 = visual.ShapeStim(
        win=win, name='cross_11', vertices='cross',
        size=(0.3, 0.3),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-3.0, interpolate=True)
    
    # --- Initialize components for Routine "let_poststim" ---
    black_screen_8 = visual.Rect(
        win=win, name='black_screen_8',
        width=(2, 1)[0], height=(2, 1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    cross_12 = visual.ShapeStim(
        win=win, name='cross_12', vertices='cross',
        size=(0.3, 0.3),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "let_response" ---
    slider_5 = visual.Slider(win=win, name='slider_5',
        startValue=None, size=(1.0, 0.1), pos=(0, 0.3), units=win.units,
        labels=(0, 1, 2, 3, 4, 5, 6),ticks=None, granularity=1,
        style='radio', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-1, readOnly=False)
    slider_6 = visual.Slider(win=win, name='slider_6',
        startValue=None, size=(1.0, 0.1), pos=(0, 0), units=win.units,
        labels=(7, 8, 9, 10, 11, 12, 13),ticks=None, granularity=1,
        style='radio', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    slider_7 = visual.Slider(win=win, name='slider_7',
        startValue=None, size=(1.0, 0.1), pos=(0, -0.3), units=win.units,
        labels=(14, 15, 16, 17, 18, 19, 20),ticks=None, granularity=1,
        style='radio', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=-3, readOnly=False)
    
    # --- Initialize components for Routine "ast_welcome" ---
    text_7 = visual.TextStim(win=win, name='text_7',
        text='In Experiment 4, \n\nWe will play some ambient scene sounds. No response is required from you.',
        font='Open Sans',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    mouse_4 = event.Mouse(win=win)
    x, y = [None, None]
    mouse_4.mouseClock = core.Clock()
    
    # --- Initialize components for Routine "ast_prestim" ---
    black_screen_9 = visual.Rect(
        win=win, name='black_screen_9',
        width=(2, 1)[0], height=(2, 1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    cross_13 = visual.ShapeStim(
        win=win, name='cross_13', vertices='cross',
        size=(0.3, 0.3),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "stim" ---
    current_sound = sound.Sound(
        'A', 
        secs=-1, 
        stereo=True, 
        hamming=True, 
        speaker='current_sound',    name='current_sound'
    )
    current_sound.setVolume(1.0)
    cross_14 = visual.ShapeStim(
        win=win, name='cross_14', vertices='cross',
        size=(0.3, 0.3),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    
    # --- Initialize components for Routine "post_stim" ---
    black_screen_10 = visual.Rect(
        win=win, name='black_screen_10',
        width=(2, 1)[0], height=(2, 1)[1],
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=[-1.0000, -1.0000, -1.0000], fillColor=[-1.0000, -1.0000, -1.0000],
        opacity=None, depth=-1.0, interpolate=True)
    cross_15 = visual.ShapeStim(
        win=win, name='cross_15', vertices='cross',
        size=(0.3, 0.3),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=-2.0, interpolate=True)
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "welcome" ---
    # create an object to store info about Routine welcome
    welcome = data.Routine(
        name='welcome',
        components=[text_2, mouse],
    )
    welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_init
    marker_outlet.push_sample(["start"])
    # setup some python lists for storing info about the mouse
    mouse.x = []
    mouse.y = []
    mouse.leftButton = []
    mouse.midButton = []
    mouse.rightButton = []
    mouse.time = []
    gotValidClick = False  # until a click is received
    # store start times for welcome
    welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    welcome.tStart = globalClock.getTime(format='float')
    welcome.status = STARTED
    welcome.maxDuration = None
    # keep track of which components have finished
    welcomeComponents = welcome.components
    for thisComponent in welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        # *mouse* updates
        
        # if mouse is starting this frame...
        if mouse.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse.frameNStart = frameN  # exact frame index
            mouse.tStart = t  # local t and not account for scr refresh
            mouse.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse.status = STARTED
            mouse.mouseClock.reset()
            prevButtonState = mouse.getPressed()  # if button is down already this ISN'T a new click
        if mouse.status == STARTED:  # only update if started and not finished!
            buttons = mouse.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = mouse.getPos()
                    mouse.x.append(x)
                    mouse.y.append(y)
                    buttons = mouse.getPressed()
                    mouse.leftButton.append(buttons[0])
                    mouse.midButton.append(buttons[1])
                    mouse.rightButton.append(buttons[2])
                    mouse.time.append(mouse.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for welcome
    welcome.tStop = globalClock.getTime(format='float')
    welcome.tStopRefresh = tThisFlipGlobal
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse.x', mouse.x)
    thisExp.addData('mouse.y', mouse.y)
    thisExp.addData('mouse.leftButton', mouse.leftButton)
    thisExp.addData('mouse.midButton', mouse.midButton)
    thisExp.addData('mouse.rightButton', mouse.rightButton)
    thisExp.addData('mouse.time', mouse.time)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_pmt = data.TrialHandler2(
        name='trials_pmt',
        nReps=pmt_trials, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials_pmt)  # add the loop to the experiment
    thisTrials_pmt = trials_pmt.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_pmt.rgb)
    if thisTrials_pmt != None:
        for paramName in thisTrials_pmt:
            globals()[paramName] = thisTrials_pmt[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials_pmt in trials_pmt:
        currentLoop = trials_pmt
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_pmt.rgb)
        if thisTrials_pmt != None:
            for paramName in thisTrials_pmt:
                globals()[paramName] = thisTrials_pmt[paramName]
        
        # --- Prepare to start Routine "pmt_prestim" ---
        # create an object to store info about Routine pmt_prestim
        pmt_prestim = data.Routine(
            name='pmt_prestim',
            components=[black_screen, cross_7],
        )
        pmt_prestim.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from pmt_code
        #marker_outlet.push_sample(["pmt_prestim"])
        # store start times for pmt_prestim
        pmt_prestim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pmt_prestim.tStart = globalClock.getTime(format='float')
        pmt_prestim.status = STARTED
        thisExp.addData('pmt_prestim.started', pmt_prestim.tStart)
        pmt_prestim.maxDuration = None
        # keep track of which components have finished
        pmt_prestimComponents = pmt_prestim.components
        for thisComponent in pmt_prestim.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pmt_prestim" ---
        # if trial has changed, end Routine now
        if isinstance(trials_pmt, data.TrialHandler2) and thisTrials_pmt.thisN != trials_pmt.thisTrial.thisN:
            continueRoutine = False
        pmt_prestim.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *black_screen* updates
            
            # if black_screen is starting this frame...
            if black_screen.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                black_screen.frameNStart = frameN  # exact frame index
                black_screen.tStart = t  # local t and not account for scr refresh
                black_screen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_screen, 'tStartRefresh')  # time at next scr refresh
                # update status
                black_screen.status = STARTED
                black_screen.setAutoDraw(True)
            
            # if black_screen is active this frame...
            if black_screen.status == STARTED:
                # update params
                pass
            
            # if black_screen is stopping this frame...
            if black_screen.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black_screen.tStartRefresh + pmt_prestim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    black_screen.tStop = t  # not accounting for scr refresh
                    black_screen.tStopRefresh = tThisFlipGlobal  # on global time
                    black_screen.frameNStop = frameN  # exact frame index
                    # update status
                    black_screen.status = FINISHED
                    black_screen.setAutoDraw(False)
            
            # *cross_7* updates
            
            # if cross_7 is starting this frame...
            if cross_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_7.frameNStart = frameN  # exact frame index
                cross_7.tStart = t  # local t and not account for scr refresh
                cross_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_7, 'tStartRefresh')  # time at next scr refresh
                # update status
                cross_7.status = STARTED
                cross_7.setAutoDraw(True)
            
            # if cross_7 is active this frame...
            if cross_7.status == STARTED:
                # update params
                pass
            
            # if cross_7 is stopping this frame...
            if cross_7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_7.tStartRefresh + pmt_prestim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_7.tStop = t  # not accounting for scr refresh
                    cross_7.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_7.frameNStop = frameN  # exact frame index
                    # update status
                    cross_7.status = FINISHED
                    cross_7.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                pmt_prestim.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pmt_prestim.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pmt_prestim" ---
        for thisComponent in pmt_prestim.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pmt_prestim
        pmt_prestim.tStop = globalClock.getTime(format='float')
        pmt_prestim.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pmt_prestim.stopped', pmt_prestim.tStop)
        # the Routine "pmt_prestim" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "pmt_stim" ---
        # create an object to store info about Routine pmt_stim
        pmt_stim = data.Routine(
            name='pmt_stim',
            components=[gray_screen, cross_8],
        )
        pmt_stim.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_4
        marker_outlet.push_sample(["pmt_stim"])
        # store start times for pmt_stim
        pmt_stim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pmt_stim.tStart = globalClock.getTime(format='float')
        pmt_stim.status = STARTED
        thisExp.addData('pmt_stim.started', pmt_stim.tStart)
        pmt_stim.maxDuration = None
        # keep track of which components have finished
        pmt_stimComponents = pmt_stim.components
        for thisComponent in pmt_stim.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pmt_stim" ---
        # if trial has changed, end Routine now
        if isinstance(trials_pmt, data.TrialHandler2) and thisTrials_pmt.thisN != trials_pmt.thisTrial.thisN:
            continueRoutine = False
        pmt_stim.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *gray_screen* updates
            
            # if gray_screen is starting this frame...
            if gray_screen.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                gray_screen.frameNStart = frameN  # exact frame index
                gray_screen.tStart = t  # local t and not account for scr refresh
                gray_screen.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(gray_screen, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'gray_screen.started')
                # update status
                gray_screen.status = STARTED
                gray_screen.setAutoDraw(True)
            
            # if gray_screen is active this frame...
            if gray_screen.status == STARTED:
                # update params
                pass
            
            # if gray_screen is stopping this frame...
            if gray_screen.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > gray_screen.tStartRefresh + pmt_stim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    gray_screen.tStop = t  # not accounting for scr refresh
                    gray_screen.tStopRefresh = tThisFlipGlobal  # on global time
                    gray_screen.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'gray_screen.stopped')
                    # update status
                    gray_screen.status = FINISHED
                    gray_screen.setAutoDraw(False)
            
            # *cross_8* updates
            
            # if cross_8 is starting this frame...
            if cross_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_8.frameNStart = frameN  # exact frame index
                cross_8.tStart = t  # local t and not account for scr refresh
                cross_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_8, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_8.started')
                # update status
                cross_8.status = STARTED
                cross_8.setAutoDraw(True)
            
            # if cross_8 is active this frame...
            if cross_8.status == STARTED:
                # update params
                pass
            
            # if cross_8 is stopping this frame...
            if cross_8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_8.tStartRefresh + pmt_stim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_8.tStop = t  # not accounting for scr refresh
                    cross_8.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_8.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_8.stopped')
                    # update status
                    cross_8.status = FINISHED
                    cross_8.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                pmt_stim.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pmt_stim.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pmt_stim" ---
        for thisComponent in pmt_stim.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pmt_stim
        pmt_stim.tStop = globalClock.getTime(format='float')
        pmt_stim.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pmt_stim.stopped', pmt_stim.tStop)
        # the Routine "pmt_stim" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "pmt_poststim" ---
        # create an object to store info about Routine pmt_poststim
        pmt_poststim = data.Routine(
            name='pmt_poststim',
            components=[black_screen_2, cross_9],
        )
        pmt_poststim.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_5
        marker_outlet.push_sample(["pmt_poststim"])
        # store start times for pmt_poststim
        pmt_poststim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        pmt_poststim.tStart = globalClock.getTime(format='float')
        pmt_poststim.status = STARTED
        thisExp.addData('pmt_poststim.started', pmt_poststim.tStart)
        pmt_poststim.maxDuration = None
        # keep track of which components have finished
        pmt_poststimComponents = pmt_poststim.components
        for thisComponent in pmt_poststim.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "pmt_poststim" ---
        # if trial has changed, end Routine now
        if isinstance(trials_pmt, data.TrialHandler2) and thisTrials_pmt.thisN != trials_pmt.thisTrial.thisN:
            continueRoutine = False
        pmt_poststim.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *black_screen_2* updates
            
            # if black_screen_2 is starting this frame...
            if black_screen_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                black_screen_2.frameNStart = frameN  # exact frame index
                black_screen_2.tStart = t  # local t and not account for scr refresh
                black_screen_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_screen_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                black_screen_2.status = STARTED
                black_screen_2.setAutoDraw(True)
            
            # if black_screen_2 is active this frame...
            if black_screen_2.status == STARTED:
                # update params
                pass
            
            # if black_screen_2 is stopping this frame...
            if black_screen_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black_screen_2.tStartRefresh + pmt_poststim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    black_screen_2.tStop = t  # not accounting for scr refresh
                    black_screen_2.tStopRefresh = tThisFlipGlobal  # on global time
                    black_screen_2.frameNStop = frameN  # exact frame index
                    # update status
                    black_screen_2.status = FINISHED
                    black_screen_2.setAutoDraw(False)
            
            # *cross_9* updates
            
            # if cross_9 is starting this frame...
            if cross_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_9.frameNStart = frameN  # exact frame index
                cross_9.tStart = t  # local t and not account for scr refresh
                cross_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_9, 'tStartRefresh')  # time at next scr refresh
                # update status
                cross_9.status = STARTED
                cross_9.setAutoDraw(True)
            
            # if cross_9 is active this frame...
            if cross_9.status == STARTED:
                # update params
                pass
            
            # if cross_9 is stopping this frame...
            if cross_9.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_9.tStartRefresh + pmt_poststim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_9.tStop = t  # not accounting for scr refresh
                    cross_9.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_9.frameNStop = frameN  # exact frame index
                    # update status
                    cross_9.status = FINISHED
                    cross_9.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                pmt_poststim.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in pmt_poststim.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "pmt_poststim" ---
        for thisComponent in pmt_poststim.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for pmt_poststim
        pmt_poststim.tStop = globalClock.getTime(format='float')
        pmt_poststim.tStopRefresh = tThisFlipGlobal
        thisExp.addData('pmt_poststim.stopped', pmt_poststim.tStop)
        # the Routine "pmt_poststim" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed pmt_trials repeats of 'trials_pmt'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "hlt_welcome" ---
    # create an object to store info about Routine hlt_welcome
    hlt_welcome = data.Routine(
        name='hlt_welcome',
        components=[text, mouse_2],
    )
    hlt_welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse_2
    mouse_2.x = []
    mouse_2.y = []
    mouse_2.leftButton = []
    mouse_2.midButton = []
    mouse_2.rightButton = []
    mouse_2.time = []
    gotValidClick = False  # until a click is received
    # store start times for hlt_welcome
    hlt_welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    hlt_welcome.tStart = globalClock.getTime(format='float')
    hlt_welcome.status = STARTED
    thisExp.addData('hlt_welcome.started', hlt_welcome.tStart)
    hlt_welcome.maxDuration = None
    # keep track of which components have finished
    hlt_welcomeComponents = hlt_welcome.components
    for thisComponent in hlt_welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "hlt_welcome" ---
    hlt_welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        # *mouse_2* updates
        
        # if mouse_2 is starting this frame...
        if mouse_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_2.frameNStart = frameN  # exact frame index
            mouse_2.tStart = t  # local t and not account for scr refresh
            mouse_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_2, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_2.status = STARTED
            mouse_2.mouseClock.reset()
            prevButtonState = mouse_2.getPressed()  # if button is down already this ISN'T a new click
        if mouse_2.status == STARTED:  # only update if started and not finished!
            buttons = mouse_2.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = mouse_2.getPos()
                    mouse_2.x.append(x)
                    mouse_2.y.append(y)
                    buttons = mouse_2.getPressed()
                    mouse_2.leftButton.append(buttons[0])
                    mouse_2.midButton.append(buttons[1])
                    mouse_2.rightButton.append(buttons[2])
                    mouse_2.time.append(mouse_2.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            hlt_welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in hlt_welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "hlt_welcome" ---
    for thisComponent in hlt_welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for hlt_welcome
    hlt_welcome.tStop = globalClock.getTime(format='float')
    hlt_welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('hlt_welcome.stopped', hlt_welcome.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_2.x', mouse_2.x)
    thisExp.addData('mouse_2.y', mouse_2.y)
    thisExp.addData('mouse_2.leftButton', mouse_2.leftButton)
    thisExp.addData('mouse_2.midButton', mouse_2.midButton)
    thisExp.addData('mouse_2.rightButton', mouse_2.rightButton)
    thisExp.addData('mouse_2.time', mouse_2.time)
    thisExp.nextEntry()
    # the Routine "hlt_welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_hlt = data.TrialHandler2(
        name='trials_hlt',
        nReps=hlt_trials, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials_hlt)  # add the loop to the experiment
    thisTrials_hlt = trials_hlt.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_hlt.rgb)
    if thisTrials_hlt != None:
        for paramName in thisTrials_hlt:
            globals()[paramName] = thisTrials_hlt[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials_hlt in trials_hlt:
        currentLoop = trials_hlt
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_hlt.rgb)
        if thisTrials_hlt != None:
            for paramName in thisTrials_hlt:
                globals()[paramName] = thisTrials_hlt[paramName]
        
        # --- Prepare to start Routine "hlt_prestim" ---
        # create an object to store info about Routine hlt_prestim
        hlt_prestim = data.Routine(
            name='hlt_prestim',
            components=[black_screen_3, cross_2],
        )
        hlt_prestim.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_6
        sound_path = hlt_stims[hlt_idx]
        name = os.path.basename(sound_path).split('.')[0]
        marker_outlet.push_sample([f"hlt_prestim-{name}"])
        thisExp.addData("current_hlt_stim",name)
        # store start times for hlt_prestim
        hlt_prestim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        hlt_prestim.tStart = globalClock.getTime(format='float')
        hlt_prestim.status = STARTED
        thisExp.addData('hlt_prestim.started', hlt_prestim.tStart)
        hlt_prestim.maxDuration = None
        # keep track of which components have finished
        hlt_prestimComponents = hlt_prestim.components
        for thisComponent in hlt_prestim.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "hlt_prestim" ---
        # if trial has changed, end Routine now
        if isinstance(trials_hlt, data.TrialHandler2) and thisTrials_hlt.thisN != trials_hlt.thisTrial.thisN:
            continueRoutine = False
        hlt_prestim.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *black_screen_3* updates
            
            # if black_screen_3 is starting this frame...
            if black_screen_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                black_screen_3.frameNStart = frameN  # exact frame index
                black_screen_3.tStart = t  # local t and not account for scr refresh
                black_screen_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_screen_3, 'tStartRefresh')  # time at next scr refresh
                # update status
                black_screen_3.status = STARTED
                black_screen_3.setAutoDraw(True)
            
            # if black_screen_3 is active this frame...
            if black_screen_3.status == STARTED:
                # update params
                pass
            
            # if black_screen_3 is stopping this frame...
            if black_screen_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black_screen_3.tStartRefresh + hlt_prestim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    black_screen_3.tStop = t  # not accounting for scr refresh
                    black_screen_3.tStopRefresh = tThisFlipGlobal  # on global time
                    black_screen_3.frameNStop = frameN  # exact frame index
                    # update status
                    black_screen_3.status = FINISHED
                    black_screen_3.setAutoDraw(False)
            
            # *cross_2* updates
            
            # if cross_2 is starting this frame...
            if cross_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_2.frameNStart = frameN  # exact frame index
                cross_2.tStart = t  # local t and not account for scr refresh
                cross_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_2, 'tStartRefresh')  # time at next scr refresh
                # update status
                cross_2.status = STARTED
                cross_2.setAutoDraw(True)
            
            # if cross_2 is active this frame...
            if cross_2.status == STARTED:
                # update params
                pass
            
            # if cross_2 is stopping this frame...
            if cross_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_2.tStartRefresh + hlt_prestim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_2.tStop = t  # not accounting for scr refresh
                    cross_2.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_2.frameNStop = frameN  # exact frame index
                    # update status
                    cross_2.status = FINISHED
                    cross_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                hlt_prestim.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in hlt_prestim.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "hlt_prestim" ---
        for thisComponent in hlt_prestim.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for hlt_prestim
        hlt_prestim.tStop = globalClock.getTime(format='float')
        hlt_prestim.tStopRefresh = tThisFlipGlobal
        thisExp.addData('hlt_prestim.stopped', hlt_prestim.tStop)
        # the Routine "hlt_prestim" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "hlt_stim" ---
        # create an object to store info about Routine hlt_stim
        hlt_stim = data.Routine(
            name='hlt_stim',
            components=[hlt_sound, black_screen_4, cross_5],
        )
        hlt_stim.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_7
        hlt_sound.setSound(sound_path, hamming=True)
        marker_outlet.push_sample([f"hlt_stim-{name}"])
        sound_duration = current_sound.getDuration()
        print(f'{name}: {sound_duration}')
        if hlt_idx < len(hlt_stims)-1:
            hlt_idx+=1
        else:
            hlt_idx = 0
        hlt_sound.setSound(sound_path, secs=hlt_stim_duration, hamming=True)
        hlt_sound.setVolume(1.0, log=False)
        hlt_sound.seek(0)
        # store start times for hlt_stim
        hlt_stim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        hlt_stim.tStart = globalClock.getTime(format='float')
        hlt_stim.status = STARTED
        thisExp.addData('hlt_stim.started', hlt_stim.tStart)
        hlt_stim.maxDuration = None
        # keep track of which components have finished
        hlt_stimComponents = hlt_stim.components
        for thisComponent in hlt_stim.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "hlt_stim" ---
        # if trial has changed, end Routine now
        if isinstance(trials_hlt, data.TrialHandler2) and thisTrials_hlt.thisN != trials_hlt.thisTrial.thisN:
            continueRoutine = False
        hlt_stim.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *hlt_sound* updates
            
            # if hlt_sound is starting this frame...
            if hlt_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                hlt_sound.frameNStart = frameN  # exact frame index
                hlt_sound.tStart = t  # local t and not account for scr refresh
                hlt_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('hlt_sound.started', tThisFlipGlobal)
                # update status
                hlt_sound.status = STARTED
                hlt_sound.play(when=win)  # sync with win flip
            
            # if hlt_sound is stopping this frame...
            if hlt_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > hlt_sound.tStartRefresh + hlt_stim_duration-frameTolerance or hlt_sound.isFinished:
                    # keep track of stop time/frame for later
                    hlt_sound.tStop = t  # not accounting for scr refresh
                    hlt_sound.tStopRefresh = tThisFlipGlobal  # on global time
                    hlt_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'hlt_sound.stopped')
                    # update status
                    hlt_sound.status = FINISHED
                    hlt_sound.stop()
            
            # *black_screen_4* updates
            
            # if black_screen_4 is starting this frame...
            if black_screen_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                black_screen_4.frameNStart = frameN  # exact frame index
                black_screen_4.tStart = t  # local t and not account for scr refresh
                black_screen_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_screen_4, 'tStartRefresh')  # time at next scr refresh
                # update status
                black_screen_4.status = STARTED
                black_screen_4.setAutoDraw(True)
            
            # if black_screen_4 is active this frame...
            if black_screen_4.status == STARTED:
                # update params
                pass
            
            # if black_screen_4 is stopping this frame...
            if black_screen_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black_screen_4.tStartRefresh + hlt_stim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    black_screen_4.tStop = t  # not accounting for scr refresh
                    black_screen_4.tStopRefresh = tThisFlipGlobal  # on global time
                    black_screen_4.frameNStop = frameN  # exact frame index
                    # update status
                    black_screen_4.status = FINISHED
                    black_screen_4.setAutoDraw(False)
            
            # *cross_5* updates
            
            # if cross_5 is starting this frame...
            if cross_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_5.frameNStart = frameN  # exact frame index
                cross_5.tStart = t  # local t and not account for scr refresh
                cross_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_5, 'tStartRefresh')  # time at next scr refresh
                # update status
                cross_5.status = STARTED
                cross_5.setAutoDraw(True)
            
            # if cross_5 is active this frame...
            if cross_5.status == STARTED:
                # update params
                pass
            
            # if cross_5 is stopping this frame...
            if cross_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_5.tStartRefresh + hlt_stim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_5.tStop = t  # not accounting for scr refresh
                    cross_5.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_5.frameNStop = frameN  # exact frame index
                    # update status
                    cross_5.status = FINISHED
                    cross_5.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[hlt_sound]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                hlt_stim.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in hlt_stim.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "hlt_stim" ---
        for thisComponent in hlt_stim.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for hlt_stim
        hlt_stim.tStop = globalClock.getTime(format='float')
        hlt_stim.tStopRefresh = tThisFlipGlobal
        thisExp.addData('hlt_stim.stopped', hlt_stim.tStop)
        hlt_sound.pause()  # ensure sound has stopped at end of Routine
        # the Routine "hlt_stim" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "hlt_poststim" ---
        # create an object to store info about Routine hlt_poststim
        hlt_poststim = data.Routine(
            name='hlt_poststim',
            components=[black_screen_5, cross_6],
        )
        hlt_poststim.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_8
        marker_outlet.push_sample([f"hlt_poststim-{name}"])
        # store start times for hlt_poststim
        hlt_poststim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        hlt_poststim.tStart = globalClock.getTime(format='float')
        hlt_poststim.status = STARTED
        thisExp.addData('hlt_poststim.started', hlt_poststim.tStart)
        hlt_poststim.maxDuration = None
        # keep track of which components have finished
        hlt_poststimComponents = hlt_poststim.components
        for thisComponent in hlt_poststim.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "hlt_poststim" ---
        # if trial has changed, end Routine now
        if isinstance(trials_hlt, data.TrialHandler2) and thisTrials_hlt.thisN != trials_hlt.thisTrial.thisN:
            continueRoutine = False
        hlt_poststim.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *black_screen_5* updates
            
            # if black_screen_5 is starting this frame...
            if black_screen_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                black_screen_5.frameNStart = frameN  # exact frame index
                black_screen_5.tStart = t  # local t and not account for scr refresh
                black_screen_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_screen_5, 'tStartRefresh')  # time at next scr refresh
                # update status
                black_screen_5.status = STARTED
                black_screen_5.setAutoDraw(True)
            
            # if black_screen_5 is active this frame...
            if black_screen_5.status == STARTED:
                # update params
                pass
            
            # if black_screen_5 is stopping this frame...
            if black_screen_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black_screen_5.tStartRefresh + hlt_poststim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    black_screen_5.tStop = t  # not accounting for scr refresh
                    black_screen_5.tStopRefresh = tThisFlipGlobal  # on global time
                    black_screen_5.frameNStop = frameN  # exact frame index
                    # update status
                    black_screen_5.status = FINISHED
                    black_screen_5.setAutoDraw(False)
            
            # *cross_6* updates
            
            # if cross_6 is starting this frame...
            if cross_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_6.frameNStart = frameN  # exact frame index
                cross_6.tStart = t  # local t and not account for scr refresh
                cross_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_6, 'tStartRefresh')  # time at next scr refresh
                # update status
                cross_6.status = STARTED
                cross_6.setAutoDraw(True)
            
            # if cross_6 is active this frame...
            if cross_6.status == STARTED:
                # update params
                pass
            
            # if cross_6 is stopping this frame...
            if cross_6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_6.tStartRefresh + hlt_poststim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_6.tStop = t  # not accounting for scr refresh
                    cross_6.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_6.frameNStop = frameN  # exact frame index
                    # update status
                    cross_6.status = FINISHED
                    cross_6.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                hlt_poststim.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in hlt_poststim.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "hlt_poststim" ---
        for thisComponent in hlt_poststim.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for hlt_poststim
        hlt_poststim.tStop = globalClock.getTime(format='float')
        hlt_poststim.tStopRefresh = tThisFlipGlobal
        thisExp.addData('hlt_poststim.stopped', hlt_poststim.tStop)
        # the Routine "hlt_poststim" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "hlt_response" ---
        # create an object to store info about Routine hlt_response
        hlt_response = data.Routine(
            name='hlt_response',
            components=[slider_4],
        )
        hlt_response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_9
        marker_outlet.push_sample([f"hlt_response-{name}"])
        slider_4.reset()
        # store start times for hlt_response
        hlt_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        hlt_response.tStart = globalClock.getTime(format='float')
        hlt_response.status = STARTED
        thisExp.addData('hlt_response.started', hlt_response.tStart)
        hlt_response.maxDuration = None
        # keep track of which components have finished
        hlt_responseComponents = hlt_response.components
        for thisComponent in hlt_response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "hlt_response" ---
        # if trial has changed, end Routine now
        if isinstance(trials_hlt, data.TrialHandler2) and thisTrials_hlt.thisN != trials_hlt.thisTrial.thisN:
            continueRoutine = False
        hlt_response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *slider_4* updates
            
            # if slider_4 is starting this frame...
            if slider_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_4.frameNStart = frameN  # exact frame index
                slider_4.tStart = t  # local t and not account for scr refresh
                slider_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_4, 'tStartRefresh')  # time at next scr refresh
                # update status
                slider_4.status = STARTED
                slider_4.setAutoDraw(True)
            
            # if slider_4 is active this frame...
            if slider_4.status == STARTED:
                # update params
                pass
            
            # Check slider_4 for response to end Routine
            if slider_4.getRating() is not None and slider_4.status == STARTED:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                hlt_response.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in hlt_response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "hlt_response" ---
        for thisComponent in hlt_response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for hlt_response
        hlt_response.tStop = globalClock.getTime(format='float')
        hlt_response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('hlt_response.stopped', hlt_response.tStop)
        trials_hlt.addData('slider_4.response', slider_4.getRating())
        trials_hlt.addData('slider_4.rt', slider_4.getRT())
        # the Routine "hlt_response" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed hlt_trials repeats of 'trials_hlt'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "let_welcome" ---
    # create an object to store info about Routine let_welcome
    let_welcome = data.Routine(
        name='let_welcome',
        components=[text_6, mouse_3],
    )
    let_welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse_3
    mouse_3.x = []
    mouse_3.y = []
    mouse_3.leftButton = []
    mouse_3.midButton = []
    mouse_3.rightButton = []
    mouse_3.time = []
    gotValidClick = False  # until a click is received
    # store start times for let_welcome
    let_welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    let_welcome.tStart = globalClock.getTime(format='float')
    let_welcome.status = STARTED
    thisExp.addData('let_welcome.started', let_welcome.tStart)
    let_welcome.maxDuration = None
    # keep track of which components have finished
    let_welcomeComponents = let_welcome.components
    for thisComponent in let_welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "let_welcome" ---
    let_welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_6* updates
        
        # if text_6 is starting this frame...
        if text_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_6.frameNStart = frameN  # exact frame index
            text_6.tStart = t  # local t and not account for scr refresh
            text_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_6.started')
            # update status
            text_6.status = STARTED
            text_6.setAutoDraw(True)
        
        # if text_6 is active this frame...
        if text_6.status == STARTED:
            # update params
            pass
        # *mouse_3* updates
        
        # if mouse_3 is starting this frame...
        if mouse_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_3.frameNStart = frameN  # exact frame index
            mouse_3.tStart = t  # local t and not account for scr refresh
            mouse_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_3, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_3.status = STARTED
            mouse_3.mouseClock.reset()
            prevButtonState = mouse_3.getPressed()  # if button is down already this ISN'T a new click
        if mouse_3.status == STARTED:  # only update if started and not finished!
            buttons = mouse_3.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = mouse_3.getPos()
                    mouse_3.x.append(x)
                    mouse_3.y.append(y)
                    buttons = mouse_3.getPressed()
                    mouse_3.leftButton.append(buttons[0])
                    mouse_3.midButton.append(buttons[1])
                    mouse_3.rightButton.append(buttons[2])
                    mouse_3.time.append(mouse_3.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            let_welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in let_welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "let_welcome" ---
    for thisComponent in let_welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for let_welcome
    let_welcome.tStop = globalClock.getTime(format='float')
    let_welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('let_welcome.stopped', let_welcome.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_3.x', mouse_3.x)
    thisExp.addData('mouse_3.y', mouse_3.y)
    thisExp.addData('mouse_3.leftButton', mouse_3.leftButton)
    thisExp.addData('mouse_3.midButton', mouse_3.midButton)
    thisExp.addData('mouse_3.rightButton', mouse_3.rightButton)
    thisExp.addData('mouse_3.time', mouse_3.time)
    thisExp.nextEntry()
    # the Routine "let_welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_let = data.TrialHandler2(
        name='trials_let',
        nReps=let_trials, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials_let)  # add the loop to the experiment
    thisTrials_let = trials_let.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_let.rgb)
    if thisTrials_let != None:
        for paramName in thisTrials_let:
            globals()[paramName] = thisTrials_let[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials_let in trials_let:
        currentLoop = trials_let
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_let.rgb)
        if thisTrials_let != None:
            for paramName in thisTrials_let:
                globals()[paramName] = thisTrials_let[paramName]
        
        # --- Prepare to start Routine "let_prestim" ---
        # create an object to store info about Routine let_prestim
        let_prestim = data.Routine(
            name='let_prestim',
            components=[black_screen_6, cross_10],
        )
        let_prestim.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_10
        sound_path = let_stims[let_idx]
        name = os.path.basename(sound_path).split('.')[0]
        marker_outlet.push_sample([f"let_prestim-{name}"])
        thisExp.addData("current_let_stim",name)
        
        # store start times for let_prestim
        let_prestim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        let_prestim.tStart = globalClock.getTime(format='float')
        let_prestim.status = STARTED
        thisExp.addData('let_prestim.started', let_prestim.tStart)
        let_prestim.maxDuration = None
        # keep track of which components have finished
        let_prestimComponents = let_prestim.components
        for thisComponent in let_prestim.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "let_prestim" ---
        # if trial has changed, end Routine now
        if isinstance(trials_let, data.TrialHandler2) and thisTrials_let.thisN != trials_let.thisTrial.thisN:
            continueRoutine = False
        let_prestim.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *black_screen_6* updates
            
            # if black_screen_6 is starting this frame...
            if black_screen_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                black_screen_6.frameNStart = frameN  # exact frame index
                black_screen_6.tStart = t  # local t and not account for scr refresh
                black_screen_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_screen_6, 'tStartRefresh')  # time at next scr refresh
                # update status
                black_screen_6.status = STARTED
                black_screen_6.setAutoDraw(True)
            
            # if black_screen_6 is active this frame...
            if black_screen_6.status == STARTED:
                # update params
                pass
            
            # if black_screen_6 is stopping this frame...
            if black_screen_6.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black_screen_6.tStartRefresh + let_prestim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    black_screen_6.tStop = t  # not accounting for scr refresh
                    black_screen_6.tStopRefresh = tThisFlipGlobal  # on global time
                    black_screen_6.frameNStop = frameN  # exact frame index
                    # update status
                    black_screen_6.status = FINISHED
                    black_screen_6.setAutoDraw(False)
            
            # *cross_10* updates
            
            # if cross_10 is starting this frame...
            if cross_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_10.frameNStart = frameN  # exact frame index
                cross_10.tStart = t  # local t and not account for scr refresh
                cross_10.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_10, 'tStartRefresh')  # time at next scr refresh
                # update status
                cross_10.status = STARTED
                cross_10.setAutoDraw(True)
            
            # if cross_10 is active this frame...
            if cross_10.status == STARTED:
                # update params
                pass
            
            # if cross_10 is stopping this frame...
            if cross_10.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_10.tStartRefresh + let_prestim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_10.tStop = t  # not accounting for scr refresh
                    cross_10.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_10.frameNStop = frameN  # exact frame index
                    # update status
                    cross_10.status = FINISHED
                    cross_10.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                let_prestim.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in let_prestim.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "let_prestim" ---
        for thisComponent in let_prestim.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for let_prestim
        let_prestim.tStop = globalClock.getTime(format='float')
        let_prestim.tStopRefresh = tThisFlipGlobal
        thisExp.addData('let_prestim.stopped', let_prestim.tStop)
        # the Routine "let_prestim" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "let_stim" ---
        # create an object to store info about Routine let_stim
        let_stim = data.Routine(
            name='let_stim',
            components=[let_sound, black_screen_7, cross_11],
        )
        let_stim.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_11
        let_sound.setSound(sound_path, hamming=True)
        marker_outlet.push_sample([f"let_stim-{name}"])
        sound_duration = current_sound.getDuration()
        print(f'{name}: {sound_duration}')
        if let_idx < len(let_stims)-1:
            let_idx+=1
        else:
            let_idx = 0
        let_sound.setSound(sound_path, secs=let_stim_duration, hamming=True)
        let_sound.setVolume(1.0, log=False)
        let_sound.seek(0)
        # store start times for let_stim
        let_stim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        let_stim.tStart = globalClock.getTime(format='float')
        let_stim.status = STARTED
        thisExp.addData('let_stim.started', let_stim.tStart)
        let_stim.maxDuration = None
        # keep track of which components have finished
        let_stimComponents = let_stim.components
        for thisComponent in let_stim.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "let_stim" ---
        # if trial has changed, end Routine now
        if isinstance(trials_let, data.TrialHandler2) and thisTrials_let.thisN != trials_let.thisTrial.thisN:
            continueRoutine = False
        let_stim.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *let_sound* updates
            
            # if let_sound is starting this frame...
            if let_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                let_sound.frameNStart = frameN  # exact frame index
                let_sound.tStart = t  # local t and not account for scr refresh
                let_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('let_sound.started', tThisFlipGlobal)
                # update status
                let_sound.status = STARTED
                let_sound.play(when=win)  # sync with win flip
            
            # if let_sound is stopping this frame...
            if let_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > let_sound.tStartRefresh + let_stim_duration-frameTolerance or let_sound.isFinished:
                    # keep track of stop time/frame for later
                    let_sound.tStop = t  # not accounting for scr refresh
                    let_sound.tStopRefresh = tThisFlipGlobal  # on global time
                    let_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'let_sound.stopped')
                    # update status
                    let_sound.status = FINISHED
                    let_sound.stop()
            
            # *black_screen_7* updates
            
            # if black_screen_7 is starting this frame...
            if black_screen_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                black_screen_7.frameNStart = frameN  # exact frame index
                black_screen_7.tStart = t  # local t and not account for scr refresh
                black_screen_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_screen_7, 'tStartRefresh')  # time at next scr refresh
                # update status
                black_screen_7.status = STARTED
                black_screen_7.setAutoDraw(True)
            
            # if black_screen_7 is active this frame...
            if black_screen_7.status == STARTED:
                # update params
                pass
            
            # if black_screen_7 is stopping this frame...
            if black_screen_7.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black_screen_7.tStartRefresh + let_stim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    black_screen_7.tStop = t  # not accounting for scr refresh
                    black_screen_7.tStopRefresh = tThisFlipGlobal  # on global time
                    black_screen_7.frameNStop = frameN  # exact frame index
                    # update status
                    black_screen_7.status = FINISHED
                    black_screen_7.setAutoDraw(False)
            
            # *cross_11* updates
            
            # if cross_11 is starting this frame...
            if cross_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_11.frameNStart = frameN  # exact frame index
                cross_11.tStart = t  # local t and not account for scr refresh
                cross_11.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_11, 'tStartRefresh')  # time at next scr refresh
                # update status
                cross_11.status = STARTED
                cross_11.setAutoDraw(True)
            
            # if cross_11 is active this frame...
            if cross_11.status == STARTED:
                # update params
                pass
            
            # if cross_11 is stopping this frame...
            if cross_11.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_11.tStartRefresh + let_stim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_11.tStop = t  # not accounting for scr refresh
                    cross_11.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_11.frameNStop = frameN  # exact frame index
                    # update status
                    cross_11.status = FINISHED
                    cross_11.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[let_sound]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                let_stim.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in let_stim.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "let_stim" ---
        for thisComponent in let_stim.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for let_stim
        let_stim.tStop = globalClock.getTime(format='float')
        let_stim.tStopRefresh = tThisFlipGlobal
        thisExp.addData('let_stim.stopped', let_stim.tStop)
        let_sound.pause()  # ensure sound has stopped at end of Routine
        # the Routine "let_stim" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "let_poststim" ---
        # create an object to store info about Routine let_poststim
        let_poststim = data.Routine(
            name='let_poststim',
            components=[black_screen_8, cross_12],
        )
        let_poststim.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_12
        marker_outlet.push_sample([f"let_poststim-{name}"])
        
        # store start times for let_poststim
        let_poststim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        let_poststim.tStart = globalClock.getTime(format='float')
        let_poststim.status = STARTED
        thisExp.addData('let_poststim.started', let_poststim.tStart)
        let_poststim.maxDuration = None
        # keep track of which components have finished
        let_poststimComponents = let_poststim.components
        for thisComponent in let_poststim.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "let_poststim" ---
        # if trial has changed, end Routine now
        if isinstance(trials_let, data.TrialHandler2) and thisTrials_let.thisN != trials_let.thisTrial.thisN:
            continueRoutine = False
        let_poststim.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *black_screen_8* updates
            
            # if black_screen_8 is starting this frame...
            if black_screen_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                black_screen_8.frameNStart = frameN  # exact frame index
                black_screen_8.tStart = t  # local t and not account for scr refresh
                black_screen_8.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_screen_8, 'tStartRefresh')  # time at next scr refresh
                # update status
                black_screen_8.status = STARTED
                black_screen_8.setAutoDraw(True)
            
            # if black_screen_8 is active this frame...
            if black_screen_8.status == STARTED:
                # update params
                pass
            
            # if black_screen_8 is stopping this frame...
            if black_screen_8.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black_screen_8.tStartRefresh + let_poststim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    black_screen_8.tStop = t  # not accounting for scr refresh
                    black_screen_8.tStopRefresh = tThisFlipGlobal  # on global time
                    black_screen_8.frameNStop = frameN  # exact frame index
                    # update status
                    black_screen_8.status = FINISHED
                    black_screen_8.setAutoDraw(False)
            
            # *cross_12* updates
            
            # if cross_12 is starting this frame...
            if cross_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_12.frameNStart = frameN  # exact frame index
                cross_12.tStart = t  # local t and not account for scr refresh
                cross_12.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_12, 'tStartRefresh')  # time at next scr refresh
                # update status
                cross_12.status = STARTED
                cross_12.setAutoDraw(True)
            
            # if cross_12 is active this frame...
            if cross_12.status == STARTED:
                # update params
                pass
            
            # if cross_12 is stopping this frame...
            if cross_12.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_12.tStartRefresh + let_poststim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_12.tStop = t  # not accounting for scr refresh
                    cross_12.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_12.frameNStop = frameN  # exact frame index
                    # update status
                    cross_12.status = FINISHED
                    cross_12.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                let_poststim.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in let_poststim.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "let_poststim" ---
        for thisComponent in let_poststim.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for let_poststim
        let_poststim.tStop = globalClock.getTime(format='float')
        let_poststim.tStopRefresh = tThisFlipGlobal
        thisExp.addData('let_poststim.stopped', let_poststim.tStop)
        # the Routine "let_poststim" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "let_response" ---
        # create an object to store info about Routine let_response
        let_response = data.Routine(
            name='let_response',
            components=[slider_5, slider_6, slider_7],
        )
        let_response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_13
        marker_outlet.push_sample([f"let_response-{name}"])
        slider_5.reset()
        slider_6.reset()
        slider_7.reset()
        # store start times for let_response
        let_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        let_response.tStart = globalClock.getTime(format='float')
        let_response.status = STARTED
        thisExp.addData('let_response.started', let_response.tStart)
        let_response.maxDuration = None
        # keep track of which components have finished
        let_responseComponents = let_response.components
        for thisComponent in let_response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "let_response" ---
        # if trial has changed, end Routine now
        if isinstance(trials_let, data.TrialHandler2) and thisTrials_let.thisN != trials_let.thisTrial.thisN:
            continueRoutine = False
        let_response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *slider_5* updates
            
            # if slider_5 is starting this frame...
            if slider_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_5.frameNStart = frameN  # exact frame index
                slider_5.tStart = t  # local t and not account for scr refresh
                slider_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_5, 'tStartRefresh')  # time at next scr refresh
                # update status
                slider_5.status = STARTED
                slider_5.setAutoDraw(True)
            
            # if slider_5 is active this frame...
            if slider_5.status == STARTED:
                # update params
                pass
            
            # Check slider_5 for response to end Routine
            if slider_5.getRating() is not None and slider_5.status == STARTED:
                continueRoutine = False
            
            # *slider_6* updates
            
            # if slider_6 is starting this frame...
            if slider_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_6.frameNStart = frameN  # exact frame index
                slider_6.tStart = t  # local t and not account for scr refresh
                slider_6.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_6, 'tStartRefresh')  # time at next scr refresh
                # update status
                slider_6.status = STARTED
                slider_6.setAutoDraw(True)
            
            # if slider_6 is active this frame...
            if slider_6.status == STARTED:
                # update params
                pass
            
            # Check slider_6 for response to end Routine
            if slider_6.getRating() is not None and slider_6.status == STARTED:
                continueRoutine = False
            
            # *slider_7* updates
            
            # if slider_7 is starting this frame...
            if slider_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                slider_7.frameNStart = frameN  # exact frame index
                slider_7.tStart = t  # local t and not account for scr refresh
                slider_7.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_7, 'tStartRefresh')  # time at next scr refresh
                # update status
                slider_7.status = STARTED
                slider_7.setAutoDraw(True)
            
            # if slider_7 is active this frame...
            if slider_7.status == STARTED:
                # update params
                pass
            
            # Check slider_7 for response to end Routine
            if slider_7.getRating() is not None and slider_7.status == STARTED:
                continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                let_response.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in let_response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "let_response" ---
        for thisComponent in let_response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for let_response
        let_response.tStop = globalClock.getTime(format='float')
        let_response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('let_response.stopped', let_response.tStop)
        trials_let.addData('slider_5.response', slider_5.getRating())
        trials_let.addData('slider_5.rt', slider_5.getRT())
        trials_let.addData('slider_6.response', slider_6.getRating())
        trials_let.addData('slider_6.rt', slider_6.getRT())
        trials_let.addData('slider_7.response', slider_7.getRating())
        trials_let.addData('slider_7.rt', slider_7.getRT())
        # the Routine "let_response" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed let_trials repeats of 'trials_let'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "ast_welcome" ---
    # create an object to store info about Routine ast_welcome
    ast_welcome = data.Routine(
        name='ast_welcome',
        components=[text_7, mouse_4],
    )
    ast_welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # setup some python lists for storing info about the mouse_4
    mouse_4.x = []
    mouse_4.y = []
    mouse_4.leftButton = []
    mouse_4.midButton = []
    mouse_4.rightButton = []
    mouse_4.time = []
    gotValidClick = False  # until a click is received
    # store start times for ast_welcome
    ast_welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    ast_welcome.tStart = globalClock.getTime(format='float')
    ast_welcome.status = STARTED
    thisExp.addData('ast_welcome.started', ast_welcome.tStart)
    ast_welcome.maxDuration = None
    # keep track of which components have finished
    ast_welcomeComponents = ast_welcome.components
    for thisComponent in ast_welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "ast_welcome" ---
    ast_welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_7* updates
        
        # if text_7 is starting this frame...
        if text_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_7.frameNStart = frameN  # exact frame index
            text_7.tStart = t  # local t and not account for scr refresh
            text_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_7, 'tStartRefresh')  # time at next scr refresh
            # update status
            text_7.status = STARTED
            text_7.setAutoDraw(True)
        
        # if text_7 is active this frame...
        if text_7.status == STARTED:
            # update params
            pass
        # *mouse_4* updates
        
        # if mouse_4 is starting this frame...
        if mouse_4.status == NOT_STARTED and t >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            mouse_4.frameNStart = frameN  # exact frame index
            mouse_4.tStart = t  # local t and not account for scr refresh
            mouse_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(mouse_4, 'tStartRefresh')  # time at next scr refresh
            # update status
            mouse_4.status = STARTED
            mouse_4.mouseClock.reset()
            prevButtonState = mouse_4.getPressed()  # if button is down already this ISN'T a new click
        if mouse_4.status == STARTED:  # only update if started and not finished!
            buttons = mouse_4.getPressed()
            if buttons != prevButtonState:  # button state changed?
                prevButtonState = buttons
                if sum(buttons) > 0:  # state changed to a new click
                    pass
                    x, y = mouse_4.getPos()
                    mouse_4.x.append(x)
                    mouse_4.y.append(y)
                    buttons = mouse_4.getPressed()
                    mouse_4.leftButton.append(buttons[0])
                    mouse_4.midButton.append(buttons[1])
                    mouse_4.rightButton.append(buttons[2])
                    mouse_4.time.append(mouse_4.mouseClock.getTime())
                    
                    continueRoutine = False  # end routine on response
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            ast_welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in ast_welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "ast_welcome" ---
    for thisComponent in ast_welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for ast_welcome
    ast_welcome.tStop = globalClock.getTime(format='float')
    ast_welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('ast_welcome.stopped', ast_welcome.tStop)
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('mouse_4.x', mouse_4.x)
    thisExp.addData('mouse_4.y', mouse_4.y)
    thisExp.addData('mouse_4.leftButton', mouse_4.leftButton)
    thisExp.addData('mouse_4.midButton', mouse_4.midButton)
    thisExp.addData('mouse_4.rightButton', mouse_4.rightButton)
    thisExp.addData('mouse_4.time', mouse_4.time)
    thisExp.nextEntry()
    # the Routine "ast_welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials_ast = data.TrialHandler2(
        name='trials_ast',
        nReps=ast_trials, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(trials_ast)  # add the loop to the experiment
    thisTrials_ast = trials_ast.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_ast.rgb)
    if thisTrials_ast != None:
        for paramName in thisTrials_ast:
            globals()[paramName] = thisTrials_ast[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials_ast in trials_ast:
        currentLoop = trials_ast
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_ast.rgb)
        if thisTrials_ast != None:
            for paramName in thisTrials_ast:
                globals()[paramName] = thisTrials_ast[paramName]
        
        # --- Prepare to start Routine "ast_prestim" ---
        # create an object to store info about Routine ast_prestim
        ast_prestim = data.Routine(
            name='ast_prestim',
            components=[black_screen_9, cross_13],
        )
        ast_prestim.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_3
        sound_path = ast_stims[ast_idx]
        name = os.path.basename(sound_path).split('.')[0]
        marker_outlet.push_sample([f"ast_prestim-name"])
        # store start times for ast_prestim
        ast_prestim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        ast_prestim.tStart = globalClock.getTime(format='float')
        ast_prestim.status = STARTED
        thisExp.addData('ast_prestim.started', ast_prestim.tStart)
        ast_prestim.maxDuration = None
        # keep track of which components have finished
        ast_prestimComponents = ast_prestim.components
        for thisComponent in ast_prestim.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "ast_prestim" ---
        # if trial has changed, end Routine now
        if isinstance(trials_ast, data.TrialHandler2) and thisTrials_ast.thisN != trials_ast.thisTrial.thisN:
            continueRoutine = False
        ast_prestim.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *black_screen_9* updates
            
            # if black_screen_9 is starting this frame...
            if black_screen_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                black_screen_9.frameNStart = frameN  # exact frame index
                black_screen_9.tStart = t  # local t and not account for scr refresh
                black_screen_9.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_screen_9, 'tStartRefresh')  # time at next scr refresh
                # update status
                black_screen_9.status = STARTED
                black_screen_9.setAutoDraw(True)
            
            # if black_screen_9 is active this frame...
            if black_screen_9.status == STARTED:
                # update params
                pass
            
            # if black_screen_9 is stopping this frame...
            if black_screen_9.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black_screen_9.tStartRefresh + ast_prestim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    black_screen_9.tStop = t  # not accounting for scr refresh
                    black_screen_9.tStopRefresh = tThisFlipGlobal  # on global time
                    black_screen_9.frameNStop = frameN  # exact frame index
                    # update status
                    black_screen_9.status = FINISHED
                    black_screen_9.setAutoDraw(False)
            
            # *cross_13* updates
            
            # if cross_13 is starting this frame...
            if cross_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_13.frameNStart = frameN  # exact frame index
                cross_13.tStart = t  # local t and not account for scr refresh
                cross_13.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_13, 'tStartRefresh')  # time at next scr refresh
                # update status
                cross_13.status = STARTED
                cross_13.setAutoDraw(True)
            
            # if cross_13 is active this frame...
            if cross_13.status == STARTED:
                # update params
                pass
            
            # if cross_13 is stopping this frame...
            if cross_13.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_13.tStartRefresh + ast_prestim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_13.tStop = t  # not accounting for scr refresh
                    cross_13.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_13.frameNStop = frameN  # exact frame index
                    # update status
                    cross_13.status = FINISHED
                    cross_13.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                ast_prestim.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in ast_prestim.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "ast_prestim" ---
        for thisComponent in ast_prestim.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for ast_prestim
        ast_prestim.tStop = globalClock.getTime(format='float')
        ast_prestim.tStopRefresh = tThisFlipGlobal
        thisExp.addData('ast_prestim.stopped', ast_prestim.tStop)
        # the Routine "ast_prestim" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "stim" ---
        # create an object to store info about Routine stim
        stim = data.Routine(
            name='stim',
            components=[current_sound, cross_14],
        )
        stim.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from pupil_code
        current_sound.setSound(sound_path, hamming=True)
        marker_outlet.push_sample([f"ast_stim-{name}"])
        sound_duration = current_sound.getDuration()
        print(f'{name}: {sound_duration}')
        print(ast_idx)
        if ast_idx < len(ast_stims)-1:
            ast_idx+=1
        else:
            ast_idx = 0
        current_sound.setSound(sound_path, secs=ast_stim_duration, hamming=True)
        current_sound.setVolume(1.0, log=False)
        current_sound.seek(0)
        # store start times for stim
        stim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        stim.tStart = globalClock.getTime(format='float')
        stim.status = STARTED
        thisExp.addData('stim.started', stim.tStart)
        stim.maxDuration = None
        # keep track of which components have finished
        stimComponents = stim.components
        for thisComponent in stim.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "stim" ---
        # if trial has changed, end Routine now
        if isinstance(trials_ast, data.TrialHandler2) and thisTrials_ast.thisN != trials_ast.thisTrial.thisN:
            continueRoutine = False
        stim.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *current_sound* updates
            
            # if current_sound is starting this frame...
            if current_sound.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                current_sound.frameNStart = frameN  # exact frame index
                current_sound.tStart = t  # local t and not account for scr refresh
                current_sound.tStartRefresh = tThisFlipGlobal  # on global time
                # add timestamp to datafile
                thisExp.addData('current_sound.started', tThisFlipGlobal)
                # update status
                current_sound.status = STARTED
                current_sound.play(when=win)  # sync with win flip
            
            # if current_sound is stopping this frame...
            if current_sound.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > current_sound.tStartRefresh + ast_stim_duration-frameTolerance or current_sound.isFinished:
                    # keep track of stop time/frame for later
                    current_sound.tStop = t  # not accounting for scr refresh
                    current_sound.tStopRefresh = tThisFlipGlobal  # on global time
                    current_sound.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'current_sound.stopped')
                    # update status
                    current_sound.status = FINISHED
                    current_sound.stop()
            
            # *cross_14* updates
            
            # if cross_14 is starting this frame...
            if cross_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_14.frameNStart = frameN  # exact frame index
                cross_14.tStart = t  # local t and not account for scr refresh
                cross_14.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_14, 'tStartRefresh')  # time at next scr refresh
                # update status
                cross_14.status = STARTED
                cross_14.setAutoDraw(True)
            
            # if cross_14 is active this frame...
            if cross_14.status == STARTED:
                # update params
                pass
            
            # if cross_14 is stopping this frame...
            if cross_14.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_14.tStartRefresh + ast_stim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_14.tStop = t  # not accounting for scr refresh
                    cross_14.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_14.frameNStop = frameN  # exact frame index
                    # update status
                    cross_14.status = FINISHED
                    cross_14.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[current_sound]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                stim.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in stim.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "stim" ---
        for thisComponent in stim.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for stim
        stim.tStop = globalClock.getTime(format='float')
        stim.tStopRefresh = tThisFlipGlobal
        thisExp.addData('stim.stopped', stim.tStop)
        current_sound.pause()  # ensure sound has stopped at end of Routine
        # the Routine "stim" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "post_stim" ---
        # create an object to store info about Routine post_stim
        post_stim = data.Routine(
            name='post_stim',
            components=[black_screen_10, cross_15],
        )
        post_stim.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_increment
        marker_outlet.push_sample([f"ast_poststim-{name}"])
        # store start times for post_stim
        post_stim.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        post_stim.tStart = globalClock.getTime(format='float')
        post_stim.status = STARTED
        thisExp.addData('post_stim.started', post_stim.tStart)
        post_stim.maxDuration = None
        # keep track of which components have finished
        post_stimComponents = post_stim.components
        for thisComponent in post_stim.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "post_stim" ---
        # if trial has changed, end Routine now
        if isinstance(trials_ast, data.TrialHandler2) and thisTrials_ast.thisN != trials_ast.thisTrial.thisN:
            continueRoutine = False
        post_stim.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *black_screen_10* updates
            
            # if black_screen_10 is starting this frame...
            if black_screen_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                black_screen_10.frameNStart = frameN  # exact frame index
                black_screen_10.tStart = t  # local t and not account for scr refresh
                black_screen_10.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(black_screen_10, 'tStartRefresh')  # time at next scr refresh
                # update status
                black_screen_10.status = STARTED
                black_screen_10.setAutoDraw(True)
            
            # if black_screen_10 is active this frame...
            if black_screen_10.status == STARTED:
                # update params
                pass
            
            # if black_screen_10 is stopping this frame...
            if black_screen_10.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > black_screen_10.tStartRefresh + ast_poststim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    black_screen_10.tStop = t  # not accounting for scr refresh
                    black_screen_10.tStopRefresh = tThisFlipGlobal  # on global time
                    black_screen_10.frameNStop = frameN  # exact frame index
                    # update status
                    black_screen_10.status = FINISHED
                    black_screen_10.setAutoDraw(False)
            
            # *cross_15* updates
            
            # if cross_15 is starting this frame...
            if cross_15.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cross_15.frameNStart = frameN  # exact frame index
                cross_15.tStart = t  # local t and not account for scr refresh
                cross_15.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_15, 'tStartRefresh')  # time at next scr refresh
                # update status
                cross_15.status = STARTED
                cross_15.setAutoDraw(True)
            
            # if cross_15 is active this frame...
            if cross_15.status == STARTED:
                # update params
                pass
            
            # if cross_15 is stopping this frame...
            if cross_15.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_15.tStartRefresh + ast_poststim_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_15.tStop = t  # not accounting for scr refresh
                    cross_15.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_15.frameNStop = frameN  # exact frame index
                    # update status
                    cross_15.status = FINISHED
                    cross_15.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                post_stim.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in post_stim.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "post_stim" ---
        for thisComponent in post_stim.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for post_stim
        post_stim.tStop = globalClock.getTime(format='float')
        post_stim.tStopRefresh = tThisFlipGlobal
        thisExp.addData('post_stim.stopped', post_stim.tStop)
        # the Routine "post_stim" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed ast_trials repeats of 'trials_ast'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # Run 'End Experiment' code from code_init
    marker_outlet.push_sample(["end"])
    core.wait(1)
    lsl_socket.sendall(b"stop\n")
    
    print("Saving Response")
    processed_data_with_type = []
    participant = expInfo['participant']
    exp_date = expInfo['date']
    path = os.path.join('data',participant,f"{participant}_{expName}_{exp_date}.csv")
    d = pd.read_csv(path)
    
    for idx, row in d.iterrows():
        if not pd.isna(row['current_hlt_stim']):
            user_value = row['trials_hlt.slider_4.response'] if not pd.isna(row['trials_hlt.slider_4.response']) else "No Value"
            user_rt = row['slider_4.rt'] if not pd.isna(row['slider_4.rt']) else "No RT"
            processed_data_with_type.append({
                "Stim Type": "HLT",
                "Stim Name": row['current_hlt_stim'],
                "Repeat Number": int(row['thisRepN']) + 1 if not pd.isna(row['thisRepN']) else None,
                "User Value": user_value,
                "User Response Time (s)": f"{user_rt:.3f}"
            })
        
        if not pd.isna(row['current_let_stim']):
            user_value = row['slider_4.response'] if not pd.isna(row['slider_4.response']) else "No Value"
            for response_col, rt_col in [('trials_let.slider_5.response', 'slider_5.rt'),
                                                 ('trials_let.slider_6.response', 'slider_6.rt'),
                                                 ('trials_let.slider_7.response', 'slider_7.rt')]:
                if not pd.isna(row[response_col]):
                    user_value = row[response_col]
                    user_rt = row[rt_col]
                    break
                
            user_value = user_value if user_value is not None else "No Value"
            user_rt = user_rt if user_rt is not None else "No RT"
            processed_data_with_type.append({
                "Stim Type": "LET",
                "Stim Name": row['current_let_stim'],
                "Repeat Number": int(row['thisRepN']) + 1 if not pd.isna(row['thisRepN']) else None,
                "User Value": user_value,
                "User Response Time (s)": f"{user_rt:.3f}"
            })
            
    processed_df_with_type = pd.DataFrame(processed_data_with_type)
    response_path = os.path.join(os.getcwd(),"exp_data",participant,f'{participant}_response.csv') 
    processed_df_with_type.to_csv(response_path, index=False)
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='iso'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
