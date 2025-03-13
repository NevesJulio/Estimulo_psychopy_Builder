#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on março 13, 2025, at 01:49
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

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'Estimulo_ICE'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
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
_fullScr = True
_winSize = (1024, 768)
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
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Julio\\Desktop\\Programacao\\stimulus\\Estimulo_ICE_lastrun.py',
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
            logging.getLevel('info')
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
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
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
    
    # --- Initialize components for Routine "codigo" ---
    
    # --- Initialize components for Routine "trial" ---
    image = visual.ImageStim(
        win=win,
        name='image', units='cm', 
        image='gaborVert.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', units='cm', 
        image='gaborVert.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "trial_2" ---
    image_3 = visual.ImageStim(
        win=win,
        name='image_3', units='cm', 
        image='gaborVert.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_4 = visual.ImageStim(
        win=win,
        name='image_4', units='cm', 
        image='gaborVert.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "trial" ---
    image = visual.ImageStim(
        win=win,
        name='image', units='cm', 
        image='gaborVert.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', units='cm', 
        image='gaborVert.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "trial_2" ---
    image_3 = visual.ImageStim(
        win=win,
        name='image_3', units='cm', 
        image='gaborVert.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_4 = visual.ImageStim(
        win=win,
        name='image_4', units='cm', 
        image='gaborVert.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "trial" ---
    image = visual.ImageStim(
        win=win,
        name='image', units='cm', 
        image='gaborVert.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', units='cm', 
        image='gaborVert.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "trial_2" ---
    image_3 = visual.ImageStim(
        win=win,
        name='image_3', units='cm', 
        image='gaborVert.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_4 = visual.ImageStim(
        win=win,
        name='image_4', units='cm', 
        image='gaborVert.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "trial" ---
    image = visual.ImageStim(
        win=win,
        name='image', units='cm', 
        image='gaborVert.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_2 = visual.ImageStim(
        win=win,
        name='image_2', units='cm', 
        image='gaborVert.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "trial_2" ---
    image_3 = visual.ImageStim(
        win=win,
        name='image_3', units='cm', 
        image='gaborVert.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_4 = visual.ImageStim(
        win=win,
        name='image_4', units='cm', 
        image='gaborVert.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "trial_3" ---
    image_5 = visual.ImageStim(
        win=win,
        name='image_5', units='cm', 
        image='gaborHoriz.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_6 = visual.ImageStim(
        win=win,
        name='image_6', units='cm', 
        image='gaborHoriz.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=True,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "trial_4" ---
    image_7 = visual.ImageStim(
        win=win,
        name='image_7', units='cm', 
        image='gaborHoriz.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_8 = visual.ImageStim(
        win=win,
        name='image_8', units='cm', 
        image='gaborHoriz.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "trial_3" ---
    image_5 = visual.ImageStim(
        win=win,
        name='image_5', units='cm', 
        image='gaborHoriz.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_6 = visual.ImageStim(
        win=win,
        name='image_6', units='cm', 
        image='gaborHoriz.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=True,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "trial_4" ---
    image_7 = visual.ImageStim(
        win=win,
        name='image_7', units='cm', 
        image='gaborHoriz.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_8 = visual.ImageStim(
        win=win,
        name='image_8', units='cm', 
        image='gaborHoriz.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "trial_3" ---
    image_5 = visual.ImageStim(
        win=win,
        name='image_5', units='cm', 
        image='gaborHoriz.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_6 = visual.ImageStim(
        win=win,
        name='image_6', units='cm', 
        image='gaborHoriz.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=True,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "trial_4" ---
    image_7 = visual.ImageStim(
        win=win,
        name='image_7', units='cm', 
        image='gaborHoriz.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_8 = visual.ImageStim(
        win=win,
        name='image_8', units='cm', 
        image='gaborHoriz.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "trial_3" ---
    image_5 = visual.ImageStim(
        win=win,
        name='image_5', units='cm', 
        image='gaborHoriz.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_6 = visual.ImageStim(
        win=win,
        name='image_6', units='cm', 
        image='gaborHoriz.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=True,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "trial_4" ---
    image_7 = visual.ImageStim(
        win=win,
        name='image_7', units='cm', 
        image='gaborHoriz.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_8 = visual.ImageStim(
        win=win,
        name='image_8', units='cm', 
        image='gaborHoriz.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=(2, 2),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
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
    
    # set up handler to look after randomisation of conditions etc
    MainLoop = data.TrialHandler2(
        name='MainLoop',
        nReps=8.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Estimulos.csv'), 
        seed=None, 
    )
    thisExp.addLoop(MainLoop)  # add the loop to the experiment
    thisMainLoop = MainLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisMainLoop.rgb)
    if thisMainLoop != None:
        for paramName in thisMainLoop:
            globals()[paramName] = thisMainLoop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisMainLoop in MainLoop:
        currentLoop = MainLoop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisMainLoop.rgb)
        if thisMainLoop != None:
            for paramName in thisMainLoop:
                globals()[paramName] = thisMainLoop[paramName]
        
        # --- Prepare to start Routine "codigo" ---
        # create an object to store info about Routine codigo
        codigo = data.Routine(
            name='codigo',
            components=[],
        )
        codigo.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for codigo
        codigo.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        codigo.tStart = globalClock.getTime(format='float')
        codigo.status = STARTED
        thisExp.addData('codigo.started', codigo.tStart)
        codigo.maxDuration = None
        # keep track of which components have finished
        codigoComponents = codigo.components
        for thisComponent in codigo.components:
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
        
        # --- Run Routine "codigo" ---
        # if trial has changed, end Routine now
        if isinstance(MainLoop, data.TrialHandler2) and thisMainLoop.thisN != MainLoop.thisTrial.thisN:
            continueRoutine = False
        codigo.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from code
            if LoopName == 'Estimulo_1':
                rep1 = 1
                rep2 = 0
                rep3 = 0
                rep4 = 0
                rep5 = 1
                rep6 = 0
                rep7 = 0
                rep8 = 0
            elif LoopName == 'Estimulo_2':
                rep1 = 0
                rep2 = 1
                rep3 = 0
                rep4 = 0
                rep5 = 0
                rep6 = 0
                rep7 = 0
                rep8 = 0
            elif LoopName == 'Estimulo_3':
                rep1 = 0
                rep2 = 0
                rep3 = 1
                rep4 = 0
                rep5 = 0
                rep6 = 0
                rep7 = 0
                rep8 = 0
            elif LoopName == 'Estimulo_4':
                rep1 = 0
                rep2 = 0
                rep3 = 0
                rep4 = 1
                rep5 = 0
                rep6 = 0
                rep7 = 0
                rep8 = 0
            elif LoopName == 'Estimulo_5':
                rep1 = 0
                rep2 = 0
                rep3 = 0
                rep4 = 0
                rep5 = 1
                rep6 = 0
                rep7 = 0
                rep8 = 0
            elif LoopName == 'Estimulo_6':
                rep1 = 0
                rep2 = 0
                rep3 = 0
                rep4 = 0
                rep5 = 0
                rep6 = 1
                rep7 = 0
                rep8 = 0
            elif LoopName == 'Estimulo_7':
                rep1 = 0
                rep2 = 0
                rep3 = 0
                rep4 = 0
                rep5 = 0
                rep6 = 0
                rep7 = 1
                rep8 = 0
            elif LoopName == 'Estimulo_8':
                rep1 = 0
                rep2 = 0
                rep3 = 0
                rep4 = 0
                rep5 = 0
                rep6 = 0
                rep7 = 0
                rep8 = 1
            
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
                codigo.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in codigo.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "codigo" ---
        for thisComponent in codigo.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for codigo
        codigo.tStop = globalClock.getTime(format='float')
        codigo.tStopRefresh = tThisFlipGlobal
        thisExp.addData('codigo.stopped', codigo.tStop)
        # the Routine "codigo" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        loop_1 = data.TrialHandler2(
            name='loop_1',
            nReps=rep1, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('Incrementos.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(loop_1)  # add the loop to the experiment
        thisLoop_1 = loop_1.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_1.rgb)
        if thisLoop_1 != None:
            for paramName in thisLoop_1:
                globals()[paramName] = thisLoop_1[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisLoop_1 in loop_1:
            currentLoop = loop_1
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_1.rgb)
            if thisLoop_1 != None:
                for paramName in thisLoop_1:
                    globals()[paramName] = thisLoop_1[paramName]
            
            # --- Prepare to start Routine "trial" ---
            # create an object to store info about Routine trial
            trial = data.Routine(
                name='trial',
                components=[image, image_2],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image.setPos((px1, py1))
            image_2.setPos((px2, py2))
            # store start times for trial
            trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial.tStart = globalClock.getTime(format='float')
            trial.status = STARTED
            thisExp.addData('trial.started', trial.tStart)
            trial.maxDuration = None
            # keep track of which components have finished
            trialComponents = trial.components
            for thisComponent in trial.components:
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
            
            # --- Run Routine "trial" ---
            # if trial has changed, end Routine now
            if isinstance(loop_1, data.TrialHandler2) and thisLoop_1.thisN != loop_1.thisTrial.thisN:
                continueRoutine = False
            trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image* updates
                
                # if image is starting this frame...
                if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image.frameNStart = frameN  # exact frame index
                    image.tStart = t  # local t and not account for scr refresh
                    image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.started')
                    # update status
                    image.status = STARTED
                    image.setAutoDraw(True)
                
                # if image is active this frame...
                if image.status == STARTED:
                    # update params
                    pass
                
                # if image is stopping this frame...
                if image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image.tStop = t  # not accounting for scr refresh
                        image.tStopRefresh = tThisFlipGlobal  # on global time
                        image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image.stopped')
                        # update status
                        image.status = FINISHED
                        image.setAutoDraw(False)
                
                # *image_2* updates
                
                # if image_2 is starting this frame...
                if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_2.frameNStart = frameN  # exact frame index
                    image_2.tStart = t  # local t and not account for scr refresh
                    image_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_2.started')
                    # update status
                    image_2.status = STARTED
                    image_2.setAutoDraw(True)
                
                # if image_2 is active this frame...
                if image_2.status == STARTED:
                    # update params
                    pass
                
                # if image_2 is stopping this frame...
                if image_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_2.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_2.tStop = t  # not accounting for scr refresh
                        image_2.tStopRefresh = tThisFlipGlobal  # on global time
                        image_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_2.stopped')
                        # update status
                        image_2.status = FINISHED
                        image_2.setAutoDraw(False)
                
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
                    trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial
            trial.tStop = globalClock.getTime(format='float')
            trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial.stopped', trial.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial.maxDurationReached:
                routineTimer.addTime(-trial.maxDuration)
            elif trial.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            
            # --- Prepare to start Routine "trial_2" ---
            # create an object to store info about Routine trial_2
            trial_2 = data.Routine(
                name='trial_2',
                components=[image_3, image_4],
            )
            trial_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_3.setPos((px3, py3))
            image_4.setPos((px4,py4))
            # store start times for trial_2
            trial_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_2.tStart = globalClock.getTime(format='float')
            trial_2.status = STARTED
            thisExp.addData('trial_2.started', trial_2.tStart)
            trial_2.maxDuration = None
            # keep track of which components have finished
            trial_2Components = trial_2.components
            for thisComponent in trial_2.components:
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
            
            # --- Run Routine "trial_2" ---
            # if trial has changed, end Routine now
            if isinstance(loop_1, data.TrialHandler2) and thisLoop_1.thisN != loop_1.thisTrial.thisN:
                continueRoutine = False
            trial_2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_3* updates
                
                # if image_3 is starting this frame...
                if image_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_3.frameNStart = frameN  # exact frame index
                    image_3.tStart = t  # local t and not account for scr refresh
                    image_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_3.started')
                    # update status
                    image_3.status = STARTED
                    image_3.setAutoDraw(True)
                
                # if image_3 is active this frame...
                if image_3.status == STARTED:
                    # update params
                    pass
                
                # if image_3 is stopping this frame...
                if image_3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_3.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_3.tStop = t  # not accounting for scr refresh
                        image_3.tStopRefresh = tThisFlipGlobal  # on global time
                        image_3.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_3.stopped')
                        # update status
                        image_3.status = FINISHED
                        image_3.setAutoDraw(False)
                
                # *image_4* updates
                
                # if image_4 is starting this frame...
                if image_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_4.frameNStart = frameN  # exact frame index
                    image_4.tStart = t  # local t and not account for scr refresh
                    image_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_4.started')
                    # update status
                    image_4.status = STARTED
                    image_4.setAutoDraw(True)
                
                # if image_4 is active this frame...
                if image_4.status == STARTED:
                    # update params
                    pass
                
                # if image_4 is stopping this frame...
                if image_4.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_4.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_4.tStop = t  # not accounting for scr refresh
                        image_4.tStopRefresh = tThisFlipGlobal  # on global time
                        image_4.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_4.stopped')
                        # update status
                        image_4.status = FINISHED
                        image_4.setAutoDraw(False)
                
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
                    trial_2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_2" ---
            for thisComponent in trial_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_2
            trial_2.tStop = globalClock.getTime(format='float')
            trial_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_2.stopped', trial_2.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial_2.maxDurationReached:
                routineTimer.addTime(-trial_2.maxDuration)
            elif trial_2.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            thisExp.nextEntry()
            
        # completed rep1 repeats of 'loop_1'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # set up handler to look after randomisation of conditions etc
        loop_2 = data.TrialHandler2(
            name='loop_2',
            nReps=rep2, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('incremento_loop2.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(loop_2)  # add the loop to the experiment
        thisLoop_2 = loop_2.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_2.rgb)
        if thisLoop_2 != None:
            for paramName in thisLoop_2:
                globals()[paramName] = thisLoop_2[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisLoop_2 in loop_2:
            currentLoop = loop_2
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_2.rgb)
            if thisLoop_2 != None:
                for paramName in thisLoop_2:
                    globals()[paramName] = thisLoop_2[paramName]
            
            # --- Prepare to start Routine "trial" ---
            # create an object to store info about Routine trial
            trial = data.Routine(
                name='trial',
                components=[image, image_2],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image.setPos((px1, py1))
            image_2.setPos((px2, py2))
            # store start times for trial
            trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial.tStart = globalClock.getTime(format='float')
            trial.status = STARTED
            thisExp.addData('trial.started', trial.tStart)
            trial.maxDuration = None
            # keep track of which components have finished
            trialComponents = trial.components
            for thisComponent in trial.components:
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
            
            # --- Run Routine "trial" ---
            # if trial has changed, end Routine now
            if isinstance(loop_2, data.TrialHandler2) and thisLoop_2.thisN != loop_2.thisTrial.thisN:
                continueRoutine = False
            trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image* updates
                
                # if image is starting this frame...
                if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image.frameNStart = frameN  # exact frame index
                    image.tStart = t  # local t and not account for scr refresh
                    image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.started')
                    # update status
                    image.status = STARTED
                    image.setAutoDraw(True)
                
                # if image is active this frame...
                if image.status == STARTED:
                    # update params
                    pass
                
                # if image is stopping this frame...
                if image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image.tStop = t  # not accounting for scr refresh
                        image.tStopRefresh = tThisFlipGlobal  # on global time
                        image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image.stopped')
                        # update status
                        image.status = FINISHED
                        image.setAutoDraw(False)
                
                # *image_2* updates
                
                # if image_2 is starting this frame...
                if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_2.frameNStart = frameN  # exact frame index
                    image_2.tStart = t  # local t and not account for scr refresh
                    image_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_2.started')
                    # update status
                    image_2.status = STARTED
                    image_2.setAutoDraw(True)
                
                # if image_2 is active this frame...
                if image_2.status == STARTED:
                    # update params
                    pass
                
                # if image_2 is stopping this frame...
                if image_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_2.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_2.tStop = t  # not accounting for scr refresh
                        image_2.tStopRefresh = tThisFlipGlobal  # on global time
                        image_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_2.stopped')
                        # update status
                        image_2.status = FINISHED
                        image_2.setAutoDraw(False)
                
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
                    trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial
            trial.tStop = globalClock.getTime(format='float')
            trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial.stopped', trial.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial.maxDurationReached:
                routineTimer.addTime(-trial.maxDuration)
            elif trial.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            
            # --- Prepare to start Routine "trial_2" ---
            # create an object to store info about Routine trial_2
            trial_2 = data.Routine(
                name='trial_2',
                components=[image_3, image_4],
            )
            trial_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_3.setPos((px3, py3))
            image_4.setPos((px4,py4))
            # store start times for trial_2
            trial_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_2.tStart = globalClock.getTime(format='float')
            trial_2.status = STARTED
            thisExp.addData('trial_2.started', trial_2.tStart)
            trial_2.maxDuration = None
            # keep track of which components have finished
            trial_2Components = trial_2.components
            for thisComponent in trial_2.components:
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
            
            # --- Run Routine "trial_2" ---
            # if trial has changed, end Routine now
            if isinstance(loop_2, data.TrialHandler2) and thisLoop_2.thisN != loop_2.thisTrial.thisN:
                continueRoutine = False
            trial_2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_3* updates
                
                # if image_3 is starting this frame...
                if image_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_3.frameNStart = frameN  # exact frame index
                    image_3.tStart = t  # local t and not account for scr refresh
                    image_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_3.started')
                    # update status
                    image_3.status = STARTED
                    image_3.setAutoDraw(True)
                
                # if image_3 is active this frame...
                if image_3.status == STARTED:
                    # update params
                    pass
                
                # if image_3 is stopping this frame...
                if image_3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_3.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_3.tStop = t  # not accounting for scr refresh
                        image_3.tStopRefresh = tThisFlipGlobal  # on global time
                        image_3.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_3.stopped')
                        # update status
                        image_3.status = FINISHED
                        image_3.setAutoDraw(False)
                
                # *image_4* updates
                
                # if image_4 is starting this frame...
                if image_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_4.frameNStart = frameN  # exact frame index
                    image_4.tStart = t  # local t and not account for scr refresh
                    image_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_4.started')
                    # update status
                    image_4.status = STARTED
                    image_4.setAutoDraw(True)
                
                # if image_4 is active this frame...
                if image_4.status == STARTED:
                    # update params
                    pass
                
                # if image_4 is stopping this frame...
                if image_4.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_4.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_4.tStop = t  # not accounting for scr refresh
                        image_4.tStopRefresh = tThisFlipGlobal  # on global time
                        image_4.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_4.stopped')
                        # update status
                        image_4.status = FINISHED
                        image_4.setAutoDraw(False)
                
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
                    trial_2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_2" ---
            for thisComponent in trial_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_2
            trial_2.tStop = globalClock.getTime(format='float')
            trial_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_2.stopped', trial_2.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial_2.maxDurationReached:
                routineTimer.addTime(-trial_2.maxDuration)
            elif trial_2.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            thisExp.nextEntry()
            
        # completed rep2 repeats of 'loop_2'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # set up handler to look after randomisation of conditions etc
        loop_3 = data.TrialHandler2(
            name='loop_3',
            nReps=rep3, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('incremento_loop3.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(loop_3)  # add the loop to the experiment
        thisLoop_3 = loop_3.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_3.rgb)
        if thisLoop_3 != None:
            for paramName in thisLoop_3:
                globals()[paramName] = thisLoop_3[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisLoop_3 in loop_3:
            currentLoop = loop_3
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_3.rgb)
            if thisLoop_3 != None:
                for paramName in thisLoop_3:
                    globals()[paramName] = thisLoop_3[paramName]
            
            # --- Prepare to start Routine "trial" ---
            # create an object to store info about Routine trial
            trial = data.Routine(
                name='trial',
                components=[image, image_2],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image.setPos((px1, py1))
            image_2.setPos((px2, py2))
            # store start times for trial
            trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial.tStart = globalClock.getTime(format='float')
            trial.status = STARTED
            thisExp.addData('trial.started', trial.tStart)
            trial.maxDuration = None
            # keep track of which components have finished
            trialComponents = trial.components
            for thisComponent in trial.components:
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
            
            # --- Run Routine "trial" ---
            # if trial has changed, end Routine now
            if isinstance(loop_3, data.TrialHandler2) and thisLoop_3.thisN != loop_3.thisTrial.thisN:
                continueRoutine = False
            trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image* updates
                
                # if image is starting this frame...
                if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image.frameNStart = frameN  # exact frame index
                    image.tStart = t  # local t and not account for scr refresh
                    image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.started')
                    # update status
                    image.status = STARTED
                    image.setAutoDraw(True)
                
                # if image is active this frame...
                if image.status == STARTED:
                    # update params
                    pass
                
                # if image is stopping this frame...
                if image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image.tStop = t  # not accounting for scr refresh
                        image.tStopRefresh = tThisFlipGlobal  # on global time
                        image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image.stopped')
                        # update status
                        image.status = FINISHED
                        image.setAutoDraw(False)
                
                # *image_2* updates
                
                # if image_2 is starting this frame...
                if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_2.frameNStart = frameN  # exact frame index
                    image_2.tStart = t  # local t and not account for scr refresh
                    image_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_2.started')
                    # update status
                    image_2.status = STARTED
                    image_2.setAutoDraw(True)
                
                # if image_2 is active this frame...
                if image_2.status == STARTED:
                    # update params
                    pass
                
                # if image_2 is stopping this frame...
                if image_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_2.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_2.tStop = t  # not accounting for scr refresh
                        image_2.tStopRefresh = tThisFlipGlobal  # on global time
                        image_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_2.stopped')
                        # update status
                        image_2.status = FINISHED
                        image_2.setAutoDraw(False)
                
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
                    trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial
            trial.tStop = globalClock.getTime(format='float')
            trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial.stopped', trial.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial.maxDurationReached:
                routineTimer.addTime(-trial.maxDuration)
            elif trial.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            
            # --- Prepare to start Routine "trial_2" ---
            # create an object to store info about Routine trial_2
            trial_2 = data.Routine(
                name='trial_2',
                components=[image_3, image_4],
            )
            trial_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_3.setPos((px3, py3))
            image_4.setPos((px4,py4))
            # store start times for trial_2
            trial_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_2.tStart = globalClock.getTime(format='float')
            trial_2.status = STARTED
            thisExp.addData('trial_2.started', trial_2.tStart)
            trial_2.maxDuration = None
            # keep track of which components have finished
            trial_2Components = trial_2.components
            for thisComponent in trial_2.components:
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
            
            # --- Run Routine "trial_2" ---
            # if trial has changed, end Routine now
            if isinstance(loop_3, data.TrialHandler2) and thisLoop_3.thisN != loop_3.thisTrial.thisN:
                continueRoutine = False
            trial_2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_3* updates
                
                # if image_3 is starting this frame...
                if image_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_3.frameNStart = frameN  # exact frame index
                    image_3.tStart = t  # local t and not account for scr refresh
                    image_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_3.started')
                    # update status
                    image_3.status = STARTED
                    image_3.setAutoDraw(True)
                
                # if image_3 is active this frame...
                if image_3.status == STARTED:
                    # update params
                    pass
                
                # if image_3 is stopping this frame...
                if image_3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_3.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_3.tStop = t  # not accounting for scr refresh
                        image_3.tStopRefresh = tThisFlipGlobal  # on global time
                        image_3.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_3.stopped')
                        # update status
                        image_3.status = FINISHED
                        image_3.setAutoDraw(False)
                
                # *image_4* updates
                
                # if image_4 is starting this frame...
                if image_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_4.frameNStart = frameN  # exact frame index
                    image_4.tStart = t  # local t and not account for scr refresh
                    image_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_4.started')
                    # update status
                    image_4.status = STARTED
                    image_4.setAutoDraw(True)
                
                # if image_4 is active this frame...
                if image_4.status == STARTED:
                    # update params
                    pass
                
                # if image_4 is stopping this frame...
                if image_4.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_4.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_4.tStop = t  # not accounting for scr refresh
                        image_4.tStopRefresh = tThisFlipGlobal  # on global time
                        image_4.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_4.stopped')
                        # update status
                        image_4.status = FINISHED
                        image_4.setAutoDraw(False)
                
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
                    trial_2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_2" ---
            for thisComponent in trial_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_2
            trial_2.tStop = globalClock.getTime(format='float')
            trial_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_2.stopped', trial_2.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial_2.maxDurationReached:
                routineTimer.addTime(-trial_2.maxDuration)
            elif trial_2.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            thisExp.nextEntry()
            
        # completed rep3 repeats of 'loop_3'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # set up handler to look after randomisation of conditions etc
        loop_4 = data.TrialHandler2(
            name='loop_4',
            nReps=rep4, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('Incrementos_loop_4.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(loop_4)  # add the loop to the experiment
        thisLoop_4 = loop_4.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_4.rgb)
        if thisLoop_4 != None:
            for paramName in thisLoop_4:
                globals()[paramName] = thisLoop_4[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisLoop_4 in loop_4:
            currentLoop = loop_4
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_4.rgb)
            if thisLoop_4 != None:
                for paramName in thisLoop_4:
                    globals()[paramName] = thisLoop_4[paramName]
            
            # --- Prepare to start Routine "trial" ---
            # create an object to store info about Routine trial
            trial = data.Routine(
                name='trial',
                components=[image, image_2],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image.setPos((px1, py1))
            image_2.setPos((px2, py2))
            # store start times for trial
            trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial.tStart = globalClock.getTime(format='float')
            trial.status = STARTED
            thisExp.addData('trial.started', trial.tStart)
            trial.maxDuration = None
            # keep track of which components have finished
            trialComponents = trial.components
            for thisComponent in trial.components:
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
            
            # --- Run Routine "trial" ---
            # if trial has changed, end Routine now
            if isinstance(loop_4, data.TrialHandler2) and thisLoop_4.thisN != loop_4.thisTrial.thisN:
                continueRoutine = False
            trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image* updates
                
                # if image is starting this frame...
                if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image.frameNStart = frameN  # exact frame index
                    image.tStart = t  # local t and not account for scr refresh
                    image.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image.started')
                    # update status
                    image.status = STARTED
                    image.setAutoDraw(True)
                
                # if image is active this frame...
                if image.status == STARTED:
                    # update params
                    pass
                
                # if image is stopping this frame...
                if image.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image.tStop = t  # not accounting for scr refresh
                        image.tStopRefresh = tThisFlipGlobal  # on global time
                        image.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image.stopped')
                        # update status
                        image.status = FINISHED
                        image.setAutoDraw(False)
                
                # *image_2* updates
                
                # if image_2 is starting this frame...
                if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_2.frameNStart = frameN  # exact frame index
                    image_2.tStart = t  # local t and not account for scr refresh
                    image_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_2.started')
                    # update status
                    image_2.status = STARTED
                    image_2.setAutoDraw(True)
                
                # if image_2 is active this frame...
                if image_2.status == STARTED:
                    # update params
                    pass
                
                # if image_2 is stopping this frame...
                if image_2.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_2.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_2.tStop = t  # not accounting for scr refresh
                        image_2.tStopRefresh = tThisFlipGlobal  # on global time
                        image_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_2.stopped')
                        # update status
                        image_2.status = FINISHED
                        image_2.setAutoDraw(False)
                
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
                    trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial
            trial.tStop = globalClock.getTime(format='float')
            trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial.stopped', trial.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial.maxDurationReached:
                routineTimer.addTime(-trial.maxDuration)
            elif trial.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            
            # --- Prepare to start Routine "trial_2" ---
            # create an object to store info about Routine trial_2
            trial_2 = data.Routine(
                name='trial_2',
                components=[image_3, image_4],
            )
            trial_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_3.setPos((px3, py3))
            image_4.setPos((px4,py4))
            # store start times for trial_2
            trial_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_2.tStart = globalClock.getTime(format='float')
            trial_2.status = STARTED
            thisExp.addData('trial_2.started', trial_2.tStart)
            trial_2.maxDuration = None
            # keep track of which components have finished
            trial_2Components = trial_2.components
            for thisComponent in trial_2.components:
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
            
            # --- Run Routine "trial_2" ---
            # if trial has changed, end Routine now
            if isinstance(loop_4, data.TrialHandler2) and thisLoop_4.thisN != loop_4.thisTrial.thisN:
                continueRoutine = False
            trial_2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *image_3* updates
                
                # if image_3 is starting this frame...
                if image_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_3.frameNStart = frameN  # exact frame index
                    image_3.tStart = t  # local t and not account for scr refresh
                    image_3.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_3, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_3.started')
                    # update status
                    image_3.status = STARTED
                    image_3.setAutoDraw(True)
                
                # if image_3 is active this frame...
                if image_3.status == STARTED:
                    # update params
                    pass
                
                # if image_3 is stopping this frame...
                if image_3.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_3.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_3.tStop = t  # not accounting for scr refresh
                        image_3.tStopRefresh = tThisFlipGlobal  # on global time
                        image_3.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_3.stopped')
                        # update status
                        image_3.status = FINISHED
                        image_3.setAutoDraw(False)
                
                # *image_4* updates
                
                # if image_4 is starting this frame...
                if image_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_4.frameNStart = frameN  # exact frame index
                    image_4.tStart = t  # local t and not account for scr refresh
                    image_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_4.started')
                    # update status
                    image_4.status = STARTED
                    image_4.setAutoDraw(True)
                
                # if image_4 is active this frame...
                if image_4.status == STARTED:
                    # update params
                    pass
                
                # if image_4 is stopping this frame...
                if image_4.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_4.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_4.tStop = t  # not accounting for scr refresh
                        image_4.tStopRefresh = tThisFlipGlobal  # on global time
                        image_4.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_4.stopped')
                        # update status
                        image_4.status = FINISHED
                        image_4.setAutoDraw(False)
                
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
                    trial_2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_2" ---
            for thisComponent in trial_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_2
            trial_2.tStop = globalClock.getTime(format='float')
            trial_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_2.stopped', trial_2.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial_2.maxDurationReached:
                routineTimer.addTime(-trial_2.maxDuration)
            elif trial_2.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            thisExp.nextEntry()
            
        # completed rep4 repeats of 'loop_4'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # set up handler to look after randomisation of conditions etc
        loop_5 = data.TrialHandler2(
            name='loop_5',
            nReps=rep5, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('Incrementos.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(loop_5)  # add the loop to the experiment
        thisLoop_5 = loop_5.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_5.rgb)
        if thisLoop_5 != None:
            for paramName in thisLoop_5:
                globals()[paramName] = thisLoop_5[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisLoop_5 in loop_5:
            currentLoop = loop_5
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_5.rgb)
            if thisLoop_5 != None:
                for paramName in thisLoop_5:
                    globals()[paramName] = thisLoop_5[paramName]
            
            # --- Prepare to start Routine "trial_3" ---
            # create an object to store info about Routine trial_3
            trial_3 = data.Routine(
                name='trial_3',
                components=[image_5, image_6],
            )
            trial_3.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_5.setPos((px1, py1))
            image_6.setPos((px2, py2))
            # store start times for trial_3
            trial_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_3.tStart = globalClock.getTime(format='float')
            trial_3.status = STARTED
            thisExp.addData('trial_3.started', trial_3.tStart)
            trial_3.maxDuration = 0.3
            # keep track of which components have finished
            trial_3Components = trial_3.components
            for thisComponent in trial_3.components:
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
            
            # --- Run Routine "trial_3" ---
            # if trial has changed, end Routine now
            if isinstance(loop_5, data.TrialHandler2) and thisLoop_5.thisN != loop_5.thisTrial.thisN:
                continueRoutine = False
            trial_3.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > trial_3.maxDuration-frameTolerance:
                    trial_3.maxDurationReached = True
                    continueRoutine = False
                
                # *image_5* updates
                
                # if image_5 is starting this frame...
                if image_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_5.frameNStart = frameN  # exact frame index
                    image_5.tStart = t  # local t and not account for scr refresh
                    image_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_5.started')
                    # update status
                    image_5.status = STARTED
                    image_5.setAutoDraw(True)
                
                # if image_5 is active this frame...
                if image_5.status == STARTED:
                    # update params
                    pass
                
                # if image_5 is stopping this frame...
                if image_5.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_5.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_5.tStop = t  # not accounting for scr refresh
                        image_5.tStopRefresh = tThisFlipGlobal  # on global time
                        image_5.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_5.stopped')
                        # update status
                        image_5.status = FINISHED
                        image_5.setAutoDraw(False)
                
                # *image_6* updates
                
                # if image_6 is starting this frame...
                if image_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_6.frameNStart = frameN  # exact frame index
                    image_6.tStart = t  # local t and not account for scr refresh
                    image_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_6, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_6.started')
                    # update status
                    image_6.status = STARTED
                    image_6.setAutoDraw(True)
                
                # if image_6 is active this frame...
                if image_6.status == STARTED:
                    # update params
                    pass
                
                # if image_6 is stopping this frame...
                if image_6.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_6.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_6.tStop = t  # not accounting for scr refresh
                        image_6.tStopRefresh = tThisFlipGlobal  # on global time
                        image_6.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_6.stopped')
                        # update status
                        image_6.status = FINISHED
                        image_6.setAutoDraw(False)
                
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
                    trial_3.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_3.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_3" ---
            for thisComponent in trial_3.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_3
            trial_3.tStop = globalClock.getTime(format='float')
            trial_3.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_3.stopped', trial_3.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial_3.maxDurationReached:
                routineTimer.addTime(-trial_3.maxDuration)
            elif trial_3.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            
            # --- Prepare to start Routine "trial_4" ---
            # create an object to store info about Routine trial_4
            trial_4 = data.Routine(
                name='trial_4',
                components=[image_7, image_8],
            )
            trial_4.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_7.setPos((px3, py3))
            image_8.setPos((px4, py4))
            # store start times for trial_4
            trial_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_4.tStart = globalClock.getTime(format='float')
            trial_4.status = STARTED
            thisExp.addData('trial_4.started', trial_4.tStart)
            trial_4.maxDuration = 0.3
            # keep track of which components have finished
            trial_4Components = trial_4.components
            for thisComponent in trial_4.components:
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
            
            # --- Run Routine "trial_4" ---
            # if trial has changed, end Routine now
            if isinstance(loop_5, data.TrialHandler2) and thisLoop_5.thisN != loop_5.thisTrial.thisN:
                continueRoutine = False
            trial_4.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > trial_4.maxDuration-frameTolerance:
                    trial_4.maxDurationReached = True
                    continueRoutine = False
                
                # *image_7* updates
                
                # if image_7 is starting this frame...
                if image_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_7.frameNStart = frameN  # exact frame index
                    image_7.tStart = t  # local t and not account for scr refresh
                    image_7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_7, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_7.started')
                    # update status
                    image_7.status = STARTED
                    image_7.setAutoDraw(True)
                
                # if image_7 is active this frame...
                if image_7.status == STARTED:
                    # update params
                    pass
                
                # if image_7 is stopping this frame...
                if image_7.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_7.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_7.tStop = t  # not accounting for scr refresh
                        image_7.tStopRefresh = tThisFlipGlobal  # on global time
                        image_7.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_7.stopped')
                        # update status
                        image_7.status = FINISHED
                        image_7.setAutoDraw(False)
                
                # *image_8* updates
                
                # if image_8 is starting this frame...
                if image_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_8.frameNStart = frameN  # exact frame index
                    image_8.tStart = t  # local t and not account for scr refresh
                    image_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_8, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_8.started')
                    # update status
                    image_8.status = STARTED
                    image_8.setAutoDraw(True)
                
                # if image_8 is active this frame...
                if image_8.status == STARTED:
                    # update params
                    pass
                
                # if image_8 is stopping this frame...
                if image_8.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_8.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_8.tStop = t  # not accounting for scr refresh
                        image_8.tStopRefresh = tThisFlipGlobal  # on global time
                        image_8.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_8.stopped')
                        # update status
                        image_8.status = FINISHED
                        image_8.setAutoDraw(False)
                
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
                    trial_4.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_4.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_4" ---
            for thisComponent in trial_4.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_4
            trial_4.tStop = globalClock.getTime(format='float')
            trial_4.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_4.stopped', trial_4.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial_4.maxDurationReached:
                routineTimer.addTime(-trial_4.maxDuration)
            elif trial_4.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            thisExp.nextEntry()
            
        # completed rep5 repeats of 'loop_5'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # set up handler to look after randomisation of conditions etc
        loop_6 = data.TrialHandler2(
            name='loop_6',
            nReps=rep6, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('incremento_loop2.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(loop_6)  # add the loop to the experiment
        thisLoop_6 = loop_6.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_6.rgb)
        if thisLoop_6 != None:
            for paramName in thisLoop_6:
                globals()[paramName] = thisLoop_6[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisLoop_6 in loop_6:
            currentLoop = loop_6
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_6.rgb)
            if thisLoop_6 != None:
                for paramName in thisLoop_6:
                    globals()[paramName] = thisLoop_6[paramName]
            
            # --- Prepare to start Routine "trial_3" ---
            # create an object to store info about Routine trial_3
            trial_3 = data.Routine(
                name='trial_3',
                components=[image_5, image_6],
            )
            trial_3.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_5.setPos((px1, py1))
            image_6.setPos((px2, py2))
            # store start times for trial_3
            trial_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_3.tStart = globalClock.getTime(format='float')
            trial_3.status = STARTED
            thisExp.addData('trial_3.started', trial_3.tStart)
            trial_3.maxDuration = 0.3
            # keep track of which components have finished
            trial_3Components = trial_3.components
            for thisComponent in trial_3.components:
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
            
            # --- Run Routine "trial_3" ---
            # if trial has changed, end Routine now
            if isinstance(loop_6, data.TrialHandler2) and thisLoop_6.thisN != loop_6.thisTrial.thisN:
                continueRoutine = False
            trial_3.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > trial_3.maxDuration-frameTolerance:
                    trial_3.maxDurationReached = True
                    continueRoutine = False
                
                # *image_5* updates
                
                # if image_5 is starting this frame...
                if image_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_5.frameNStart = frameN  # exact frame index
                    image_5.tStart = t  # local t and not account for scr refresh
                    image_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_5.started')
                    # update status
                    image_5.status = STARTED
                    image_5.setAutoDraw(True)
                
                # if image_5 is active this frame...
                if image_5.status == STARTED:
                    # update params
                    pass
                
                # if image_5 is stopping this frame...
                if image_5.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_5.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_5.tStop = t  # not accounting for scr refresh
                        image_5.tStopRefresh = tThisFlipGlobal  # on global time
                        image_5.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_5.stopped')
                        # update status
                        image_5.status = FINISHED
                        image_5.setAutoDraw(False)
                
                # *image_6* updates
                
                # if image_6 is starting this frame...
                if image_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_6.frameNStart = frameN  # exact frame index
                    image_6.tStart = t  # local t and not account for scr refresh
                    image_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_6, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_6.started')
                    # update status
                    image_6.status = STARTED
                    image_6.setAutoDraw(True)
                
                # if image_6 is active this frame...
                if image_6.status == STARTED:
                    # update params
                    pass
                
                # if image_6 is stopping this frame...
                if image_6.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_6.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_6.tStop = t  # not accounting for scr refresh
                        image_6.tStopRefresh = tThisFlipGlobal  # on global time
                        image_6.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_6.stopped')
                        # update status
                        image_6.status = FINISHED
                        image_6.setAutoDraw(False)
                
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
                    trial_3.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_3.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_3" ---
            for thisComponent in trial_3.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_3
            trial_3.tStop = globalClock.getTime(format='float')
            trial_3.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_3.stopped', trial_3.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial_3.maxDurationReached:
                routineTimer.addTime(-trial_3.maxDuration)
            elif trial_3.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            
            # --- Prepare to start Routine "trial_4" ---
            # create an object to store info about Routine trial_4
            trial_4 = data.Routine(
                name='trial_4',
                components=[image_7, image_8],
            )
            trial_4.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_7.setPos((px3, py3))
            image_8.setPos((px4, py4))
            # store start times for trial_4
            trial_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_4.tStart = globalClock.getTime(format='float')
            trial_4.status = STARTED
            thisExp.addData('trial_4.started', trial_4.tStart)
            trial_4.maxDuration = 0.3
            # keep track of which components have finished
            trial_4Components = trial_4.components
            for thisComponent in trial_4.components:
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
            
            # --- Run Routine "trial_4" ---
            # if trial has changed, end Routine now
            if isinstance(loop_6, data.TrialHandler2) and thisLoop_6.thisN != loop_6.thisTrial.thisN:
                continueRoutine = False
            trial_4.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > trial_4.maxDuration-frameTolerance:
                    trial_4.maxDurationReached = True
                    continueRoutine = False
                
                # *image_7* updates
                
                # if image_7 is starting this frame...
                if image_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_7.frameNStart = frameN  # exact frame index
                    image_7.tStart = t  # local t and not account for scr refresh
                    image_7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_7, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_7.started')
                    # update status
                    image_7.status = STARTED
                    image_7.setAutoDraw(True)
                
                # if image_7 is active this frame...
                if image_7.status == STARTED:
                    # update params
                    pass
                
                # if image_7 is stopping this frame...
                if image_7.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_7.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_7.tStop = t  # not accounting for scr refresh
                        image_7.tStopRefresh = tThisFlipGlobal  # on global time
                        image_7.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_7.stopped')
                        # update status
                        image_7.status = FINISHED
                        image_7.setAutoDraw(False)
                
                # *image_8* updates
                
                # if image_8 is starting this frame...
                if image_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_8.frameNStart = frameN  # exact frame index
                    image_8.tStart = t  # local t and not account for scr refresh
                    image_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_8, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_8.started')
                    # update status
                    image_8.status = STARTED
                    image_8.setAutoDraw(True)
                
                # if image_8 is active this frame...
                if image_8.status == STARTED:
                    # update params
                    pass
                
                # if image_8 is stopping this frame...
                if image_8.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_8.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_8.tStop = t  # not accounting for scr refresh
                        image_8.tStopRefresh = tThisFlipGlobal  # on global time
                        image_8.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_8.stopped')
                        # update status
                        image_8.status = FINISHED
                        image_8.setAutoDraw(False)
                
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
                    trial_4.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_4.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_4" ---
            for thisComponent in trial_4.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_4
            trial_4.tStop = globalClock.getTime(format='float')
            trial_4.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_4.stopped', trial_4.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial_4.maxDurationReached:
                routineTimer.addTime(-trial_4.maxDuration)
            elif trial_4.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            thisExp.nextEntry()
            
        # completed rep6 repeats of 'loop_6'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # set up handler to look after randomisation of conditions etc
        loop_7 = data.TrialHandler2(
            name='loop_7',
            nReps=rep7, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('incremento_loop3.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(loop_7)  # add the loop to the experiment
        thisLoop_7 = loop_7.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_7.rgb)
        if thisLoop_7 != None:
            for paramName in thisLoop_7:
                globals()[paramName] = thisLoop_7[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisLoop_7 in loop_7:
            currentLoop = loop_7
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_7.rgb)
            if thisLoop_7 != None:
                for paramName in thisLoop_7:
                    globals()[paramName] = thisLoop_7[paramName]
            
            # --- Prepare to start Routine "trial_3" ---
            # create an object to store info about Routine trial_3
            trial_3 = data.Routine(
                name='trial_3',
                components=[image_5, image_6],
            )
            trial_3.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_5.setPos((px1, py1))
            image_6.setPos((px2, py2))
            # store start times for trial_3
            trial_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_3.tStart = globalClock.getTime(format='float')
            trial_3.status = STARTED
            thisExp.addData('trial_3.started', trial_3.tStart)
            trial_3.maxDuration = 0.3
            # keep track of which components have finished
            trial_3Components = trial_3.components
            for thisComponent in trial_3.components:
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
            
            # --- Run Routine "trial_3" ---
            # if trial has changed, end Routine now
            if isinstance(loop_7, data.TrialHandler2) and thisLoop_7.thisN != loop_7.thisTrial.thisN:
                continueRoutine = False
            trial_3.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > trial_3.maxDuration-frameTolerance:
                    trial_3.maxDurationReached = True
                    continueRoutine = False
                
                # *image_5* updates
                
                # if image_5 is starting this frame...
                if image_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_5.frameNStart = frameN  # exact frame index
                    image_5.tStart = t  # local t and not account for scr refresh
                    image_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_5.started')
                    # update status
                    image_5.status = STARTED
                    image_5.setAutoDraw(True)
                
                # if image_5 is active this frame...
                if image_5.status == STARTED:
                    # update params
                    pass
                
                # if image_5 is stopping this frame...
                if image_5.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_5.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_5.tStop = t  # not accounting for scr refresh
                        image_5.tStopRefresh = tThisFlipGlobal  # on global time
                        image_5.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_5.stopped')
                        # update status
                        image_5.status = FINISHED
                        image_5.setAutoDraw(False)
                
                # *image_6* updates
                
                # if image_6 is starting this frame...
                if image_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_6.frameNStart = frameN  # exact frame index
                    image_6.tStart = t  # local t and not account for scr refresh
                    image_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_6, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_6.started')
                    # update status
                    image_6.status = STARTED
                    image_6.setAutoDraw(True)
                
                # if image_6 is active this frame...
                if image_6.status == STARTED:
                    # update params
                    pass
                
                # if image_6 is stopping this frame...
                if image_6.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_6.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_6.tStop = t  # not accounting for scr refresh
                        image_6.tStopRefresh = tThisFlipGlobal  # on global time
                        image_6.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_6.stopped')
                        # update status
                        image_6.status = FINISHED
                        image_6.setAutoDraw(False)
                
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
                    trial_3.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_3.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_3" ---
            for thisComponent in trial_3.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_3
            trial_3.tStop = globalClock.getTime(format='float')
            trial_3.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_3.stopped', trial_3.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial_3.maxDurationReached:
                routineTimer.addTime(-trial_3.maxDuration)
            elif trial_3.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            
            # --- Prepare to start Routine "trial_4" ---
            # create an object to store info about Routine trial_4
            trial_4 = data.Routine(
                name='trial_4',
                components=[image_7, image_8],
            )
            trial_4.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_7.setPos((px3, py3))
            image_8.setPos((px4, py4))
            # store start times for trial_4
            trial_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_4.tStart = globalClock.getTime(format='float')
            trial_4.status = STARTED
            thisExp.addData('trial_4.started', trial_4.tStart)
            trial_4.maxDuration = 0.3
            # keep track of which components have finished
            trial_4Components = trial_4.components
            for thisComponent in trial_4.components:
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
            
            # --- Run Routine "trial_4" ---
            # if trial has changed, end Routine now
            if isinstance(loop_7, data.TrialHandler2) and thisLoop_7.thisN != loop_7.thisTrial.thisN:
                continueRoutine = False
            trial_4.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > trial_4.maxDuration-frameTolerance:
                    trial_4.maxDurationReached = True
                    continueRoutine = False
                
                # *image_7* updates
                
                # if image_7 is starting this frame...
                if image_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_7.frameNStart = frameN  # exact frame index
                    image_7.tStart = t  # local t and not account for scr refresh
                    image_7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_7, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_7.started')
                    # update status
                    image_7.status = STARTED
                    image_7.setAutoDraw(True)
                
                # if image_7 is active this frame...
                if image_7.status == STARTED:
                    # update params
                    pass
                
                # if image_7 is stopping this frame...
                if image_7.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_7.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_7.tStop = t  # not accounting for scr refresh
                        image_7.tStopRefresh = tThisFlipGlobal  # on global time
                        image_7.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_7.stopped')
                        # update status
                        image_7.status = FINISHED
                        image_7.setAutoDraw(False)
                
                # *image_8* updates
                
                # if image_8 is starting this frame...
                if image_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_8.frameNStart = frameN  # exact frame index
                    image_8.tStart = t  # local t and not account for scr refresh
                    image_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_8, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_8.started')
                    # update status
                    image_8.status = STARTED
                    image_8.setAutoDraw(True)
                
                # if image_8 is active this frame...
                if image_8.status == STARTED:
                    # update params
                    pass
                
                # if image_8 is stopping this frame...
                if image_8.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_8.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_8.tStop = t  # not accounting for scr refresh
                        image_8.tStopRefresh = tThisFlipGlobal  # on global time
                        image_8.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_8.stopped')
                        # update status
                        image_8.status = FINISHED
                        image_8.setAutoDraw(False)
                
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
                    trial_4.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_4.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_4" ---
            for thisComponent in trial_4.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_4
            trial_4.tStop = globalClock.getTime(format='float')
            trial_4.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_4.stopped', trial_4.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial_4.maxDurationReached:
                routineTimer.addTime(-trial_4.maxDuration)
            elif trial_4.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            thisExp.nextEntry()
            
        # completed rep7 repeats of 'loop_7'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        # set up handler to look after randomisation of conditions etc
        loop_8 = data.TrialHandler2(
            name='loop_8',
            nReps=rep8, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions('Incrementos_loop_4.xlsx'), 
            seed=None, 
        )
        thisExp.addLoop(loop_8)  # add the loop to the experiment
        thisLoop_8 = loop_8.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLoop_8.rgb)
        if thisLoop_8 != None:
            for paramName in thisLoop_8:
                globals()[paramName] = thisLoop_8[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisLoop_8 in loop_8:
            currentLoop = loop_8
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisLoop_8.rgb)
            if thisLoop_8 != None:
                for paramName in thisLoop_8:
                    globals()[paramName] = thisLoop_8[paramName]
            
            # --- Prepare to start Routine "trial_3" ---
            # create an object to store info about Routine trial_3
            trial_3 = data.Routine(
                name='trial_3',
                components=[image_5, image_6],
            )
            trial_3.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_5.setPos((px1, py1))
            image_6.setPos((px2, py2))
            # store start times for trial_3
            trial_3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_3.tStart = globalClock.getTime(format='float')
            trial_3.status = STARTED
            thisExp.addData('trial_3.started', trial_3.tStart)
            trial_3.maxDuration = 0.3
            # keep track of which components have finished
            trial_3Components = trial_3.components
            for thisComponent in trial_3.components:
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
            
            # --- Run Routine "trial_3" ---
            # if trial has changed, end Routine now
            if isinstance(loop_8, data.TrialHandler2) and thisLoop_8.thisN != loop_8.thisTrial.thisN:
                continueRoutine = False
            trial_3.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > trial_3.maxDuration-frameTolerance:
                    trial_3.maxDurationReached = True
                    continueRoutine = False
                
                # *image_5* updates
                
                # if image_5 is starting this frame...
                if image_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_5.frameNStart = frameN  # exact frame index
                    image_5.tStart = t  # local t and not account for scr refresh
                    image_5.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_5, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_5.started')
                    # update status
                    image_5.status = STARTED
                    image_5.setAutoDraw(True)
                
                # if image_5 is active this frame...
                if image_5.status == STARTED:
                    # update params
                    pass
                
                # if image_5 is stopping this frame...
                if image_5.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_5.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_5.tStop = t  # not accounting for scr refresh
                        image_5.tStopRefresh = tThisFlipGlobal  # on global time
                        image_5.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_5.stopped')
                        # update status
                        image_5.status = FINISHED
                        image_5.setAutoDraw(False)
                
                # *image_6* updates
                
                # if image_6 is starting this frame...
                if image_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_6.frameNStart = frameN  # exact frame index
                    image_6.tStart = t  # local t and not account for scr refresh
                    image_6.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_6, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_6.started')
                    # update status
                    image_6.status = STARTED
                    image_6.setAutoDraw(True)
                
                # if image_6 is active this frame...
                if image_6.status == STARTED:
                    # update params
                    pass
                
                # if image_6 is stopping this frame...
                if image_6.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_6.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_6.tStop = t  # not accounting for scr refresh
                        image_6.tStopRefresh = tThisFlipGlobal  # on global time
                        image_6.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_6.stopped')
                        # update status
                        image_6.status = FINISHED
                        image_6.setAutoDraw(False)
                
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
                    trial_3.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_3.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_3" ---
            for thisComponent in trial_3.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_3
            trial_3.tStop = globalClock.getTime(format='float')
            trial_3.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_3.stopped', trial_3.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial_3.maxDurationReached:
                routineTimer.addTime(-trial_3.maxDuration)
            elif trial_3.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            
            # --- Prepare to start Routine "trial_4" ---
            # create an object to store info about Routine trial_4
            trial_4 = data.Routine(
                name='trial_4',
                components=[image_7, image_8],
            )
            trial_4.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            image_7.setPos((px3, py3))
            image_8.setPos((px4, py4))
            # store start times for trial_4
            trial_4.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial_4.tStart = globalClock.getTime(format='float')
            trial_4.status = STARTED
            thisExp.addData('trial_4.started', trial_4.tStart)
            trial_4.maxDuration = 0.3
            # keep track of which components have finished
            trial_4Components = trial_4.components
            for thisComponent in trial_4.components:
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
            
            # --- Run Routine "trial_4" ---
            # if trial has changed, end Routine now
            if isinstance(loop_8, data.TrialHandler2) and thisLoop_8.thisN != loop_8.thisTrial.thisN:
                continueRoutine = False
            trial_4.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # is it time to end the Routine? (based on local clock)
                if tThisFlip > trial_4.maxDuration-frameTolerance:
                    trial_4.maxDurationReached = True
                    continueRoutine = False
                
                # *image_7* updates
                
                # if image_7 is starting this frame...
                if image_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_7.frameNStart = frameN  # exact frame index
                    image_7.tStart = t  # local t and not account for scr refresh
                    image_7.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_7, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_7.started')
                    # update status
                    image_7.status = STARTED
                    image_7.setAutoDraw(True)
                
                # if image_7 is active this frame...
                if image_7.status == STARTED:
                    # update params
                    pass
                
                # if image_7 is stopping this frame...
                if image_7.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_7.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_7.tStop = t  # not accounting for scr refresh
                        image_7.tStopRefresh = tThisFlipGlobal  # on global time
                        image_7.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_7.stopped')
                        # update status
                        image_7.status = FINISHED
                        image_7.setAutoDraw(False)
                
                # *image_8* updates
                
                # if image_8 is starting this frame...
                if image_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_8.frameNStart = frameN  # exact frame index
                    image_8.tStart = t  # local t and not account for scr refresh
                    image_8.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_8, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_8.started')
                    # update status
                    image_8.status = STARTED
                    image_8.setAutoDraw(True)
                
                # if image_8 is active this frame...
                if image_8.status == STARTED:
                    # update params
                    pass
                
                # if image_8 is stopping this frame...
                if image_8.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > image_8.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        image_8.tStop = t  # not accounting for scr refresh
                        image_8.tStopRefresh = tThisFlipGlobal  # on global time
                        image_8.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'image_8.stopped')
                        # update status
                        image_8.status = FINISHED
                        image_8.setAutoDraw(False)
                
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
                    trial_4.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial_4.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial_4" ---
            for thisComponent in trial_4.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial_4
            trial_4.tStop = globalClock.getTime(format='float')
            trial_4.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial_4.stopped', trial_4.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if trial_4.maxDurationReached:
                routineTimer.addTime(-trial_4.maxDuration)
            elif trial_4.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            thisExp.nextEntry()
            
        # completed rep8 repeats of 'loop_8'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        thisExp.nextEntry()
        
    # completed 8.0 repeats of 'MainLoop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
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
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
