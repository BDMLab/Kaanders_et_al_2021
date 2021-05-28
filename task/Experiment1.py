
# imports
from __future__ import division  # so that 1/3=0.333 instead of 1/3=0
from psychopy import visual, core, data, event, logging, sound, gui
from psychopy.constants import *  # things like STARTED, FINISHED
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import sin, cos, tan, log, log10, pi, average, sqrt, std, deg2rad, rad2deg, linspace, asarray, random
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
from pyglet.window import key # to detect key state, whether key is held down, to move slider on key hold
import pyglet
import pandas as pd

# Store info about the experiment session
expName = 'MetaDots'
expInfo = {'participant':'', 'session':'001'}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False: core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName

# Setup filename for saving
filename = 'data/%s_%s_%s' %(expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath=None,
    savePickle=True, saveWideText=True,
    dataFileName=filename)
#save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp

# Start Code - component code to be run before the window creation

# set up variable to track current state of key press, to move slider when keys held down

keyState = key.KeyStateHandler()

# Setup the Window
win = visual.Window(size=(1280, 800), fullscr=True, screen=0, allowGUI=True, allowStencil=False,
    monitor=u'testMonitor', color=u'black', colorSpace=u'rgb',
    blendMode=u'avg', useFBO=True
    )
win.winHandle.push_handlers(keyState)

# store frame rate of monitor if we can measure it successfully
expInfo['frameRate']=win.getActualFrameRate()
if expInfo['frameRate']!=None:
    frameDur = 1.0/round(expInfo['frameRate'])
else:
    frameDur = 1.0/60.0 # couldn't get a reliable measure so guess
  

# Set up variables for random_dots
DotStable = 50
DotDifference = 20
DotVariable = DotStable - DotDifference

DotLeft = 1
DotRight = 1

DotLeftList=[40, 50, 53, 35, 23]
DotRightList=[60, 30, 58, 25, 8]

######################## COMPONENTS ##########################

#Instalize components for Routine Introduction
IntroductionClock =core.Clock()

IntroductionText =visual.TextStim(win,text='Welcome to this experiment. \n\n\nPress space to find out what the task involves!',
    units='cm', pos=(0,2.0),  height=1, wrapWidth=30)

#Instalize components for Routine "Instructions 1"
Instructions1Clock =core.Clock()

Instructions1Text =visual.TextStim(win, text='You will see two circles on the screen, each with a number of dots inside. Your task is to try to guess which circle contains the most dots. \n\n\nPlease press space to see some example stimuli.',
    units='cm', pos=(0,2.0), height=1, wrapWidth=30)

# Initialize components for ITI
ITIClock =core.Clock()

FixationCrossLeft =visual.TextStim(win,text="+",
    units='cm', pos=(-11, 4.0), height=3, wrapWidth=15)

FixationCrossRight =visual.TextStim(win,text="+",
    units='cm', pos=(11, 4.0), height=3, wrapWidth=15)

DotLeftText =visual.TextStim(win,text=u"0",
    units='cm', pos=(-11,4.0),  height=1, wrapWidth=30)

DotRightText =visual.TextStim(win,text=u"0",
    units='cm', pos=(11,4.0),  height=1, wrapWidth=30)

DotDifText =visual.TextStim(win,text=u"0",
    units='cm', pos=(0,0),  height=1, wrapWidth=30)

# Initialize components for Main Experiment
ChoiceClock =core.Clock()

CircleLeft = visual.Circle(win, radius=6.0, edges=1000,
    lineColor=(1.0, 1.0, 1.0), fillColorSpace=u'rgb', fillColor=u'black', lineWidth=3,
    pos=(-11.0, 4.0), interpolate=False, units='cm')

CircleRight = visual.Circle(win, radius=6.0, edges=1000,
    lineColor=(1.0, 1.0, 1.0), fillColorSpace=u'rgb', fillColor=u'black', lineWidth=3,
    pos=(11.0, 4.0), interpolate=False, units='cm')

DotPatchLeft =visual.DotStim(win, color=(1.0,1.0,1.0), dir=270, units='cm',
    nDots=int(DotLeft), fieldShape='circle', fieldPos=(-11.0, 4.0),fieldSize=11.0,
    dotSize= 15,
    dotLife=-1, #number of frames for each dot to be drawn
    signalDots='same', #are the signal dots the 'same' on each frame? (see Scase et al)
    noiseDots='position', #do the noise dots follow random- 'walk', 'direction', or 'position'
    speed=0.0, coherence=1.0)

DotPatchRight =visual.DotStim(win, color=(1.0,1.0,1.0), dir=270, units='cm',
    nDots=int(DotLeft), fieldShape='circle', fieldPos=(11.0, 4.0),fieldSize=11.0,
    dotSize= 15,
    dotLife=-1, #number of frames for each dot to be drawn
    signalDots='same', #are the signal dots the 'same' on each frame? (see Scase et al)
    noiseDots='position', #do the noise dots follow random- 'walk', 'direction', or 'position'
    speed=0.0, coherence=1.0)

DotExampleLeft =visual.TextStim(win,text=sin, units='cm',
    pos=(-11.0,-3.0), height=1, wrapWidth=10)
    
DotExampleRight =visual.TextStim(win,text=sin, units='cm',
    pos=(11.0,-3.0), height=1, wrapWidth=10)

DotExampleInstructions =visual.TextStim(win,text="Please press space to continue",
    units='cm', pos=(0,-8.0), height=1, wrapWidth=15)
    
DotCalibrationInstructions = visual.TextStim(win,text="Which circle contains the most dots?",
    units='cm', pos=(0,-8.0), height=1, wrapWidth=20)
    
#Initialize components for Routine Instructions2
Instructions2Clock =core.Clock()

Instructions2Text =visual.TextStim(win,text=u"The first part of the task is to choose which circle contains the most dots and then rate your confidence in this decision. We will next familiarise you with this part of the task. Press the left arrow key \u2190 to pick the left circle and the right arrow key \u2192 to pick the right circle. After each trial you will be asked to rate your confidence in your decision by using the arrow keys (left \u2190 and right \u2192) to move a slider on a scale. There will be a green tick under the circle that you chose and a red cross under the circle you did not choose . Don't worry if some of your decisions feel like guesses - it is a hard task! \n\n\nPress the space bar to continue.",
    units='cm', pos=(0,2.0),  height=1, wrapWidth=30)

# Initialize components for Confidence trials
ConfidenceClock = core.Clock()
InstrConf = visual.TextStim(win,text="How confident are you about your choice?",
    units='cm', pos=(0,-8.0), height=0.5, wrapWidth=20)
InstrConfKeys = visual.TextStim(win,text="Use the left and right arrow keys to move the slider on the scale.",
    units='cm', pos=(0,-9.0), height=0.5, wrapWidth=20)
Confidence1 = visual.RatingScale(win=win, name='Confidence', marker=u'triangle', markerColor=u'orange', leftKeys='left', rightKeys='right', size=0.6,
    pos=[0.0, -0.5], low=0, high=1, precision=100, labels=[u'Certainly wrong', u'Certainly correct'], scale=u'',
    markerStart = u'0.5', tickHeight=u'0', showAccept=False, acceptKeys=[u'down'])

FeedbackPlusLeft = visual.ImageStim(win=win, image='plus.png', ori=0, name='FeedbackPlusLeft',
units='cm', pos=(-11.0,-4.5))
    
FeedbackPlusRight = visual.ImageStim(win=win, image='plus.png', ori=0, name='FeedbackPlusRight',
units='cm', pos=(11.0,-4.5))

FeedbackCrossLeft = visual.ImageStim(win=win, image='cross.png', ori=0, name='FeedbackCrossLeft',
units='cm', pos=(-11.0,-4.5))

FeedbackCrossRight = visual.ImageStim(win=win, image='cross.png', ori=0, name='FeedbackCrossRight',
units='cm', pos=(11.0,-4.5))

#Initialize components for Routine Instructions3
Instructions3Clock =core.Clock()

Instructions3Text =visual.TextStim(win,text=u"In the main task, you will have to make the same decisions about circles with dots in them. However, now after each decision you get additional viewing time of the dots. By pressing the left and right arrow keys you can switch between seeing the dots in either of the two circles. You can see which circle you chose by the green tick at the bottom of the circle and by the red cross below the circle you did not choose. \n\n\nPlease press the space bar to continue.",
    units='cm', pos=(0,2.0),  height=1, wrapWidth=30)

# Initialize components for Routine Instructions3a
Instructions3aClock = core.Clock()

Instructions3aText =visual.TextStim(win,text=u"You will now have some trials to practice this. You can switch between the two circles as often as you want within the allocated time. When the viewing time is up, you are given the opportunity to revise your decision. Use the left and right arrow keys to pick one of the circles again. You are asked to rate your confidence in your final decision as well. \n\nIn these example trials you will also get feedback about the amount of points you would have earned if these were part of the main task. You can earn a maximum of 200 points in each trial. In the main task, the number of points you earn will not be presented on the screen after each trial.\n\n\nPlease press the space bar to continue.",
    units='cm', pos=(0,2.0),  height=1, wrapWidth=30) 

# Initialize components for Routine Feedback
FeedbackClock=core.Clock()
FeedbackText = visual.TextStim(win,text=u"You have earned 0 example points in this trial. \n\n\nPlease press the space bar to continue.",
    units='cm', pos=(0,2.0),  height=1, wrapWidth=30)

# Initialize components for Routine "Confidence"
ConfClock = core.Clock()
#Confidence2 = visual.RatingScale(win=win, name='Confidence', marker=u'triangle', markerColor=u'orange', leftKeys=None, rightKeys=None, size=1.0, pos=[0.0, -0.3], low=0, high=1, precision=100, labels=[u'Less', u'More'], scale=u'', markerStart=u'0.5', tickHeight=u'0', showAccept=False, acceptKeys=[u'down', u'return'])

# Initialize components for Routine "Phase II"
CipClock = core.Clock()
Instr1Cip = visual.TextStim(win,text="Use the left and right arrow keys to see the dots in the circles again.",
    units='cm', pos=(0,-8.0), height=0.5, wrapWidth=20)
Instr2Cip = visual.TextStim(win,text="You can switch between the two circles as often as you want within the allocated time.",
    units='cm', pos=(0,-9.0), height=0.5, wrapWidth=20)

#Initialize components for routine "Choice2"
Choice2Clock = core.Clock()
DotCalibrationInstructions2 = visual.TextStim(win,text="Now choose again.",
    units='cm', pos=(0,-8.0), height=1, wrapWidth=20)

# Initialize components for routine "Confidence2"
Conf2Clock = core.Clock()
Instr1Conf2 = visual.TextStim(win,text="How confident are you about your choice now?",
    units='cm', pos=(0,-8.0), height=0.5, wrapWidth=20)
Example_Trial = -1

#Instalize components for Routine Instructions4
Instructions4Clock =core.Clock()

Instructions4Text =visual.TextStim(win,text=u"The training is now complete and you will begin the main task that will work almost the same as the last practice trials. Now you can earn money for each answer and confidence rating: you will receive \u00A31 per 12000 points earned in addition to the \u00A310 show-up fee! Please review the paragraph on how your earnings are calculated in the provided paper instructions. You will get a chance to rest every so often, please use these breaks as it is a challenging task. \n\nIf you have any questions please ask the experimenter now.\n\n\nOtherwise, press the space bar to continue.",
    units='cm', pos=(0,2.0),  height=1, wrapWidth=30)

Trial = -1

# Instalize components for "Rest"
RestClock = core.Clock()
RestText = visual.TextStim(win=win, ori=0, name='rest_prompt_txt', text=u'Great! \nYou have earned \u00A30 so far! Now take a rest and press spacebar when you are ready to begin the next block.', font=u'Arial', pos=[0, 0], height=0.08, wrapWidth=None, color=u'white', colorSpace=u'rgb', opacity=1, depth=0.0)

# Initialize components for "Break"
BreakClock = core.Clock()
BreakText = visual.TextStim(win=win, ori=0, name='break_txt', text=u'Great! \nYou have earned \u00A30 so far! You are halfway through this task. \n\nPlease contact the experimenter.', font=u'Arial', pos=[0, 0], height=0.08, wrapWidth=None, color=u'white', colorSpace=u'rgb', opacity=1, depth=0.0)

# Instalize components for "Thank You"
ThankYouClock = core.Clock()
ThankYouText = visual.TextStim(win=win, ori=0, name='rest_prompt_txt', text=u'You have now completed this experiment. \nYou have earned \u00A30 in addition to your show-up fee! Thank you for your participation. Please inform the experimenter that you have finished.', font=u'Arial', pos=[0, 0], height=0.08, wrapWidth=None, color=u'white', colorSpace=u'rgb', opacity=1, depth=0.0)

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# add up correct answers
CorrAnswers = 0
example_points = 0
points = 0

####################### Introduction to experiment ###########################

#------Prepare to start Routine "Introduction"-------
t = 0
IntroductionClock.reset()  # clock 
frameN = -1
# update component parameters for each repeat
IntroductionResponse = event.BuilderKeyResponse()  # create an object of type KeyResponse
IntroductionResponse.status = NOT_STARTED
# keep track of which components have finished
IntroductionComponents = []
IntroductionComponents.append(IntroductionText)
IntroductionComponents.append(IntroductionResponse)
for thisComponent in IntroductionComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "Introduction"-------
continueRoutine = True
while continueRoutine:
    # get current time
    t = IntroductionClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *IntroductionText* updates
    if t >= 0.0 and IntroductionText.status == NOT_STARTED:
        # keep track of start time/frame for later
        IntroductionText.tStart = t  # underestimates by a little under one frame
        IntroductionText.frameNStart = frameN  # exact frame index
        IntroductionText.setAutoDraw(True)
    
    # *IntroductionResponse* updates
    if t >= 0 and IntroductionResponse.status == NOT_STARTED:
        # keep track of start time/frame for later
        IntroductionResponse.tStart = t  # underestimates by a little under one frame
        IntroductionResponse.frameNStart = frameN  # exact frame index
        IntroductionResponse.status = STARTED
        # keyboard checking is just starting
        IntroductionResponse.clock.reset()  # now t=0
        event.clearEvents(eventType='keyboard')
    if IntroductionResponse.status == STARTED:
        theseKeys = event.getKeys(keyList=['space'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            IntroductionResponse.keys = theseKeys[-1]  # just the last key pressed
            IntroductionResponse.rt = IntroductionResponse.clock.getTime()
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in IntroductionComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()
    else:  # this Routine was not non-slip safe so reset non-slip timer
        routineTimer.reset()

#-------Ending Routine "Introduction"-------
for thisComponent in IntroductionComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if IntroductionResponse.keys in ['', [], None]:  # No response was made
   IntroductionResponse.keys=None
# Move on to next section
thisExp.nextEntry()

#------Prepare to start Routine "Instructions1"-------
t = 0
Instructions1Clock.reset()  # clock 
frameN = -1
# update component parameters for each repeat
Instructions1Response = event.BuilderKeyResponse()  # create an object of type KeyResponse
Instructions1Response.status = NOT_STARTED
# keep track of which components have finished
Instructions1Components = []
Instructions1Components.append(Instructions1Text)
Instructions1Components.append(Instructions1Response)
for thisComponent in Instructions1Components:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "Instructions1"-------
continueRoutine = True
while continueRoutine:
    # get current time
    t = Instructions1Clock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *Instructions1Text* updates
    if t >= 0.0 and Instructions1Text.status == NOT_STARTED:
        # keep track of start time/frame for later
        Instructions1Text.tStart = t  # underestimates by a little under one frame
        Instructions1Text.frameNStart = frameN  # exact frame index
        Instructions1Text.setAutoDraw(True)
    
    # *Instructions1Response* updates
    if t >= 0 and Instructions1Response.status == NOT_STARTED:
        # keep track of start time/frame for later
        Instructions1Response.tStart = t  # underestimates by a little under one frame
        Instructions1Response.frameNStart = frameN  # exact frame index
        Instructions1Response.status = STARTED
        # keyboard checking is just starting
        Instructions1Response.clock.reset()  # now t=0
        event.clearEvents(eventType='keyboard')
    if Instructions1Response.status == STARTED:
        theseKeys = event.getKeys(keyList=['space'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            Instructions1Response.keys = theseKeys[-1]  # just the last key pressed
            Instructions1Response.rt = Instructions1Response.clock.getTime()
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in Instructions1Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()
    else:  # this Routine was not non-slip safe so reset non-slip timer
        routineTimer.reset()

#-------Ending Routine "Instructions1"-------
for thisComponent in Instructions1Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if Instructions1Response.keys in ['', [], None]:  # No response was made
   Instructions1Response.keys=None
# store data for thisExp (ExperimentHandler)
thisExp.addData('Instructions1Response.keys',Instructions1Response.keys)
if Instructions1Response.keys != None:  # we had a response
    thisExp.addData('Instructions1Response.rt', Instructions1Response.rt)
thisExp.nextEntry()

############################# EXAMPLE LOOP ###############################

# set up handler to look after randomisation of conditions etc
choice_loop = data.TrialHandler(nReps=1, method=u'sequential', extraInfo=expInfo, originPath=None,
    trialList=data.importConditions(u'ExampleTrials.xlsx'),
    seed=None, name='choice_loop')
thisExp.addLoop(choice_loop)
thischoice_loop = choice_loop.trialList[0]
if thischoice_loop != None:
    for paramName in thischoice_loop.keys():
        exec paramName + '= thischoice_loop.' + paramName

# Set up counter to keep track of binary loop cycle, so that the rest prompt is shown only after a certain number of trials
choice_loop_counter = 0

for thischoice_loop in choice_loop:
    currentLoop = choice_loop
    if thischoice_loop != None:
        for paramName in thischoice_loop.keys():
            exec paramName + '= thischoice_loop.' + paramName
            
    # Increase loop counter by one
    DotLeft=int(DotLeftList[choice_loop_counter])
    DotRight=int(DotRightList[choice_loop_counter])
    choice_loop_counter += 1

    #------Prepare to start Routine "choice"-------
    t = 0
    ChoiceClock.reset()
    frameN = -1
    
     # update component parameters for each repeat
    DotPatchLeft=visual.DotStim(win, color=(1.0,1.0,1.0), dir=270, units='cm',
    nDots=int(DotLeft), fieldShape='circle', fieldPos=(-11.0, 4.0), fieldSize=11,
    dotSize= 15,
    dotLife=-1, #number of frames for each dot to be drawn
    signalDots='same', #are the signal dots the 'same' on each frame? (see Scase et al)
    noiseDots='position', #do the noise dots follow random- 'walk', 'direction', or 'position'
    speed=0.0, coherence=1.0)
    
    DotPatchRight=visual.DotStim(win, color=(1.0,1.0,1.0), dir=270, units='cm',
    nDots=int(DotRight), fieldShape='circle', fieldPos=(11, 4.0),fieldSize=11,
    dotSize= 15,
    dotLife=-1, #number of frames for each dot to be drawn
    signalDots='same', #are the signal dots the 'same' on each frame? (see Scase et al)
    noiseDots='position', #do the noise dots follow random- 'walk', 'direction', or 'position'
    speed=0.0, coherence=1.0)

    DotExampleLeft.setText(ExampleNrLeft)
    DotExampleRight.setText(ExampleNrRight)
    
    event.clearEvents(eventType='keyboard')
    key_resp_choice = event.BuilderKeyResponse()
    key_resp_choice.status = NOT_STARTED
     # keep track of which components have finished
    dot_choiceComponents = []
    dot_choiceComponents.append(CircleLeft)
    dot_choiceComponents.append(CircleRight)
    dot_choiceComponents.append(DotPatchLeft)
    dot_choiceComponents.append(DotPatchRight)
    dot_choiceComponents.append(DotExampleInstructions)
    dot_choiceComponents.append(DotExampleLeft)
    dot_choiceComponents.append(DotExampleRight)
    for thisComponent in dot_choiceComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
 
 #-------Start Routine "dot_choice"-------
    continueRoutine = True
    while continueRoutine:
        t = ChoiceClock.getTime()
        frameN = frameN + 1
        
        # *CircleLeft* updates
        if t >= 0.2 and CircleLeft.status == NOT_STARTED:
            CircleLeft.tStart = t
            CircleLeft.frameNStart = frameN
            CircleLeft.setAutoDraw(True)

            
        # *CircleRight* updates
        if t >= 0.2 and CircleRight.status == NOT_STARTED:
            CircleRight.tStart = t
            CircleRight.frameNStart = frameN
            CircleRight.setAutoDraw(True)

            
         # *DotPatchLeft* updates
        if t >= 0.2 and DotPatchLeft.status == NOT_STARTED:
            DotPatchLeft.tStart = t
            DotPatchLeft.frameNStart = frameN
            DotPatchLeft.setAutoDraw(True)


        # *DotPatchRight* updates
        if t >= 0.2 and DotPatchRight.status == NOT_STARTED:
            DotPatchRight.tStart = t
            DotPatchRight.frameNStart = frameN
            DotPatchRight.setAutoDraw(True)


        # *DotExampleInstructions* updates
        if t >= 0.2 and DotExampleInstructions.status == NOT_STARTED:
            DotExampleInstructions.tStart = t
            DotExampleInstructions.frameNStart = frameN
            DotExampleInstructions.setAutoDraw(True)
            
        # *DotExampleLeft* updates
        if t >= 0.2 and DotExampleLeft.status == NOT_STARTED:
            DotExampleLeft.tStart = t
            DotExampleLeft.frameNStart = frameN
            DotExampleLeft.setAutoDraw(True)
            
        # *DotExampleRight* updates
        if t >= 0.2 and DotExampleRight.status == NOT_STARTED:
            DotExampleRight.tStart = t
            DotExampleRight.frameNStart = frameN
            DotExampleRight.setAutoDraw(True)
            
        # *key_resp_choice* updates
        if t >= 0.2 and key_resp_choice.status == NOT_STARTED:
            key_resp_choice.tStart = t
            key_resp_choice.frameNStart = frameN
            key_resp_choice.status = STARTED
            key_resp_choice.clock.reset()
        if key_resp_choice.status == STARTED:
            theseKeys = event.getKeys(keyList=['space'])
            
            # check for quit:
            if 'escape' in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:
                key_resp_choice.keys = theseKeys[-1]
                key_resp_choice.rt = key_resp_choice.clock.getTime()
                continueRoutine = False
        
        # check if all components have finished
        if not continueRoutine:
            routineTimer.reset()
            break
        continueRoutine = False
        for thisComponent in dot_choiceComponents:
            if hasattr(thisComponent, 'status') and thisComponent.status != FINISHED:
                continueRoutine = True
                break
            
         # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=['escape']):
            core.quit()
            
        # refresh the screen
        if continueRoutine:
            win.flip()
        else:
            routineTimer.reset()
    
     #-------Ending Routine "choice"-------
    for thisComponent in dot_choiceComponents:
        if hasattr(thisComponent, 'setAutoDraw'):
            thisComponent.setAutoDraw(False)
    
    # check responses
    if key_resp_choice.keys in ['', [], None]:
        key_resp_choice.keys = None
    
    # store data for binary (TrialHandler)
    choice_loop.addData('key_resp_choice.keys', key_resp_choice.keys)
    if key_resp_choice.keys != None:
        choice_loop.addData('key_resp_choice.rt', key_resp_choice.rt)
    choice_loop.addData('DotPatchRight.nDots', DotPatchRight.nDots)
    choice_loop.addData('DotPatchLeft.nDots', DotPatchLeft.nDots)
    thisExp.nextEntry()
    
########################################### Examples Finished ###################################################

#------Prepare to start Routine "Instructions2"-------
t = 0
Instructions2Clock.reset()  # clock 
frameN = -1
# update component parameters for each repeat
Instructions2Response = event.BuilderKeyResponse()  # create an object of type KeyResponse
Instructions2Response.status = NOT_STARTED
# keep track of which components have finished
Instructions2Components = []
Instructions2Components.append(Instructions2Text)
Instructions2Components.append(Instructions2Response)
for thisComponent in Instructions2Components:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "Instructions2"-------
continueRoutine = True
while continueRoutine:
    # get current time
    t = Instructions2Clock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *Instructions2Text* updates
    if t >= 0.0 and Instructions2Text.status == NOT_STARTED:
        # keep track of start time/frame for later
        Instructions2Text.tStart = t  # underestimates by a little under one frame
        Instructions2Text.frameNStart = frameN  # exact frame index
        Instructions2Text.setAutoDraw(True)
    
    # *Instructions2Response* updates
    if t >= 0 and Instructions2Response.status == NOT_STARTED:
        # keep track of start time/frame for later
        Instructions2Response.tStart = t  # underestimates by a little under one frame
        Instructions2Response.frameNStart = frameN  # exact frame index
        Instructions2Response.status = STARTED
        # keyboard checking is just starting
        Instructions2Response.clock.reset()  # now t=0
        event.clearEvents(eventType='keyboard')
    if Instructions2Response.status == STARTED:
        theseKeys = event.getKeys(keyList=['space'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            Instructions2Response.keys = theseKeys[-1]  # just the last key pressed
            Instructions2Response.rt = Instructions2Response.clock.getTime()
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in Instructions2Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()
    else:  # this Routine was not non-slip safe so reset non-slip timer
        routineTimer.reset()

#-------Ending Routine "Instructions2"-------
for thisComponent in Instructions2Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if Instructions2Response.keys in ['', [], None]:  # No response was made
   Instructions2Response.keys=None
# store data for thisExp (ExperimentHandler)
thisExp.addData('Instructions2Response.keys',Instructions2Response.keys)
if Instructions2Response.keys != None:  # we had a response
    thisExp.addData('Instructions2Response.rt', Instructions2Response.rt)
thisExp.nextEntry()

############################# Calibration Phase ############################

#create the staircase handler
CalStaircase = data.StairHandler(startVal = 20.0,
                          stepType = 'lin', stepSizes=[4,4,3,3,2,2,1,1],
                          nUp=1, nDown=2, minVal=1, maxVal=49)  #will home in on the 70% threshold
                          
for ThisIncrement in CalStaircase:
    
    # Set dot difference for this trial
    DotDifference = ThisIncrement
    #randomise whether the stable dotarray is the smaller or the larger value
    DotDiffDenominator = np.random.choice([-1,1])
    if DotDiffDenominator == -1:
        DotVariable = DotStable - DotDifference
    else:
        DotVariable = DotStable + DotDifference
    #randomise the location of the fixed and variable stimuli
    StableSide= np.random.choice([-1,1])
    if StableSide == -1:
        DotLeft = DotStable
        DotRight = DotVariable
    else:
        DotLeft =DotVariable
        DotRight = DotStable
    # Determine the correct response for the trial
    if DotLeft > DotRight:
        CorrectKey = str('left')
    elif DotRight > DotLeft:
        CorrectKey = str('right')

    #------Prepare to start Routine "ITI"-------
    t = 0
    ITIClock.reset()
    frameN = -1
    routineTimer.add(1.00000)
     # update component parameters for each repeat
    # Update Keyboard parameters
    event.clearEvents(eventType='keyboard')
    KeyRespCal = event.BuilderKeyResponse()
    KeyRespCal.status = NOT_STARTED
     # keep track of which components have finished
    dot_ITIComponents = []
    dot_ITIComponents.append(CircleLeft)
    dot_ITIComponents.append(CircleRight)
    dot_ITIComponents.append(FixationCrossLeft)
    dot_ITIComponents.append(FixationCrossRight)
    for thisComponent in dot_ITIComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

#-------Start Routine "ITI"-------
    continueRoutine = True
    while continueRoutine and routineTimer.getTime() > 0:
        t = ITIClock.getTime()
        frameN = frameN + 1

        # *CircleLeft* updates
        if t >= 0 and CircleLeft.status == NOT_STARTED:
            CircleLeft.tStart = t
            CircleLeft.frameNStart = frameN
            CircleLeft.setAutoDraw(True)


        # *CircleRight* updates
        if t >= 0 and CircleRight.status == NOT_STARTED:
            CircleRight.tStart = t
            CircleRight.frameNStart = frameN
            CircleRight.setAutoDraw(True)


         # *FixationCrossLeft* updates
        if t >= 0 and FixationCrossLeft.status == NOT_STARTED:
            FixationCrossLeft.tStart = t
            FixationCrossLeft.frameNStart = frameN
            FixationCrossLeft.setAutoDraw(True)


        # *FixationCrossRight* updates
        if t >= 0 and FixationCrossRight.status == NOT_STARTED:
            FixationCrossRight.tStart = t
            FixationCrossRight.frameNStart = frameN
            FixationCrossRight.setAutoDraw(True)


            # check for quit:
            if 'escape' in theseKeys:
                endExpNow = True

        # check if all components have finished
        if not continueRoutine:
            routineTimer.reset()
            break
        continueRoutine = False
        for thisComponent in dot_ITIComponents:
            if hasattr(thisComponent, 'status') and thisComponent.status != FINISHED:
                continueRoutine = True
                break

         # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=['escape']):
            core.quit()

        # refresh the screen
        if continueRoutine:
            win.flip()
        else:
            routineTimer.reset()

     #-------Ending Routine "ITI"-------
    for thisComponent in dot_ITIComponents:
        if hasattr(thisComponent, 'setAutoDraw'):
            thisComponent.setAutoDraw(False)

    #------Prepare to start Routine "choice"-------
    t = 0
    ChoiceClock.reset()
    frameN = -1

     # update component parameters for each repeat
    DotPatchLeft=visual.DotStim(win, color=(1.0,1.0,1.0), dir=270, units='cm',
    nDots=int(DotLeft), fieldShape='circle', fieldPos=(-11.0, 4.0), fieldSize=11,
    dotSize= 15,
    dotLife=-1, #number of frames for each dot to be drawn
    signalDots='same', #are the signal dots the 'same' on each frame? (see Scase et al)
    noiseDots='position', #do the noise dots follow random- 'walk', 'direction', or 'position'
    speed=0.0, coherence=1.0)
    
    DotPatchRight=visual.DotStim(win, color=(1.0,1.0,1.0), dir=270, units='cm',
    nDots=int(DotRight), fieldShape='circle', fieldPos=(11, 4.0),fieldSize=11,
    dotSize= 15,
    dotLife=-1, #number of frames for each dot to be drawn
    signalDots='same', #are the signal dots the 'same' on each frame? (see Scase et al)
    noiseDots='position', #do the noise dots follow random- 'walk', 'direction', or 'position'
    speed=0.0, coherence=1.0)
    
    DotLeftText.setText(DotLeft)
    DotRightText.setText(DotRight)
    DotDifText.setText(DotDifference)

    # Update Keyboard parameters
    event.clearEvents(eventType='keyboard')
    KeyRespCal = event.BuilderKeyResponse()
    KeyRespCal.status = NOT_STARTED
     # keep track of which components have finished
    dot_choiceComponents = []
    dot_choiceComponents.append(CircleLeft)
    dot_choiceComponents.append(CircleRight)
    dot_choiceComponents.append(DotPatchLeft)
    dot_choiceComponents.append(DotPatchRight)
    dot_choiceComponents.append(FixationCrossLeft)
    dot_choiceComponents.append(FixationCrossRight)
    dot_choiceComponents.append(DotCalibrationInstructions)
    for thisComponent in dot_choiceComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
 
#-------Start Routine "dot_choice"-------
    continueRoutine = True
    while continueRoutine:
        t = ChoiceClock.getTime()
        frameN = frameN + 1
        
        # *CircleLeft* updates
        if t >= 0 and CircleLeft.status == NOT_STARTED:
            CircleLeft.tStart = t
            CircleLeft.frameNStart = frameN
            CircleLeft.setAutoDraw(True)

            
        # *CircleRight* updates
        if t >= 0 and CircleRight.status == NOT_STARTED:
            CircleRight.tStart = t
            CircleRight.frameNStart = frameN
            CircleRight.setAutoDraw(True)

            
        # *DotPatchLeft* updates
        if t >= 0 and DotPatchLeft.status == NOT_STARTED:
            DotPatchLeft.tStart = t
            DotPatchLeft.frameNStart = frameN
            DotPatchLeft.setAutoDraw(True)
        if DotPatchLeft.status == STARTED and t >= (0 + (0.7-win.monitorFramePeriod*0.75)): #most of one frame period left
            DotPatchLeft.setAutoDraw(False)


        # *DotPatchRight* updates
        if t >= 0 and DotPatchRight.status == NOT_STARTED:
            DotPatchRight.tStart = t
            DotPatchRight.frameNStart = frameN
            DotPatchRight.setAutoDraw(True)
        if DotPatchRight.status == STARTED and t >= (0 + (0.7-win.monitorFramePeriod*0.75)): #most of one frame period left
            DotPatchRight.setAutoDraw(False)

        # *FixationCrossLeft* updates
        if t >= 0.7 and FixationCrossLeft.status == NOT_STARTED:
            FixationCrossLeft.tStart = t
            FixationCrossLeft.frameNStart = frameN
            FixationCrossLeft.setAutoDraw(True)

        # *FixationCrossRight* updates
        if t >= 0.7 and FixationCrossRight.status == NOT_STARTED:
            FixationCrossRight.tStart = t
            FixationCrossRight.frameNStart = frameN
            FixationCrossRight.setAutoDraw(True)

        # *DotExampleInstructions* updates
        if t >= 0 and DotCalibrationInstructions.status == NOT_STARTED:
            DotCalibrationInstructions.tStart = t
            DotCalibrationInstructions.frameNStart = frameN
            DotCalibrationInstructions.setAutoDraw(True)
            
        # *key_resp_choice* updates
        if t >= 0.7 and KeyRespCal.status == NOT_STARTED:
            KeyRespCal.tStart = t
            KeyRespCal.frameNStart = frameN
            KeyRespCal.status = STARTED
            KeyRespCal.clock.reset()
        if KeyRespCal.status == STARTED:
            theseKeys = event.getKeys(keyList=['left', 'right'])
            
            # check for quit:
            if 'escape' in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:
                KeyRespCal.keys = theseKeys[-1]
                KeyRespCal.rt = KeyRespCal.clock.getTime()
                continueRoutine = False
        
        # check if all components have finished
        if not continueRoutine:
            routineTimer.reset()
            break
        continueRoutine = False
        for thisComponent in dot_choiceComponents:
            if hasattr(thisComponent, 'status') and thisComponent.status != FINISHED:
                continueRoutine = True
                break
            
         # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=['escape']):
            core.quit()
            
        # refresh the screen
        if continueRoutine:
            win.flip()
        else:
            routineTimer.reset()
    
     #-------Ending Routine "choice"-------
    for thisComponent in dot_choiceComponents:
        if hasattr(thisComponent, 'setAutoDraw'):
            thisComponent.setAutoDraw(False)
    # check responses
    if KeyRespCal.keys in ['', [], None]:
        KeyRespCal.keys = None
        # was no response the correct answer?!
    if KeyRespCal.keys == CorrectKey: KeyRespCal.corr = 1  # correct non-response
    else: KeyRespCal.corr = 0  # failed to respond (incorrectly)
    
    # Tell the staircase if the participant is correct (TrialHandler)
    CalStaircase.addResponse(KeyRespCal.corr)
    
    # Store data for experiment:
    thisExp.addData('CalCorrect', KeyRespCal.corr)
    thisExp.addData('CalCorrectKey', CorrectKey)
    thisExp.addData('CalResponse', KeyRespCal.keys)
    if KeyRespCal.keys != None:
        thisExp.addData('CalRT', KeyRespCal.rt)
    thisExp.addData('CalDotDifference', DotDifference)
    thisExp.addData('CalDotNumberRight', DotPatchRight.nDots)
    thisExp.addData('CalDotNumberLeft', DotPatchLeft.nDots)


#------Prepare to start Routine "Confidence"-------
    t = 0
    ConfidenceClock.reset()
    frameN = -1
    Confidence1.reset()

     # keep track of which components have finished
    FeedbackComponents = []
    FeedbackComponents.append(CircleLeft)
    FeedbackComponents.append(CircleRight)
    FeedbackComponents.append(FixationCrossLeft)
    FeedbackComponents.append(FixationCrossRight)
    FeedbackComponents.append(FeedbackPlusLeft)
    FeedbackComponents.append(FeedbackPlusRight)
    FeedbackComponents.append(FeedbackCrossLeft)
    FeedbackComponents.append(FeedbackCrossRight)
    FeedbackComponents.append(Confidence1)
    for thisComponent in FeedbackComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
 
#-------Start Routine "Confidence"-------
    continueRoutine = True
    while continueRoutine:
        t = ConfidenceClock.getTime()
        frameN = frameN + 1
        
        # *CircleLeft* updates
        if t >= 0.0 and CircleLeft.status == NOT_STARTED:
            CircleLeft.tStart = t
            CircleLeft.frameNStart = frameN
            CircleLeft.setAutoDraw(True)
            
        # *CircleRight* updates
        if t >= 0.0 and CircleRight.status == NOT_STARTED:
            CircleRight.tStart = t
            CircleRight.frameNStart = frameN
            CircleRight.setAutoDraw(True)

        # *FixationCrossLeft* updates
        if t >= 0.0 and FixationCrossLeft.status == NOT_STARTED:
            FixationCrossLeft.tStart = t
            FixationCrossLeft.frameNStart = frameN
            FixationCrossLeft.setAutoDraw(True)

        # *FixationCrossRight* updates
        if t >= 0.0 and FixationCrossRight.status == NOT_STARTED:
            FixationCrossRight.tStart = t
            FixationCrossRight.frameNStart = frameN
            FixationCrossRight.setAutoDraw(True)

        # *FeedbackPlusLeft* updates
        if KeyRespCal.keys == 'left':
            if t >= 0.0 and FeedbackPlusLeft.status == NOT_STARTED:
                FeedbackPlusLeft.tStart = t
                FeedbackPlusLeft.frameNStart = frameN
                FeedbackPlusLeft.setAutoDraw(True)
            
        # *FeedbackPlusRight* updates
        if KeyRespCal.keys == 'right':
            if t >= 0.0 and FeedbackPlusRight.status == NOT_STARTED:
                FeedbackPlusRight.tStart = t
                FeedbackPlusRight.frameNStart = frameN
                FeedbackPlusRight.setAutoDraw(True)

        # *FeedbackCrossLeft* updates
        if KeyRespCal.keys == 'right':
            if t >= 0.0 and FeedbackCrossLeft.status == NOT_STARTED:
                FeedbackCrossLeft.tStart = t
                FeedbackCrossLeft.frameNStart = frameN
                FeedbackCrossLeft.setAutoDraw(True)

        # *FeedbackCrossRight* updates
        if KeyRespCal.keys == 'left':
            if t >= 0.0 and FeedbackCrossRight.status == NOT_STARTED:
                FeedbackCrossRight.tStart = t
                FeedbackCrossRight.frameNStart = frameN
                FeedbackCrossRight.setAutoDraw(True)

        # *MainRating* updates
        if t > 0.2:
            continueRoutine = Confidence1.noResponse
            while Confidence1.noResponse:
                Confidence1.draw()
                win.flip()
                hist = set(np.array(Confidence1.getHistory())[:, 0])
                if len(hist) < 2:
                    Confidence1.noResponse = True
                    Confidence1.status = 0
                if keyState[key.LEFT] == True and Confidence1.markerPlacedAt > 0.01:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt - 0.02
                    Confidence1.draw()
                elif keyState[key.LEFT] == True and Confidence1.markerPlacedAt == 0.01:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt - 0.01
                    Confidence1.draw()
                elif keyState[key.RIGHT] == True and Confidence1.markerPlacedAt < 0.99:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt + 0.02
                    Confidence1.draw()
                elif keyState[key.RIGHT] == True and Confidence1.markerPlacedAt == 0.99:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt + 0.01
                    Confidence1.draw()
            Confidence1.response = Confidence1.getRating()
            Confidence1.rt = Confidence1.getRT()
            
         # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=['escape']):
            core.quit()
            
        # refresh the screen
        if continueRoutine:
            win.flip()
        else:
            routineTimer.reset()
    
     #-------Ending Routine "Confidence"-------
    for thisComponent in FeedbackComponents:
        if hasattr(thisComponent, 'setAutoDraw'):
            thisComponent.setAutoDraw(False)

    # store data for trials (TrialHandler)
    thisExp.addData('CalConf', Confidence1.getRating())
    thisExp.addData('CalConf_rt', Confidence1.getRT())
    thisExp.nextEntry()
        
######################### End of Calibration Phase #########################

#------Prepare to start Routine "Instructions3"-------
t = 0
Instructions3Clock.reset()  # clock 
frameN = -1
# update component parameters for each repeat
Instructions3Response = event.BuilderKeyResponse()  # create an object of type KeyResponse
Instructions3Response.status = NOT_STARTED
# keep track of which components have finished
Instructions3Components = []
Instructions3Components.append(Instructions3Text)
Instructions3Components.append(Instructions3Response)
for thisComponent in Instructions3Components:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "Instructions3"-------
continueRoutine = True
while continueRoutine:
    # get current time
    t = Instructions3Clock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *Instructions2Text* updates
    if t >= 0.0 and Instructions3Text.status == NOT_STARTED:
        # keep track of start time/frame for later
        Instructions3Text.tStart = t  # underestimates by a little under one frame
        Instructions3Text.frameNStart = frameN  # exact frame index
        Instructions3Text.setAutoDraw(True)
    
    # *Instructions2Response* updates
    if t >= 0 and Instructions3Response.status == NOT_STARTED:
        # keep track of start time/frame for later
        Instructions3Response.tStart = t  # underestimates by a little under one frame
        Instructions3Response.frameNStart = frameN  # exact frame index
        Instructions3Response.status = STARTED
        # keyboard checking is just starting
        Instructions3Response.clock.reset()  # now t=0
        event.clearEvents(eventType='keyboard')
    if Instructions3Response.status == STARTED:
        theseKeys = event.getKeys(keyList=['space'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            Instructions3Response.keys = theseKeys[-1]  # just the last key pressed
            Instructions3Response.rt = Instructions3Response.clock.getTime()
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in Instructions3Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()
    else:  # this Routine was not non-slip safe so reset non-slip timer
        routineTimer.reset()

#-------Ending Routine "Instructions3"-------
for thisComponent in Instructions3Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if Instructions3Response.keys in ['', [], None]:  # No response was made
   Instructions3Response.keys=None
# store data for thisExp (ExperimentHandler)
thisExp.addData('Instructions3Response.keys',Instructions3Response.keys)
if Instructions3Response.keys != None:  # we had a response
    thisExp.addData('Instructions3Response.rt', Instructions3Response.rt)
thisExp.nextEntry()

#------Prepare to start Routine "Instructions3a"-------
t = 0
Instructions3aClock.reset()  # clock
frameN = -1
# update component parameters for each repeat
Instructions3aResponse = event.BuilderKeyResponse()  # create an object of type KeyResponse
Instructions3aResponse.status = NOT_STARTED
# keep track of which components have finished
Instructions3aComponents = []
Instructions3aComponents.append(Instructions3aText)
Instructions3aComponents.append(Instructions3aResponse)
for thisComponent in Instructions3aComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "Instructions3a"-------
continueRoutine = True
while continueRoutine:
    # get current time
    t = Instructions3aClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame

    # *Instructions2Text* updates
    if t >= 0.0 and Instructions3aText.status == NOT_STARTED:
        # keep track of start time/frame for later
        Instructions3aText.tStart = t  # underestimates by a little under one frame
        Instructions3aText.frameNStart = frameN  # exact frame index
        Instructions3aText.setAutoDraw(True)

    # *Instructions2Response* updates
    if t >= 0 and Instructions3aResponse.status == NOT_STARTED:
        # keep track of start time/frame for later
        Instructions3aResponse.tStart = t  # underestimates by a little under one frame
        Instructions3aResponse.frameNStart = frameN  # exact frame index
        Instructions3aResponse.status = STARTED
        # keyboard checking is just starting
        Instructions3aResponse.clock.reset()  # now t=0
        event.clearEvents(eventType='keyboard')
    if Instructions3aResponse.status == STARTED:
        theseKeys = event.getKeys(keyList=['space'])

        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            Instructions3aResponse.keys = theseKeys[-1]  # just the last key pressed
            Instructions3aResponse.rt = Instructions3aResponse.clock.getTime()
            # a response ends the routine
            continueRoutine = False

    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in Instructions3aComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished

    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()

    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()
    else:  # this Routine was not non-slip safe so reset non-slip timer
        routineTimer.reset()

#-------Ending Routine "Instructions3"-------
for thisComponent in Instructions3aComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if Instructions3aResponse.keys in ['', [], None]:  # No response was made
   Instructions3aResponse.keys=None
# store data for thisExp (ExperimentHandler)
thisExp.addData('Instructions3aResponse.keys',Instructions3aResponse.keys)
if Instructions3aResponse.keys != None:  # we had a response
    thisExp.addData('Instructions3aResponse.rt', Instructions3aResponse.rt)
thisExp.nextEntry()

########################## Phase II Example Staircase ##################################

#create the staircase handler
ConfidenceStaircase = data.StairHandler(startVal = DotDifference,
                          stepType = 'lin', stepSizes=1, nTrials=10,
                          nUp=1, nDown=2, minVal=1, maxVal=49)  #will home in on the 70% threshold
                          

for ThisIncrement in ConfidenceStaircase:
    # Set dot difference for this trial
    DotDifference = ThisIncrement
    #randomise whether the stable dotarray is the smaller or the larger value
    DotDiffDenominator = np.random.choice([-1,1])
    if DotDiffDenominator == -1:
        DotVariable = DotStable - DotDifference
    else:
        DotVariable = DotStable + DotDifference
    #randomise the location of the fixed and variable stimuli
    StableSide= np.random.choice([-1,1])
    if StableSide == -1:
        DotLeft = DotStable
        DotRight = DotVariable
    else:
        DotLeft =DotVariable
        DotRight = DotStable
    # Determine the correct response for the trial
    if DotLeft > DotRight:
        CorrectKey = str('left')
    elif DotRight > DotLeft:
        CorrectKey = str('right')

    #------Prepare to start Routine "ITI"-------
    t = 0
    ITIClock.reset()
    frameN = -1
    routineTimer.add(1.00000)

    Example_Trial += 1
    example_points = 0

     # update component parameters for each repeat
    # Update Keyboard parameters
    event.clearEvents(eventType='keyboard')
    KeyRespCal = event.BuilderKeyResponse()
    KeyRespCal.status = NOT_STARTED
     # keep track of which components have finished
    dot_ITIComponents = []
    dot_ITIComponents.append(CircleLeft)
    dot_ITIComponents.append(CircleRight)
    dot_ITIComponents.append(FixationCrossLeft)
    dot_ITIComponents.append(FixationCrossRight)
    for thisComponent in dot_ITIComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    #-------Start Routine "ITI"-------
    continueRoutine = True
    while continueRoutine and routineTimer.getTime() > 0:
        t = ITIClock.getTime()
        frameN = frameN + 1

        # *CircleLeft* updates
        if t >= 0 and CircleLeft.status == NOT_STARTED:
            CircleLeft.tStart = t
            CircleLeft.frameNStart = frameN
            CircleLeft.setAutoDraw(True)


        # *CircleRight* updates
        if t >= 0 and CircleRight.status == NOT_STARTED:
            CircleRight.tStart = t
            CircleRight.frameNStart = frameN
            CircleRight.setAutoDraw(True)


         # *FixationCrossLeft* updates
        if t >= 0 and FixationCrossLeft.status == NOT_STARTED:
            FixationCrossLeft.tStart = t
            FixationCrossLeft.frameNStart = frameN
            FixationCrossLeft.setAutoDraw(True)


        # *FixationCrossRight* updates
        if t >= 0 and FixationCrossRight.status == NOT_STARTED:
            FixationCrossRight.tStart = t
            FixationCrossRight.frameNStart = frameN
            FixationCrossRight.setAutoDraw(True)


            # check for quit:
            if 'escape' in theseKeys:
                endExpNow = True

        # check if all components have finished
        if not continueRoutine:
            routineTimer.reset()
            break
        continueRoutine = False
        for thisComponent in dot_ITIComponents:
            if hasattr(thisComponent, 'status') and thisComponent.status != FINISHED:
                continueRoutine = True
                break

         # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=['escape']):
            core.quit()

        # refresh the screen
        if continueRoutine:
            win.flip()
        else:
            routineTimer.reset()

     #-------Ending Routine "ITI"-------
    for thisComponent in dot_ITIComponents:
        if hasattr(thisComponent, 'setAutoDraw'):
            thisComponent.setAutoDraw(False)

        #------Prepare to start Routine "choice"-------
    t = 0
    ChoiceClock.reset()
    frameN = -1

     # update component parameters for each repeat
    DotPatchLeft=visual.DotStim(win, color=(1.0,1.0,1.0), dir=270, units='cm',
    nDots=int(DotLeft), fieldShape='circle', fieldPos=(-11.0, 4.0), fieldSize=11,
    dotSize= 15,
    dotLife=-1, #number of frames for each dot to be drawn
    signalDots='same', #are the signal dots the 'same' on each frame? (see Scase et al)
    noiseDots='position', #do the noise dots follow random- 'walk', 'direction', or 'position'
    speed=0.0, coherence=1.0)
    
    DotPatchRight=visual.DotStim(win, color=(1.0,1.0,1.0), dir=270, units='cm',
    nDots=int(DotRight), fieldShape='circle', fieldPos=(11, 4.0),fieldSize=11,
    dotSize= 15,
    dotLife=-1, #number of frames for each dot to be drawn
    signalDots='same', #are the signal dots the 'same' on each frame? (see Scase et al)
    noiseDots='position', #do the noise dots follow random- 'walk', 'direction', or 'position'
    speed=0.0, coherence=1.0)

    DotLeftText.setText(DotLeft)
    DotRightText.setText(DotRight)
    DotDifText.setText(DotDifference)

    # Update Keyboard parameters
    event.clearEvents(eventType='keyboard')
    KeyRespCal = event.BuilderKeyResponse()
    KeyRespCal.status = NOT_STARTED
     # keep track of which components have finished
    dot_choiceComponents = []
    dot_choiceComponents.append(CircleLeft)
    dot_choiceComponents.append(CircleRight)
    dot_choiceComponents.append(DotPatchLeft)
    dot_choiceComponents.append(DotPatchRight)
    dot_choiceComponents.append(FixationCrossLeft)
    dot_choiceComponents.append(FixationCrossRight)
    dot_choiceComponents.append(DotCalibrationInstructions)
    for thisComponent in dot_choiceComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

#-------Start Routine "dot_choice"-------
    continueRoutine = True
    while continueRoutine:
        t = ChoiceClock.getTime()
        frameN = frameN + 1

        # *CircleLeft* updates
        if t >= 0 and CircleLeft.status == NOT_STARTED:
            CircleLeft.tStart = t
            CircleLeft.frameNStart = frameN
            CircleLeft.setAutoDraw(True)


        # *CircleRight* updates
        if t >= 0 and CircleRight.status == NOT_STARTED:
            CircleRight.tStart = t
            CircleRight.frameNStart = frameN
            CircleRight.setAutoDraw(True)


        # *DotPatchLeft* updates
        if t >= 0 and DotPatchLeft.status == NOT_STARTED:
            DotPatchLeft.tStart = t
            DotPatchLeft.frameNStart = frameN
            DotPatchLeft.setAutoDraw(True)
        if DotPatchLeft.status == STARTED and t >= (0 + (0.7-win.monitorFramePeriod*0.75)): #most of one frame period left
            DotPatchLeft.setAutoDraw(False)


        # *DotPatchRight* updates
        if t >= 0 and DotPatchRight.status == NOT_STARTED:
            DotPatchRight.tStart = t
            DotPatchRight.frameNStart = frameN
            DotPatchRight.setAutoDraw(True)
        if DotPatchRight.status == STARTED and t >= (0 + (0.7-win.monitorFramePeriod*0.75)): #most of one frame period left
            DotPatchRight.setAutoDraw(False)

        # *FixationCrossLeft* updates
        if t >= 0.7 and FixationCrossLeft.status == NOT_STARTED:
            FixationCrossLeft.tStart = t
            FixationCrossLeft.frameNStart = frameN
            FixationCrossLeft.setAutoDraw(True)

        # *FixationCrossRight* updates
        if t >= 0.7 and FixationCrossRight.status == NOT_STARTED:
            FixationCrossRight.tStart = t
            FixationCrossRight.frameNStart = frameN
            FixationCrossRight.setAutoDraw(True)

        # *DotExampleInstructions* updates
        if t >= 0 and DotCalibrationInstructions.status == NOT_STARTED:
            DotCalibrationInstructions.tStart = t
            DotCalibrationInstructions.frameNStart = frameN
            DotCalibrationInstructions.setAutoDraw(True)

        # *key_resp_choice* updates
        if t >= 0.7 and KeyRespCal.status == NOT_STARTED:
            KeyRespCal.tStart = t
            KeyRespCal.frameNStart = frameN
            KeyRespCal.status = STARTED
            KeyRespCal.clock.reset()
        if KeyRespCal.status == STARTED:
            theseKeys = event.getKeys(keyList=['left', 'right'])

            # check for quit:
            if 'escape' in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:
                KeyRespCal.keys = theseKeys[-1]
                KeyRespCal.rt = KeyRespCal.clock.getTime()
                continueRoutine = False

        # check if all components have finished
        if not continueRoutine:
            routineTimer.reset()
            break
        continueRoutine = False
        for thisComponent in dot_choiceComponents:
            if hasattr(thisComponent, 'status') and thisComponent.status != FINISHED:
                continueRoutine = True
                break

         # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=['escape']):
            core.quit()

        # refresh the screen
        if continueRoutine:
            win.flip()
        else:
            routineTimer.reset()

     #-------Ending Routine "choice"-------
    for thisComponent in dot_choiceComponents:
        if hasattr(thisComponent, 'setAutoDraw'):
            thisComponent.setAutoDraw(False)
    # check responses
    if KeyRespCal.keys in ['', [], None]:
        KeyRespCal.keys = None
        # was no response the correct answer?!
    if KeyRespCal.keys == CorrectKey: KeyRespCal.corr = 1  # correct non-response
    else: KeyRespCal.corr = 0  # failed to respond (incorrectly)

    # Tell the staircase if the participant is correct (TrialHandler)
    ConfidenceStaircase.addResponse(KeyRespCal.corr)
    
    # Store data for experiment:
    thisExp.addData('Cal2Correct', KeyRespCal.corr)
    thisExp.addData('Cal2CorrectKey', CorrectKey)
    thisExp.addData('Cal2Response', KeyRespCal.keys)
    if KeyRespCal.keys != None:
        thisExp.addData('Cal2RT', KeyRespCal.rt)
    thisExp.addData('Cal2DotDifference', DotDifference)
    thisExp.addData('Cal2DotNumberRight', DotPatchRight.nDots)
    thisExp.addData('Cal2DotNumberLeft', DotPatchLeft.nDots)
    
    #------Prepare to start Routine "conf"-------
    t = 0
    ConfClock.reset()
    frameN = -1
    # update component parameters for each repeat
    Confidence1.reset()
    # keep track of which components have finished
    ConfComponents = []
    ConfComponents.append(CircleLeft)
    ConfComponents.append(CircleRight)
    ConfComponents.append(FixationCrossLeft)
    ConfComponents.append(FixationCrossRight)
    ConfComponents.append(FeedbackPlusLeft)
    ConfComponents.append(FeedbackPlusRight)
    ConfComponents.append(FeedbackCrossLeft)
    ConfComponents.append(FeedbackCrossRight)
    ConfComponents.append(Confidence1)
    for thisComponent in ConfComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    #-------Start Routine "conf"-------
# checking to see if participants responded to the choice question
# if not,it skips this iteration of the confidence routine
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = ConfClock.getTime()
        frameN = frameN + 1

        # *CircleLeft* updates
        if t >= 0.0 and CircleLeft.status == NOT_STARTED:
            CircleLeft.tStart = t
            CircleLeft.frameNStart = frameN
            CircleLeft.setAutoDraw(True)

        # *CircleRight* updates
        if t >= 0.0 and CircleRight.status == NOT_STARTED:
            CircleRight.tStart = t
            CircleRight.frameNStart = frameN
            CircleRight.setAutoDraw(True)

        # *FixationCrossLeft* updates
        if t >= 0.0 and FixationCrossLeft.status == NOT_STARTED:
            FixationCrossLeft.tStart = t
            FixationCrossLeft.frameNStart = frameN
            FixationCrossLeft.setAutoDraw(True)

        # *FixationCrossRight* updates
        if t >= 0.0 and FixationCrossRight.status == NOT_STARTED:
            FixationCrossRight.tStart = t
            FixationCrossRight.frameNStart = frameN
            FixationCrossRight.setAutoDraw(True)

        # *FeedbackPlusLeft* updates
        if KeyRespCal.keys == 'left':
            if t >= 0.0 and FeedbackPlusLeft.status == NOT_STARTED:
                FeedbackPlusLeft.tStart = t
                FeedbackPlusLeft.frameNStart = frameN
                FeedbackPlusLeft.setAutoDraw(True)

        # *FeedbackPlusRight* updates
        if KeyRespCal.keys == 'right':
            if t >= 0.0 and FeedbackPlusRight.status == NOT_STARTED:
                FeedbackPlusRight.tStart = t
                FeedbackPlusRight.frameNStart = frameN
                FeedbackPlusRight.setAutoDraw(True)

        # *FeedbackCrossLeft* updates
        if KeyRespCal.keys == 'right':
            if t >= 0.0 and FeedbackCrossLeft.status == NOT_STARTED:
                FeedbackCrossLeft.tStart = t
                FeedbackCrossLeft.frameNStart = frameN
                FeedbackCrossLeft.setAutoDraw(True)

        # *FeedbackCrossRight* updates
        if KeyRespCal.keys == 'left':
            if t >= 0.0 and FeedbackCrossRight.status == NOT_STARTED:
                FeedbackCrossRight.tStart = t
                FeedbackCrossRight.frameNStart = frameN
                FeedbackCrossRight.setAutoDraw(True)

        # *confidence* updates
        if t > 0.2:
            Confidence1.draw()
            continueRoutine = Confidence1.noResponse
            if Confidence1.noResponse == False:
                Confidence1.response = Confidence1.getRating()
                Confidence1.rt = Confidence1.getRT()
            elif Confidence1.noResponse == True:
                if keyState[key.LEFT] == True and Confidence1.markerPlacedAt > 0.01:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt - 0.02
                    Confidence1.draw()
                elif keyState[key.LEFT] == True and Confidence1.markerPlacedAt == 0.01:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt - 0.01
                    Confidence1.draw()
                elif keyState[key.RIGHT] == True and Confidence1.markerPlacedAt < 0.99:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt + 0.02
                    Confidence1.draw()
                elif keyState[key.RIGHT] == True and Confidence1.markerPlacedAt == 0.99:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt + 0.01
                    Confidence1.draw()

        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=['escape']):
            core.quit()
        
        # refresh the screen
        if continueRoutine:
            win.flip()
        else:
            routineTimer.reset()
    
    #-------Ending Routine "conf"-------
    for thisComponent in ConfComponents:
        if hasattr(thisComponent, 'setAutoDraw'):
            thisComponent.setAutoDraw(False)
    # store data for binary (TrialHandler)
    
    example_points += 100*(1-(KeyRespCal.corr - Confidence1.getRating())**2)
    
    thisExp.addData('Cal2Conf', Confidence1.getRating())
    thisExp.addData('Cal2Conf_RT', Confidence1.getRT())

    #-------Prepare to start routine "CIP"------
    t = 0
    CipClock.reset()
    frameN = -1
    routineTimer.add(4.000000)
    now = []

    # Update Keyboard parameters
    event.clearEvents(eventType='keyboard')
    CipKey = event.BuilderKeyResponse()
    CipKey.status = NOT_STARTED
     # keep track of which components have finished
    CipComponents = []
    CipComponents.append(CircleLeft)
    CipComponents.append(CircleRight)
    CipComponents.append(DotPatchLeft)
    CipComponents.append(DotPatchRight)
    CipComponents.append(FeedbackPlusLeft)
    CipComponents.append(FeedbackPlusRight)
    CipComponents.append(FeedbackCrossLeft)
    CipComponents.append(FeedbackCrossRight)
    CipComponents.append(Instr1Cip)
    for thisComponent in CipComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    #-------Start Routine "Cip"-------
    # checking to see if participants responded to the choice question
    # if not,it skips this iteration of the confidence routine
    continueRoutine = True
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = CipClock.getTime()
        frameN = frameN + 1

        # *CircleLeft* updates
        if t >= 0.0 and CircleLeft.status == NOT_STARTED:
            CircleLeft.tStart = t
            CircleLeft.frameNStart = frameN
            CircleLeft.setAutoDraw(True)

        # *CircleRight* updates
        if t >= 0.0 and CircleRight.status == NOT_STARTED:
            CircleRight.tStart = t
            CircleRight.frameNStart = frameN
            CircleRight.setAutoDraw(True)

        # *Instr1Cip* updates
        if t >= 0.0 and Instr1Cip.status == NOT_STARTED:
            Instr1Cip.tStart = t
            Instr1Cip.frameNStart = frameN
            Instr1Cip.setAutoDraw(True)

        # *FeedbackPlusLeft* updates
        if KeyRespCal.keys == 'left':
            if t >= 0.0 and FeedbackPlusLeft.status == NOT_STARTED:
                FeedbackPlusLeft.tStart = t
                FeedbackPlusLeft.frameNStart = frameN
                FeedbackPlusLeft.setAutoDraw(True)

        # *FeedbackPlusRight* updates
        if KeyRespCal.keys == 'right':
            if t >= 0.0 and FeedbackPlusRight.status == NOT_STARTED:
                FeedbackPlusRight.tStart = t
                FeedbackPlusRight.frameNStart = frameN
                FeedbackPlusRight.setAutoDraw(True)

        # *FeedbackCrossLeft* updates
        if KeyRespCal.keys == 'right':
            if t >= 0.0 and FeedbackCrossLeft.status == NOT_STARTED:
                FeedbackCrossLeft.tStart = t
                FeedbackCrossLeft.frameNStart = frameN
                FeedbackCrossLeft.setAutoDraw(True)

        # *FeedbackCrossRight* updates
        if KeyRespCal.keys == 'left':
            if t >= 0.0 and FeedbackCrossRight.status == NOT_STARTED:
                FeedbackCrossRight.tStart = t
                FeedbackCrossRight.frameNStart = frameN
                FeedbackCrossRight.setAutoDraw(True)

        # *CipKey* updates
        if t >= 0.0 and CipKey.status == NOT_STARTED:
            # keep track of start time/frame for later
            CipKey.tStart = t  # underestimates by a little under one frame
            CipKey.frameNStart = frameN  # exact frame index
            CipKey.status = STARTED
            # keyboard checking is just starting
            CipKey.clock.reset()  # now t=0
            event.clearEvents(eventType='keyboard')
        if CipKey.status == STARTED and t >= (0.0 + (4.0-win.monitorFramePeriod*0.75)): #most of one frame period left
            CipKey.status = STOPPED
        if CipKey.status == STARTED:
            theseKeys = event.getKeys(keyList=['left', 'right'])

            # check for quit:
            if 'escape' in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:
                now = theseKeys[-1]
                CipKey.keys.extend(theseKeys) # storing all keys
                CipKey.rt.append(CipKey.clock.getTime())
                theseKeys = event.clearEvents(eventType=None)

        # *DotPatchLeft* updates
        if t >= 0 and DotPatchLeft.status == NOT_STARTED and now == 'left':
            DotPatchLeft.tStart = t
            DotPatchLeft.frameNStart = frameN
            DotPatchLeft.setAutoDraw(True)
        if DotPatchLeft.status == STARTED and now == 'right':
            DotPatchLeft.setAutoDraw(False)
            DotPatchLeft.status = NOT_STARTED


        # *DotPatchRight* updates
        if t >= 0 and DotPatchRight.status == NOT_STARTED and now == 'right':
            DotPatchRight.tStart = t
            DotPatchRight.frameNStart = frameN
            DotPatchRight.setAutoDraw(True)
        if DotPatchRight.status == STARTED and now == 'left':
            DotPatchRight.setAutoDraw(False)
            DotPatchRight.status = NOT_STARTED

        # check if all components have finished
        if not continueRoutine:
            break
        continueRoutine = False
        for thisComponent in CipComponents:
            if hasattr(thisComponent, 'status') and thisComponent.status != FINISHED:
                continueRoutine = True
                break

         # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=['escape']):
            core.quit()

        # refresh the screen
        if continueRoutine:
            win.flip()
        else:
            routineTimer.reset()

     #-------Ending Routine "CIP"-------
    for thisComponent in CipComponents:
        if hasattr(thisComponent, 'setAutoDraw'):
            thisComponent.setAutoDraw(False)

    # check responses
    if CipKey.keys in ['', [], None]:
        CipKey.keys = None

    # Store data for experiment:
    thisExp.addData('CalCipResponse', CipKey.keys)
    if CipKey.keys != None:
        thisExp.addData('CalCipRT', CipKey.rt)

    #------Prepare to start Routine "change"-------
    t = 0
    Choice2Clock.reset()
    frameN = -1

    # Update Keyboard parameters
    event.clearEvents(eventType='keyboard')
    KeyRespChange = event.BuilderKeyResponse()
    KeyRespChange.status = NOT_STARTED
     # keep track of which components have finished
    changeComponents = []
    changeComponents.append(CircleLeft)
    changeComponents.append(CircleRight)
    changeComponents.append(FixationCrossLeft)
    changeComponents.append(FixationCrossRight)
    changeComponents.append(FeedbackPlusLeft)
    changeComponents.append(FeedbackPlusRight)
    changeComponents.append(FeedbackCrossLeft)
    changeComponents.append(FeedbackCrossRight)
    changeComponents.append(DotCalibrationInstructions2)
    for thisComponent in changeComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

#-------Start Routine "change"-------
    continueRoutine = True
    while continueRoutine:
        t = Choice2Clock.getTime()
        frameN = frameN + 1

        # *CircleLeft* updates
        if t >= 0 and CircleLeft.status == NOT_STARTED:
            CircleLeft.tStart = t
            CircleLeft.frameNStart = frameN
            CircleLeft.setAutoDraw(True)


        # *CircleRight* updates
        if t >= 0 and CircleRight.status == NOT_STARTED:
            CircleRight.tStart = t
            CircleRight.frameNStart = frameN
            CircleRight.setAutoDraw(True)

         #*FixationCrossLeft* updates
        if t >= 0 and FixationCrossLeft.status == NOT_STARTED:
            FixationCrossLeft.tStart = t
            FixationCrossLeft.frameNStart = frameN
            FixationCrossLeft.setAutoDraw(True)

         #*FixationCrossRight* updates
        if t >= 0 and FixationCrossRight.status == NOT_STARTED:
            FixationCrossRight.tStart = t
            FixationCrossRight.frameNStart = frameN
            FixationCrossRight.setAutoDraw(True)

        # *DotExampleInstructions2* updates
        if t >= 0 and DotCalibrationInstructions2.status == NOT_STARTED:
            DotCalibrationInstructions2.tStart = t
            DotCalibrationInstructions2.frameNStart = frameN
            DotCalibrationInstructions2.setAutoDraw(True)

        # *key_resp_choice* updates
        if t >= 0.7 and KeyRespChange.status == NOT_STARTED:
            theseKeys = event.clearEvents(eventType=None)
            KeyRespChange.tStart = t
            KeyRespChange.frameNStart = frameN
            KeyRespChange.status = STARTED
            KeyRespChange.clock.reset()
        if KeyRespChange.status == STARTED:
            theseKeys = event.getKeys(keyList=['left', 'right'])

            # check for quit:
            if 'escape' in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:
                KeyRespChange.keys = theseKeys[-1]
                KeyRespChange.rt = KeyRespChange.clock.getTime()
                continueRoutine = False

        # check if all components have finished
        if not continueRoutine:
            routineTimer.reset()
            break
        continueRoutine = False
        for thisComponent in changeComponents:
            if hasattr(thisComponent, 'status') and thisComponent.status != FINISHED:
                continueRoutine = True
                break

         # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=['escape']):
            core.quit()

        # refresh the screen
        if continueRoutine:
            win.flip()
        else:
            routineTimer.reset()

     #-------Ending Routine "change"-------
    for thisComponent in changeComponents:
        if hasattr(thisComponent, 'setAutoDraw'):
            thisComponent.setAutoDraw(False)
    # check responses
    if KeyRespChange.keys in ['', [], None]:
        KeyRespChange.keys = None
        # was no response the correct answer?!
    if KeyRespChange.keys == CorrectKey: KeyRespChange.corr = 1  # correct non-response
    else: KeyRespChange.corr = 0  # failed to respond (incorrectly)

    # Store data for experiment:
    thisExp.addData('ChangeCal2Correct', KeyRespChange.corr)
    thisExp.addData('ChangeCal2Response', KeyRespChange.keys)
    if KeyRespChange.keys != None:
        thisExp.addData('ChangeCal2RT', KeyRespChange.rt)

      #------Prepare to start Routine "conf2"-------
    t = 0
    Conf2Clock.reset()
    frameN = -1
    # update component parameters for each repeat
    Confidence1.reset()
    # keep track of which components have finished
    Conf2Components = []
    Conf2Components.append(CircleLeft)
    Conf2Components.append(CircleRight)
    Conf2Components.append(FixationCrossLeft)
    Conf2Components.append(FixationCrossRight)
    Conf2Components.append(FeedbackPlusLeft)
    Conf2Components.append(FeedbackPlusRight)
    Conf2Components.append(FeedbackCrossLeft)
    Conf2Components.append(FeedbackCrossRight)
    #Conf2Components.append(Instr1Conf2)
    Conf2Components.append(Confidence1)
    for thisComponent in Conf2Components:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    #-------Start Routine "conf"-------
# checking to see if participants responded to the choice question
# if not,it skips this iteration of the confidence routine
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = Conf2Clock.getTime()
        frameN = frameN + 1

        # *CircleLeft* updates
        if t >= 0.0 and CircleLeft.status == NOT_STARTED:
            CircleLeft.tStart = t
            CircleLeft.frameNStart = frameN
            CircleLeft.setAutoDraw(True)

        # *CircleRight* updates
        if t >= 0.0 and CircleRight.status == NOT_STARTED:
            CircleRight.tStart = t
            CircleRight.frameNStart = frameN
            CircleRight.setAutoDraw(True)

        # *FixationCrossLeft* updates
        if t >= 0.0 and FixationCrossLeft.status == NOT_STARTED:
            FixationCrossLeft.tStart = t
            FixationCrossLeft.frameNStart = frameN
            FixationCrossLeft.setAutoDraw(True)

        # *FixationCrossRight* updates
        if t >= 0.0 and FixationCrossRight.status == NOT_STARTED:
            FixationCrossRight.tStart = t
            FixationCrossRight.frameNStart = frameN
            FixationCrossRight.setAutoDraw(True)

        # *FeedbackPlusLeft* updates
        if KeyRespChange.keys == 'left':
            if t >= 0.0 and FeedbackPlusLeft.status == NOT_STARTED:
                FeedbackPlusLeft.tStart = t
                FeedbackPlusLeft.frameNStart = frameN
                FeedbackPlusLeft.setAutoDraw(True)

        # *FeedbackPlusRight* updates
        if KeyRespChange.keys == 'right':
            if t >= 0.0 and FeedbackPlusRight.status == NOT_STARTED:
                FeedbackPlusRight.tStart = t
                FeedbackPlusRight.frameNStart = frameN
                FeedbackPlusRight.setAutoDraw(True)

        # *FeedbackCrossLeft* updates
        if KeyRespChange.keys == 'right':
            if t >= 0.0 and FeedbackCrossLeft.status == NOT_STARTED:
                FeedbackCrossLeft.tStart = t
                FeedbackCrossLeft.frameNStart = frameN
                FeedbackCrossLeft.setAutoDraw(True)

        # *FeedbackCrossRight* updates
        if KeyRespChange.keys == 'left':
            if t >= 0.0 and FeedbackCrossRight.status == NOT_STARTED:
                FeedbackCrossRight.tStart = t
                FeedbackCrossRight.frameNStart = frameN
                FeedbackCrossRight.setAutoDraw(True)

        # *confidence* updates
        if t > 0.2:
            Confidence1.draw()
            continueRoutine = Confidence1.noResponse
            if Confidence1.noResponse == False:
                Confidence1.response = Confidence1.getRating()
                Confidence1.rt = Confidence1.getRT()
            elif Confidence1.noResponse == True:
                if keyState[key.LEFT] == True and Confidence1.markerPlacedAt > 0.01:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt - 0.02
                    Confidence1.draw()
                elif keyState[key.LEFT] == True and Confidence1.markerPlacedAt == 0.01:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt - 0.01
                    Confidence1.draw()
                elif keyState[key.RIGHT] == True and Confidence1.markerPlacedAt < 0.99:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt + 0.02
                    Confidence1.draw()
                elif keyState[key.RIGHT] == True and Confidence1.markerPlacedAt == 0.99:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt + 0.01
                    Confidence1.draw()

        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=['escape']):
            core.quit()

        # refresh the screen
        if continueRoutine:
            win.flip()
        else:
            routineTimer.reset()

    #-------Ending Routine "conf"-------
    for thisComponent in Conf2Components:
        if hasattr(thisComponent, 'setAutoDraw'):
            thisComponent.setAutoDraw(False)
            
    example_points += 100*(1-(float(KeyRespChange.corr) - Confidence1.getRating())**2)

    # store data for binary (TrialHandler)
    thisExp.addData('Cal2Conf2', Confidence1.getRating())
    thisExp.addData('Cal2Conf2_RT', Confidence1.getRT())
    
    #------Prepare to start Routine "Feedback"-------
    t = 0
    FeedbackClock.reset()  # clock 
    frameN = -1
# update component parameters for each repeat

    FeedbackText.setText(text=u'You have earned {0} example points in this trial. Press spacebar to continue.' .format(int(example_points)))

    FeedbackResponse = event.BuilderKeyResponse()  # create an object of type KeyResponse
    FeedbackResponse.status = NOT_STARTED
# keep track of which components have finished
    FeedbackComponents = []
    FeedbackComponents.append(FeedbackText)
    FeedbackComponents.append(FeedbackResponse)
    for thisComponent in FeedbackComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    #-------Start Routine "Feedback"-------
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = FeedbackClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
    
        # *FeedbackText* updates
        if t >= 0.0 and FeedbackText.status == NOT_STARTED:
        # keep track of start time/frame for later
            FeedbackText.tStart = t  # underestimates by a little under one frame
            FeedbackText.frameNStart = frameN  # exact frame index
            FeedbackText.setAutoDraw(True)
    
        # *FeedbackResponse* updates
        if t >= 0 and FeedbackResponse.status == NOT_STARTED:
            # keep track of start time/frame for later
            FeedbackResponse.tStart = t  # underestimates by a little under one frame
            FeedbackResponse.frameNStart = frameN  # exact frame index
            FeedbackResponse.status = STARTED
            # keyboard checking is just starting
            FeedbackResponse.clock.reset()  # now t=0
            event.clearEvents(eventType='keyboard')
        if FeedbackResponse.status == STARTED:
            theseKeys = event.getKeys(keyList=['space'])
        
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                FeedbackResponse.keys = theseKeys[-1]  # just the last key pressed
                FeedbackResponse.rt = FeedbackResponse.clock.getTime()
                # a response ends the routine
                continueRoutine = False
    
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineTimer.reset()  # if we abort early the non-slip timer needs reset
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in FeedbackComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
    
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
    
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
        else:  # this Routine was not non-slip safe so reset non-slip timer
            routineTimer.reset()

    #-------Ending Routine "Feedback"-------
    for thisComponent in FeedbackComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if FeedbackResponse.keys in ['', [], None]:  # No response was made
        FeedbackResponse.keys=None
    # store data for thisExp (ExperimentHandler)
    thisExp.addData('FeedbackResponse.keys',FeedbackResponse.keys)
    if FeedbackResponse.keys != None:  # we had a response
        thisExp.addData('FeedbackResponse.rt', FeedbackResponse.rt)
        thisExp.nextEntry()

#################################### End of Phase II Example #####################################

#------Prepare to start Routine "Instructions4"-------
t = 0
Instructions4Clock.reset()  # clock 
frameN = -1
# update component parameters for each repeat
Instructions4Response = event.BuilderKeyResponse()  # create an object of type KeyResponse
Instructions4Response.status = NOT_STARTED
# keep track of which components have finished
Instructions4Components = []
Instructions4Components.append(Instructions4Text)
Instructions4Components.append(Instructions4Response)
for thisComponent in Instructions4Components:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "Instructions4"-------
continueRoutine = True
while continueRoutine:
    # get current time
    t = Instructions3Clock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *Instructions4Text* updates
    if t >= 0.0 and Instructions4Text.status == NOT_STARTED:
        # keep track of start time/frame for later
        Instructions4Text.tStart = t  # underestimates by a little under one frame
        Instructions4Text.frameNStart = frameN  # exact frame index
        Instructions4Text.setAutoDraw(True)
    
    # *Instructions4Response* updates
    if t >= 0 and Instructions4Response.status == NOT_STARTED:
        # keep track of start time/frame for later
        Instructions4Response.tStart = t  # underestimates by a little under one frame
        Instructions4Response.frameNStart = frameN  # exact frame index
        Instructions4Response.status = STARTED
        # keyboard checking is just starting
        Instructions4Response.clock.reset()  # now t=0
        event.clearEvents(eventType='keyboard')
    if Instructions4Response.status == STARTED:
        theseKeys = event.getKeys(keyList=['space'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            Instructions4Response.keys = theseKeys[-1]  # just the last key pressed
            Instructions4Response.rt = Instructions4Response.clock.getTime()
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in Instructions4Components:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()
    else:  # this Routine was not non-slip safe so reset non-slip timer
        routineTimer.reset()

#-------Ending Routine "Instructions4"-------
for thisComponent in Instructions4Components:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if Instructions4Response.keys in ['', [], None]:  # No response was made
   Instructions4Response.keys=None
# store data for thisExp (ExperimentHandler)
thisExp.addData('Instructions4Response.keys',Instructions4Response.keys)
if Instructions4Response.keys != None:  # we had a response
    thisExp.addData('Instructions4Response.rt', Instructions4Response.rt)
thisExp.nextEntry()

########################## Main Staircase ##################################

# create a counter to control the breaks within the stairhandler
RestCounter = 0
#create the staircase handler
MainStaircase = data.StairHandler(startVal = DotDifference,
                          stepType = 'lin', stepSizes=1, nTrials=200,
                          nUp=1, nDown=2, minVal=1, maxVal=49)  #will home in on the 80% threshold
                          

for ThisIncrement in MainStaircase:
    # Increase the RestCounter
    RestCounter += 1
    # Set dot difference for this trial
    DotDifference = ThisIncrement
    #randomise whether the stable dotarray is the smaller or the larger value
    DotDiffDenominator = np.random.choice([-1,1])
    if DotDiffDenominator == -1:
        DotVariable = DotStable - DotDifference
    else:
        DotVariable = DotStable + DotDifference
    #randomise the location of the fixed and variable stimuli
    StableSide= np.random.choice([-1,1])
    if StableSide == -1:
        DotLeft = DotStable
        DotRight = DotVariable
    else:
        DotLeft =DotVariable
        DotRight = DotStable
    # Determine the correct response for the trial
    if DotLeft > DotRight:
        CorrectKey = str('left')
    elif DotRight > DotLeft:
        CorrectKey = str('right')

    time1 = globalClock.getTime()
    thisExp.addData('time1', time1)

    #-------Start Routine "ITI"-------
    continueRoutine = True
    while continueRoutine and routineTimer.getTime() > 0:
        t = ITIClock.getTime()
        frameN = frameN + 1

        # *CircleLeft* updates
        if t >= 0 and CircleLeft.status == NOT_STARTED:
            CircleLeft.tStart = t
            CircleLeft.frameNStart = frameN
            CircleLeft.setAutoDraw(True)


        # *CircleRight* updates
        if t >= 0 and CircleRight.status == NOT_STARTED:
            CircleRight.tStart = t
            CircleRight.frameNStart = frameN
            CircleRight.setAutoDraw(True)


         # *FixationCrossLeft* updates
        if t >= 0 and FixationCrossLeft.status == NOT_STARTED:
            FixationCrossLeft.tStart = t
            FixationCrossLeft.frameNStart = frameN
            FixationCrossLeft.setAutoDraw(True)


        # *FixationCrossRight* updates
        if t >= 0 and FixationCrossRight.status == NOT_STARTED:
            FixationCrossRight.tStart = t
            FixationCrossRight.frameNStart = frameN
            FixationCrossRight.setAutoDraw(True)


            # check for quit:
            if 'escape' in theseKeys:
                endExpNow = True

        # check if all components have finished
        if not continueRoutine:
            routineTimer.reset()
            break
        continueRoutine = False
        for thisComponent in dot_ITIComponents:
            if hasattr(thisComponent, 'status') and thisComponent.status != FINISHED:
                continueRoutine = True
                break

         # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=['escape']):
            core.quit()

        # refresh the screen
        if continueRoutine:
            win.flip()
        else:
            routineTimer.reset()

     #-------Ending Routine "ITI"-------
    for thisComponent in dot_ITIComponents:
        if hasattr(thisComponent, 'setAutoDraw'):
            thisComponent.setAutoDraw(False)
            
    time2 = globalClock.getTime()
    thisExp.addData('time2', time2)

    #------Prepare to start Routine "choice"-------
    t = 0
    ChoiceClock.reset()
    frameN = -1

    Trial += 1

     # update component parameters for each repeat
    DotPatchLeft=visual.DotStim(win, color=(1.0,1.0,1.0), dir=270, units='cm',
    nDots=int(DotLeft), fieldShape='circle', fieldPos=(-11.0, 4.0), fieldSize=11,
    dotSize= 15,
    dotLife=-1, #number of frames for each dot to be drawn
    signalDots='same', #are the signal dots the 'same' on each frame? (see Scase et al)
    noiseDots='position', #do the noise dots follow random- 'walk', 'direction', or 'position'
    speed=0.0, coherence=1.0)
    
    DotPatchRight=visual.DotStim(win, color=(1.0,1.0,1.0), dir=270, units='cm',
    nDots=int(DotRight), fieldShape='circle', fieldPos=(11, 4.0),fieldSize=11,
    dotSize= 15,
    dotLife=-1, #number of frames for each dot to be drawn
    signalDots='same', #are the signal dots the 'same' on each frame? (see Scase et al)
    noiseDots='position', #do the noise dots follow random- 'walk', 'direction', or 'position'
    speed=0.0, coherence=1.0)
    
    DotLeftText.setText(DotLeft)
    DotRightText.setText(DotRight)
    DotDifText.setText(DotDifference)

    # Update Keyboard parameters
    event.clearEvents(eventType='keyboard')
    KeyRespCal = event.BuilderKeyResponse()
    KeyRespCal.status = NOT_STARTED
     # keep track of which components have finished
    dot_choiceComponents = []
    dot_choiceComponents.append(CircleLeft)
    dot_choiceComponents.append(CircleRight)
    dot_choiceComponents.append(DotPatchLeft)
    dot_choiceComponents.append(DotPatchRight)
    dot_choiceComponents.append(FixationCrossLeft)
    dot_choiceComponents.append(FixationCrossRight)
    dot_choiceComponents.append(DotCalibrationInstructions)
    for thisComponent in dot_choiceComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
            
    time3 = globalClock.getTime()
    thisExp.addData('time3', time3)

#-------Start Routine "dot_choice"-------
    continueRoutine = True
    while continueRoutine:
        t = ChoiceClock.getTime()
        frameN = frameN + 1

        # *CircleLeft* updates
        if t >= 0 and CircleLeft.status == NOT_STARTED:
            CircleLeft.tStart = t
            CircleLeft.frameNStart = frameN
            CircleLeft.setAutoDraw(True)


        # *CircleRight* updates
        if t >= 0 and CircleRight.status == NOT_STARTED:
            CircleRight.tStart = t
            CircleRight.frameNStart = frameN
            CircleRight.setAutoDraw(True)


        # *DotPatchLeft* updates
        if t >= 0 and DotPatchLeft.status == NOT_STARTED:
            DotPatchLeft.tStart = t
            DotPatchLeft.frameNStart = frameN
            DotPatchLeft.setAutoDraw(True)
        if DotPatchLeft.status == STARTED and t >= (0 + (0.7-win.monitorFramePeriod*0.75)): #most of one frame period left
            DotPatchLeft.setAutoDraw(False)


        # *DotPatchRight* updates
        if t >= 0 and DotPatchRight.status == NOT_STARTED:
            DotPatchRight.tStart = t
            DotPatchRight.frameNStart = frameN
            DotPatchRight.setAutoDraw(True)
        if DotPatchRight.status == STARTED and t >= (0 + (0.7-win.monitorFramePeriod*0.75)): #most of one frame period left
            DotPatchRight.setAutoDraw(False)

        # *FixationCrossLeft* updates
        if t >= 0.7 and FixationCrossLeft.status == NOT_STARTED:
            FixationCrossLeft.tStart = t
            FixationCrossLeft.frameNStart = frameN
            FixationCrossLeft.setAutoDraw(True)

        # *FixationCrossRight* updates
        if t >= 0.7 and FixationCrossRight.status == NOT_STARTED:
            FixationCrossRight.tStart = t
            FixationCrossRight.frameNStart = frameN
            FixationCrossRight.setAutoDraw(True)

        # *DotExampleInstructions* updates
        if t >= 0 and DotCalibrationInstructions.status == NOT_STARTED:
            DotCalibrationInstructions.tStart = t
            DotCalibrationInstructions.frameNStart = frameN
            DotCalibrationInstructions.setAutoDraw(True)

        # *key_resp_choice* updates
        if t >= 0.7 and KeyRespCal.status == NOT_STARTED:
            KeyRespCal.tStart = t
            KeyRespCal.frameNStart = frameN
            KeyRespCal.status = STARTED
            KeyRespCal.clock.reset()
        if KeyRespCal.status == STARTED:
            theseKeys = event.getKeys(keyList=['left', 'right'])

            # check for quit:
            if 'escape' in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:
                KeyRespCal.keys = theseKeys[-1]
                KeyRespCal.rt = KeyRespCal.clock.getTime()
                continueRoutine = False

        # check if all components have finished
        if not continueRoutine:
            routineTimer.reset()
            break
        continueRoutine = False
        for thisComponent in dot_choiceComponents:
            if hasattr(thisComponent, 'status') and thisComponent.status != FINISHED:
                continueRoutine = True
                break

         # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=['escape']):
            core.quit()

        # refresh the screen
        if continueRoutine:
            win.flip()
        else:
            routineTimer.reset()

     #-------Ending Routine "choice"-------
    for thisComponent in dot_choiceComponents:
        if hasattr(thisComponent, 'setAutoDraw'):
            thisComponent.setAutoDraw(False)
    # check responses
    if KeyRespCal.keys in ['', [], None]:
        KeyRespCal.keys = None
        # was no response the correct answer?!
    if KeyRespCal.keys == CorrectKey: KeyRespCal.corr = 1  # correct non-response
    else: KeyRespCal.corr = 0  # failed to respond (incorrectly)

    # Tell the staircase if the participant is correct (TrialHandler)
    MainStaircase.addResponse(KeyRespCal.corr)
    
    # Store data for experiment:
    thisExp.addData('Correct', KeyRespCal.corr)
    thisExp.addData('CorrectKey', CorrectKey)
    thisExp.addData('Response', KeyRespCal.keys)
    if KeyRespCal.keys != None:
        thisExp.addData('RT', KeyRespCal.rt)
    thisExp.addData('DotDifference', DotDifference)
    thisExp.addData('DotNumberRight', DotPatchRight.nDots)
    thisExp.addData('DotNumberLeft', DotPatchLeft.nDots)
    
    time4 = globalClock.getTime()
    thisExp.addData('time4', time4)
    
    #------Prepare to start Routine "conf"-------
    t = 0
    ConfClock.reset()
    frameN = -1
    # update component parameters for each repeat
    Confidence1.reset()
    # keep track of which components have finished
    ConfComponents = []
    ConfComponents.append(CircleLeft)
    ConfComponents.append(CircleRight)
    ConfComponents.append(FixationCrossLeft)
    ConfComponents.append(FixationCrossRight)
    ConfComponents.append(FeedbackPlusLeft)
    ConfComponents.append(FeedbackPlusRight)
    ConfComponents.append(FeedbackCrossLeft)
    ConfComponents.append(FeedbackCrossRight)
    ConfComponents.append(Confidence1)
    for thisComponent in ConfComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
            
    time5 = globalClock.getTime()
    thisExp.addData('time5', time5)

    #-------Start Routine "conf"-------
# checking to see if participants responded to the choice question
# if not,it skips this iteration of the confidence routine
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = ConfClock.getTime()
        frameN = frameN + 1

        # *CircleLeft* updates
        if t >= 0.0 and CircleLeft.status == NOT_STARTED:
            CircleLeft.tStart = t
            CircleLeft.frameNStart = frameN
            CircleLeft.setAutoDraw(True)

        # *CircleRight* updates
        if t >= 0.0 and CircleRight.status == NOT_STARTED:
            CircleRight.tStart = t
            CircleRight.frameNStart = frameN
            CircleRight.setAutoDraw(True)

        # *FixationCrossLeft* updates
        if t >= 0.0 and FixationCrossLeft.status == NOT_STARTED:
            FixationCrossLeft.tStart = t
            FixationCrossLeft.frameNStart = frameN
            FixationCrossLeft.setAutoDraw(True)

        # *FixationCrossRight* updates
        if t >= 0.0 and FixationCrossRight.status == NOT_STARTED:
            FixationCrossRight.tStart = t
            FixationCrossRight.frameNStart = frameN
            FixationCrossRight.setAutoDraw(True)

        # *FeedbackPlusLeft* updates
        if KeyRespCal.keys == 'left':
            if t >= 0.0 and FeedbackPlusLeft.status == NOT_STARTED:
                FeedbackPlusLeft.tStart = t
                FeedbackPlusLeft.frameNStart = frameN
                FeedbackPlusLeft.setAutoDraw(True)

        # *FeedbackPlusRight* updates
        if KeyRespCal.keys == 'right':
            if t >= 0.0 and FeedbackPlusRight.status == NOT_STARTED:
                FeedbackPlusRight.tStart = t
                FeedbackPlusRight.frameNStart = frameN
                FeedbackPlusRight.setAutoDraw(True)

        # *FeedbackCrossLeft* updates
        if KeyRespCal.keys == 'right':
            if t >= 0.0 and FeedbackCrossLeft.status == NOT_STARTED:
                FeedbackCrossLeft.tStart = t
                FeedbackCrossLeft.frameNStart = frameN
                FeedbackCrossLeft.setAutoDraw(True)

        # *FeedbackCrossRight* updates
        if KeyRespCal.keys == 'left':
            if t >= 0.0 and FeedbackCrossRight.status == NOT_STARTED:
                FeedbackCrossRight.tStart = t
                FeedbackCrossRight.frameNStart = frameN
                FeedbackCrossRight.setAutoDraw(True)

        # *confidence* updates
        if t > 0.2:
            Confidence1.draw()
            continueRoutine = Confidence1.noResponse
            if Confidence1.noResponse == False:
                Confidence1.response = Confidence1.getRating()
                Confidence1.rt = Confidence1.getRT()
            elif Confidence1.noResponse == True:
                if keyState[key.LEFT] == True and Confidence1.markerPlacedAt > 0.01:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt - 0.02
                    Confidence1.draw()
                elif keyState[key.LEFT] == True and Confidence1.markerPlacedAt == 0.01:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt - 0.01
                    Confidence1.draw()
                elif keyState[key.RIGHT] == True and Confidence1.markerPlacedAt < 0.99:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt + 0.02
                    Confidence1.draw()
                elif keyState[key.RIGHT] == True and Confidence1.markerPlacedAt == 0.99:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt + 0.01
                    Confidence1.draw()

        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=['escape']):
            core.quit()

        # refresh the screen
        if continueRoutine:
            win.flip()
        else:
            routineTimer.reset()

    #-------Ending Routine "conf"-------
    for thisComponent in ConfComponents:
        if hasattr(thisComponent, 'setAutoDraw'):
            thisComponent.setAutoDraw(False)

    points += 100*(1-(float(KeyRespCal.corr) - Confidence1.getRating())**2)

    # store data for binary (TrialHandler)
    thisExp.addData('Conf', Confidence1.getRating())
    thisExp.addData('Conf_RT', Confidence1.getRT())

    time6 = globalClock.getTime()
    thisExp.addData('time6', time6)
    
    #-------Prepare to start routine "CIP"------
    t = 0
    CipClock.reset()
    frameN = -1
    routineTimer.add(4.000000)
    now = []

    # Update Keyboard parameters
    event.clearEvents(eventType='keyboard')
    CipKey = event.BuilderKeyResponse()
    CipKey.status = NOT_STARTED
     # keep track of which components have finished
    CipComponents = []
    CipComponents.append(CircleLeft)
    CipComponents.append(CircleRight)
    CipComponents.append(Instr1Cip)
    CipComponents.append(DotPatchLeft)
    CipComponents.append(DotPatchRight)
    CipComponents.append(FeedbackPlusLeft)
    CipComponents.append(FeedbackPlusRight)
    CipComponents.append(FeedbackCrossLeft)
    CipComponents.append(FeedbackCrossRight)
    for thisComponent in CipComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    time7 = globalClock.getTime()
    thisExp.addData('time7', time7)
    
    #-------Start Routine "Cip"-------
    # checking to see if participants responded to the choice question
    # if not,it skips this iteration of the confidence routine
    continueRoutine = True
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = CipClock.getTime()
        frameN = frameN + 1

        # *CircleLeft* updates
        if t >= 0.0 and CircleLeft.status == NOT_STARTED:
            CircleLeft.tStart = t
            CircleLeft.frameNStart = frameN
            CircleLeft.setAutoDraw(True)

        # *CircleRight* updates
        if t >= 0.0 and CircleRight.status == NOT_STARTED:
            CircleRight.tStart = t
            CircleRight.frameNStart = frameN
            CircleRight.setAutoDraw(True)
            
        # *Instr1Cip* updates
        if t >= 0.0 and Instr1Cip.status == NOT_STARTED:
            Instr1Cip.tStart = t
            Instr1Cip.frameNStart = frameN
            Instr1Cip.setAutoDraw(True)

        # *FeedbackPlusLeft* updates
        if KeyRespCal.keys == 'left':
            if t >= 0.0 and FeedbackPlusLeft.status == NOT_STARTED:
                FeedbackPlusLeft.tStart = t
                FeedbackPlusLeft.frameNStart = frameN
                FeedbackPlusLeft.setAutoDraw(True)

        # *FeedbackPlusRight* updates
        if KeyRespCal.keys == 'right':
            if t >= 0.0 and FeedbackPlusRight.status == NOT_STARTED:
                FeedbackPlusRight.tStart = t
                FeedbackPlusRight.frameNStart = frameN
                FeedbackPlusRight.setAutoDraw(True)

        # *FeedbackCrossLeft* updates
        if KeyRespCal.keys == 'right':
            if t >= 0.0 and FeedbackCrossLeft.status == NOT_STARTED:
                FeedbackCrossLeft.tStart = t
                FeedbackCrossLeft.frameNStart = frameN
                FeedbackCrossLeft.setAutoDraw(True)

        # *FeedbackCrossRight* updates
        if KeyRespCal.keys == 'left':
            if t >= 0.0 and FeedbackCrossRight.status == NOT_STARTED:
                FeedbackCrossRight.tStart = t
                FeedbackCrossRight.frameNStart = frameN
                FeedbackCrossRight.setAutoDraw(True)

        # *CipKey* updates
        if t >= 0.0 and CipKey.status == NOT_STARTED:
            # keep track of start time/frame for later
            CipKey.tStart = t  # underestimates by a little under one frame
            CipKey.frameNStart = frameN  # exact frame index
            CipKey.status = STARTED
            # keyboard checking is just starting
            CipKey.clock.reset()  # now t=0
            event.clearEvents(eventType='keyboard')
        if CipKey.status == STARTED and t >= (0.0 + (4.0-win.monitorFramePeriod*0.75)): #most of one frame period left
            CipKey.status = STOPPED
        if CipKey.status == STARTED:
            theseKeys = event.getKeys(keyList=['left', 'right'])

            # check for quit:
            if 'escape' in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:
                now = theseKeys[-1]
                CipKey.keys.extend(theseKeys) # storing all keys
                CipKey.rt.append(CipKey.clock.getTime())
                theseKeys = event.clearEvents(eventType=None)

        # *DotPatchLeft* updates
        if t >= 0 and DotPatchLeft.status == NOT_STARTED and now == 'left':
            DotPatchLeft.tStart = t
            DotPatchLeft.frameNStart = frameN
            DotPatchLeft.setAutoDraw(True)
        if DotPatchLeft.status == STARTED and now == 'right':
            DotPatchLeft.setAutoDraw(False)
            DotPatchLeft.status = NOT_STARTED

        # *DotPatchRight* updates
        if t >= 0 and DotPatchRight.status == NOT_STARTED and now == 'right':
            DotPatchRight.tStart = t
            DotPatchRight.frameNStart = frameN
            DotPatchRight.setAutoDraw(True)
        if DotPatchRight.status == STARTED and now == 'left':
            DotPatchRight.setAutoDraw(False)
            DotPatchRight.status = NOT_STARTED

        # check if all components have finished
        if not continueRoutine:
            break
        continueRoutine = False
        for thisComponent in CipComponents:
            if hasattr(thisComponent, 'status') and thisComponent.status != FINISHED:
                continueRoutine = True
                break

         # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=['escape']):
            core.quit()

        # refresh the screen
        if continueRoutine:
            win.flip()
        else:
            routineTimer.reset()

     #-------Ending Routine "CIP"-------
    for thisComponent in CipComponents:
        if hasattr(thisComponent, 'setAutoDraw'):
            thisComponent.setAutoDraw(False)

    # check responses
    if CipKey.keys in ['', [], None]:
        CipKey.keys = None

    # Store data for experiment:
    thisExp.addData('CipResponse', CipKey.keys)
    if CipKey.keys != None:
        thisExp.addData('CipRT', CipKey.rt)

    time8 = globalClock.getTime()
    thisExp.addData('time8', time8)

    #------Prepare to start Routine "change"-------
    t = 0
    Choice2Clock.reset()
    frameN = -1

    # Update Keyboard parameters
    event.clearEvents(eventType='keyboard')
    KeyRespChange = event.BuilderKeyResponse()
    KeyRespChange.status = NOT_STARTED
     # keep track of which components have finished
    changeComponents = []
    changeComponents.append(CircleLeft)
    changeComponents.append(CircleRight)
    changeComponents.append(FixationCrossLeft)
    changeComponents.append(FixationCrossRight)
    changeComponents.append(FeedbackPlusLeft)
    changeComponents.append(FeedbackPlusRight)
    changeComponents.append(FeedbackCrossLeft)
    changeComponents.append(FeedbackCrossRight)
    changeComponents.append(DotCalibrationInstructions2)
    for thisComponent in changeComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    time9 = globalClock.getTime()
    thisExp.addData('time9', time9)
    
#-------Start Routine "change"-------
    continueRoutine = True
    while continueRoutine:
        t = Choice2Clock.getTime()
        frameN = frameN + 1

        # *CircleLeft* updates
        if t >= 0 and CircleLeft.status == NOT_STARTED:
            CircleLeft.tStart = t
            CircleLeft.frameNStart = frameN
            CircleLeft.setAutoDraw(True)


        # *CircleRight* updates
        if t >= 0 and CircleRight.status == NOT_STARTED:
            CircleRight.tStart = t
            CircleRight.frameNStart = frameN
            CircleRight.setAutoDraw(True)

        # *FixationCrossLeft* updates
        if t >= 0 and FixationCrossLeft.status == NOT_STARTED:
            FixationCrossLeft.tStart = t
            FixationCrossLeft.frameNStart = frameN
            FixationCrossLeft.setAutoDraw(True)

        # *FixationCrossRight* updates
        if t >= 0 and FixationCrossRight.status == NOT_STARTED:
            FixationCrossRight.tStart = t
            FixationCrossRight.frameNStart = frameN
            FixationCrossRight.setAutoDraw(True)
            

        # *DotExampleInstructions2* updates
        if t >= 0 and DotCalibrationInstructions2.status == NOT_STARTED:
            DotCalibrationInstructions2.tStart = t
            DotCalibrationInstructions2.frameNStart = frameN
            DotCalibrationInstructions2.setAutoDraw(True)

        # *key_resp_choice* updates
        if t >= 0.7 and KeyRespChange.status == NOT_STARTED:
            theseKeys = event.clearEvents(eventType=None)
            KeyRespChange.tStart = t
            KeyRespChange.frameNStart = frameN
            KeyRespChange.status = STARTED
            KeyRespChange.clock.reset()
        if KeyRespChange.status == STARTED:
            theseKeys = event.getKeys(keyList=['left', 'right'])

            # check for quit:
            if 'escape' in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:
                KeyRespChange.keys = theseKeys[-1]
                KeyRespChange.rt = KeyRespChange.clock.getTime()
                continueRoutine = False

        # check if all components have finished
        if not continueRoutine:
            routineTimer.reset()
            break
        continueRoutine = False
        for thisComponent in changeComponents:
            if hasattr(thisComponent, 'status') and thisComponent.status != FINISHED:
                continueRoutine = True
                break

         # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=['escape']):
            core.quit()

        # refresh the screen
        if continueRoutine:
            win.flip()
        else:
            routineTimer.reset()

     #-------Ending Routine "choice"-------
    for thisComponent in changeComponents:
        if hasattr(thisComponent, 'setAutoDraw'):
            thisComponent.setAutoDraw(False)
    # check responses
    if KeyRespChange.keys in ['', [], None]:
        KeyRespChange.keys = None
        # was no response the correct answer?!
    if KeyRespChange.keys == CorrectKey:
        KeyRespChange.corr = 1  # correct non-response
    else: KeyRespChange.corr = 0  # failed to respond (incorrectly)

    # Store data for experiment:
    thisExp.addData('ChangeCorrect', KeyRespChange.corr)
    thisExp.addData('ChangeResponse', KeyRespChange.keys)
    if KeyRespChange.keys != None:
        thisExp.addData('ChangeRT', KeyRespChange.rt)

    time10 = globalClock.getTime()
    thisExp.addData('time10', time10)

      #------Prepare to start Routine "conf2"-------
    t = 0
    Conf2Clock.reset()
    frameN = -1
    # update component parameters for each repeat
    Confidence1.reset()
    # keep track of which components have finished
    Conf2Components = []
    Conf2Components.append(CircleLeft)
    Conf2Components.append(CircleRight)
    Conf2Components.append(FixationCrossLeft)
    Conf2Components.append(FixationCrossRight)
    Conf2Components.append(FeedbackPlusLeft)
    Conf2Components.append(FeedbackPlusRight)
    Conf2Components.append(FeedbackCrossLeft)
    Conf2Components.append(FeedbackCrossRight)
    Conf2Components.append(Confidence1)
    for thisComponent in Conf2Components:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED

    time11 = globalClock.getTime()
    thisExp.addData('time11', time11)
    
    #-------Start Routine "conf"-------
# checking to see if participants responded to the choice question
# if not,it skips this iteration of the confidence routine
    continueRoutine = True
    while continueRoutine:
        # get current time
        t = Conf2Clock.getTime()
        frameN = frameN + 1

        # *CircleLeft* updates
        if t >= 0.0 and CircleLeft.status == NOT_STARTED:
            CircleLeft.tStart = t
            CircleLeft.frameNStart = frameN
            CircleLeft.setAutoDraw(True)

        # *CircleRight* updates
        if t >= 0.0 and CircleRight.status == NOT_STARTED:
            CircleRight.tStart = t
            CircleRight.frameNStart = frameN
            CircleRight.setAutoDraw(True)

        # *FixationCrossLeft* updates
        if t >= 0.0 and FixationCrossLeft.status == NOT_STARTED:
            FixationCrossLeft.tStart = t
            FixationCrossLeft.frameNStart = frameN
            FixationCrossLeft.setAutoDraw(True)

        # *FixationCrossRight* updates
        if t >= 0.0 and FixationCrossRight.status == NOT_STARTED:
            FixationCrossRight.tStart = t
            FixationCrossRight.frameNStart = frameN
            FixationCrossRight.setAutoDraw(True)
            
        # *FeedbackPlusLeft* updates
        if KeyRespChange.keys == 'left':
            if t >= 0.0 and FeedbackPlusLeft.status == NOT_STARTED:
                FeedbackPlusLeft.tStart = t
                FeedbackPlusLeft.frameNStart = frameN
                FeedbackPlusLeft.setAutoDraw(True)

        # *FeedbackPlusRight* updates
        if KeyRespChange.keys == 'right':
            if t >= 0.0 and FeedbackPlusRight.status == NOT_STARTED:
                FeedbackPlusRight.tStart = t
                FeedbackPlusRight.frameNStart = frameN
                FeedbackPlusRight.setAutoDraw(True)

        # *FeedbackCrossLeft* updates
        if KeyRespChange.keys == 'right':
            if t >= 0.0 and FeedbackCrossLeft.status == NOT_STARTED:
                FeedbackCrossLeft.tStart = t
                FeedbackCrossLeft.frameNStart = frameN
                FeedbackCrossLeft.setAutoDraw(True)

        # *FeedbackCrossRight* updates
        if KeyRespChange.keys == 'left':
            if t >= 0.0 and FeedbackCrossRight.status == NOT_STARTED:
                FeedbackCrossRight.tStart = t
                FeedbackCrossRight.frameNStart = frameN
                FeedbackCrossRight.setAutoDraw(True)

        # *confidence* updates
        if t > 0.2:
            Confidence1.draw()
            continueRoutine = Confidence1.noResponse
            if Confidence1.noResponse == False:
                Confidence1.response = Confidence1.getRating()
                Confidence1.rt = Confidence1.getRT()
            elif Confidence1.noResponse == True:
                if keyState[key.LEFT] == True and Confidence1.markerPlacedAt > 0.01:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt - 0.02
                    Confidence1.draw()
                elif keyState[key.LEFT] == True and Confidence1.markerPlacedAt == 0.01:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt - 0.01
                    Confidence1.draw()
                elif keyState[key.RIGHT] == True and Confidence1.markerPlacedAt < 0.99:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt + 0.02
                    Confidence1.draw()
                elif keyState[key.RIGHT] == True and Confidence1.markerPlacedAt == 0.99:
                    Confidence1.markerPlacedAt = Confidence1.markerPlacedAt + 0.01
                    Confidence1.draw()

        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=['escape']):
            core.quit()

        # refresh the screen
        if continueRoutine:
            win.flip()
        else:
            routineTimer.reset()

    #-------Ending Routine "conf"-------
    for thisComponent in Conf2Components:
        if hasattr(thisComponent, 'setAutoDraw'):
            thisComponent.setAutoDraw(False)

    points += 100*(1-(float(KeyRespChange.corr) - Confidence1.getRating())**2)

    # store data for binary (TrialHandler)
    thisExp.addData('Conf2', Confidence1.getRating())
    thisExp.addData('Conf2_RT', Confidence1.getRT())
    
    time12 = globalClock.getTime()
    thisExp.addData('time12', time12)
    
    thisExp.nextEntry()

    # If the trial number is divisible by 25, display the rest prompt
    if RestCounter % 25 == 0 and RestCounter != 100 and RestCounter != 200:
        
        #------Prepare to start Routine "Rest"-------
        t = 0
        RestClock.reset()
        frameN = -1
        
        RestText.setText(text=u'Great! \nYou have earned {0} points so far! Now take a rest and press spacebar when you are ready to begin the next block.'.format(int(points)))
        # update component parameters for each repeat
        RestResponse = event.BuilderKeyResponse()
        RestResponse.status = NOT_STARTED
        # keep track of which components have finished
        RestComponents = []
        RestComponents.append(RestText)
        RestComponents.append(RestResponse)
        for thisComponent in RestComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        
        #-------Start Routine "Rest"-------
        continueRoutine = True
        while continueRoutine:
            t = RestClock.getTime()
            frameN = frameN + 1
            
            # *RestText* updates
            if t >= 0.0 and RestText.status == NOT_STARTED:
                RestText.tStart = t
                RestText.frameNStart = frameN
                RestText.setAutoDraw(True)
            
            # *RestResponse* updates
            if t >= 2.0 and RestResponse.status == NOT_STARTED:
                RestResponse.tStart = t
                RestResponse.frameNStart = frameN
                RestResponse.status = STARTED
                RestResponse.clock.reset()
                event.clearEvents(eventType='keyboard')
            if RestResponse.status == STARTED:
                theseKeys = event.getKeys(keyList=['space'])
                
                # check for quit:
                if 'escape' in theseKeys:
                    endExpNow = True
                if len(theseKeys) > 0:
                    RestResponse.keys = theseKeys[-1]
                    RestResponse.rt = RestResponse.clock.getTime()
                    continueRoutine = False
            
            # check if all components have finished
            if not continueRoutine:
                routineTimer.reset()
                break
            continueRoutine = False
            for thisComponent in RestComponents:
                if hasattr(thisComponent, 'status') and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break

            # check for quit (the Esc key)
            if endExpNow or event.getKeys(keyList=['escape']):
                core.quit()
            
            # refresh the screen
            if continueRoutine:
                win.flip()
            else:
                routineTimer.reset()

        #-------Ending Routine "Rest"-------
        for thisComponent in RestComponents:
            if hasattr(thisComponent, 'setAutoDraw'):
                thisComponent.setAutoDraw(False)
        # check responses
        if RestResponse.keys in ['', [], None]:
            RestResponse.keys = None
        # store data for MainStaircase (ExperimentHandler)
        if RestResponse.keys != None:
            thisExp.addData('RestTime', RestResponse.rt)  
            thisExp.nextEntry()

    if RestCounter == 100:

        #------Prepare to start Routine "Break"-------
        t = 0
        BreakClock.reset()
        frameN = -1
        
        BreakText.setText(text=u'Great! \nYou have earned {0} points so far! You are now halfway through the task. \n\nPlease contact the experimenter.'.format(int(points)))
        
        # update component parameters for each repeat
        BreakResponse = event.BuilderKeyResponse()
        BreakResponse.status = NOT_STARTED
        # keep track of which components have finished
        BreakComponents = []
        BreakComponents.append(BreakText)
        BreakComponents.append(BreakResponse)
        for thisComponent in BreakComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED

        #-------Start Routine "Break"-------
        continueRoutine = True
        while continueRoutine:
            t = BreakClock.getTime()
            frameN = frameN + 1

            # *BreakText* updates
            if t >= 0.0 and BreakText.status == NOT_STARTED:
                BreakText.tStart = t
                BreakText.frameNStart = frameN
                BreakText.setAutoDraw(True)

            # *BreakResponse* updates
            if t >= 2.0 and BreakResponse.status == NOT_STARTED:
                BreakResponse.tStart = t
                BreakResponse.frameNStart = frameN
                BreakResponse.status = STARTED
                BreakResponse.clock.reset()
                event.clearEvents(eventType='keyboard')
            if BreakResponse.status == STARTED:
                theseKeys = event.getKeys(keyList=['p'])

                # check for quit:
                if 'escape' in theseKeys:
                    endExpNow = True
                if len(theseKeys) > 0:
                    BreakResponse.keys = theseKeys[-1]
                    BreakResponse.rt = BreakResponse.clock.getTime()
                    continueRoutine = False

            # check if all components have finished
            if not continueRoutine:
                routineTimer.reset()
                break
            continueRoutine = False
            for thisComponent in BreakComponents:
                if hasattr(thisComponent, 'status') and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break

            # check for quit (the Esc key)
            if endExpNow or event.getKeys(keyList=['escape']):
                core.quit()

            # refresh the screen
            if continueRoutine:
                win.flip()
            else:
                routineTimer.reset()

        #-------Ending Routine "Break"-------
        for thisComponent in BreakComponents:
            if hasattr(thisComponent, 'setAutoDraw'):
                thisComponent.setAutoDraw(False)
        # check responses
        if BreakResponse.keys in ['', [], None]:
            BreakResponse.keys = None
        # store data for MainStaircase (ExperimentHandler)
        if BreakResponse.keys != None:
            thisExp.addData('RestTime', RestResponse.rt)
            thisExp.nextEntry()


#################################### End of main staircase #####################################

#------Prepare to start Routine "Thank You"-------
t = 0
ThankYouClock.reset()  # clock 
frameN = -1

earned = points//12000
ThankYouText.setText(text=u'You have now completed this experiment. \nYou have earned \u00A3{0} in addition to your show-up fee! Thank you for your participation. Please inform the experimenter that you have finished.'.format(earned))

# update component parameters for each repeat
ThankYouResponse = event.BuilderKeyResponse()  # create an object of type KeyResponse
ThankYouResponse.status = NOT_STARTED
# keep track of which components have finished
ThankYouComponents = []
ThankYouComponents.append(Instructions3Text)
ThankYouComponents.append(Instructions3Response)
for thisComponent in ThankYouComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

#-------Start Routine "ThankYou"-------
continueRoutine = True
while continueRoutine:
    # get current time
    t = ThankYouClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *ThankYouText* updates
    if t >= 0.0 and ThankYouText.status == NOT_STARTED:
        # keep track of start time/frame for later
        ThankYouText.tStart = t  # underestimates by a little under one frame
        ThankYouText.frameNStart = frameN  # exact frame index
        ThankYouText.setAutoDraw(True)
    
    # *ThankYouResponse* updates
    if t >= 0 and ThankYouResponse.status == NOT_STARTED:
        # keep track of start time/frame for later
        ThankYouResponse.tStart = t  # underestimates by a little under one frame
        ThankYouResponse.frameNStart = frameN  # exact frame index
        ThankYouResponse.status = STARTED
        # keyboard checking is just starting
        ThankYouResponse.clock.reset()  # now t=0
        event.clearEvents(eventType='keyboard')
    if ThankYouResponse.status == STARTED:
        theseKeys = event.getKeys(keyList=['space'])
        
        # check for quit:
        if "escape" in theseKeys:
            endExpNow = True
        if len(theseKeys) > 0:  # at least one key was pressed
            ThankYouResponse.keys = theseKeys[-1]  # just the last key pressed
            ThankYouResponse.rt = ThankYouResponse.clock.getTime()
            # a response ends the routine
            continueRoutine = False
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineTimer.reset()  # if we abort early the non-slip timer needs reset
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in ThankYouComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()
    else:  # this Routine was not non-slip safe so reset non-slip timer
        routineTimer.reset()

#-------Ending Routine "Instructions4"-------
for thisComponent in ThankYouComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if ThankYouResponse.keys in ['', [], None]:  # No response was made
   ThankYouResponse.keys=None
# store data for thisExp (ExperimentHandler)
thisExp.addData('ThankYouResponse.keys',ThankYouResponse.keys)
if ThankYouResponse.keys != None:  # we had a response
    thisExp.addData('ThankYouResponse.rt', ThankYouResponse.rt)
thisExp.nextEntry()
win.close()

print float(earned)

core.quit()