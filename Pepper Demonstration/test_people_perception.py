#!/usr/bin/env python

# Add time they took to reply/ choose - python time
# Use built-in tablet? maybe not!
# Always show the last thing the robot said at the bottom

from qimate import QiMate
import numpy as np
import magic_tablet
import time
import cv2
import os
import requests

ipaddress = '192.168.1.41'
s = QiMate(ipaddress, connect_on_init=True)
time.sleep(1) # give connection time to finish before initializing tablet

gesture = s.ALBehaviorManager
faceTracker = s.ALTracker
tablet = magic_tablet.connect(s.session)
speech = s.ALTextToSpeech

faceTracker.unregisterAllTargets()
faceTracker.setMode('Head') # track human faces with the robot's head
faceTracker.registerTarget('Face', 0.2) # width of face = 0.2 meters
faceTracker.track('Face')

videoDevice = s.ALVideoDevice
AL_kTopCamera = 0
AL_kQVGA = 1
AL_kBGRColorSpace = 13
cap_id = videoDevice.subscribeCamera("cap", AL_kTopCamera, AL_kQVGA, AL_kBGRColorSpace, 10)

def show_and_tell(text, animation=None, clear_animation=False, clear_text=False, clear=False):
    tablet.update(animation=animation, top=text)
    speech.say(text) # blocks until speech is finished
    if clear or clear_animation:
        tablet.update(animation=magic_tablet.BLANK)
    if clear or clear_text:
        tablet.update(top=None)

ask_choice_options = []
ask_choice_answer = None
def ask_choice_callback(index):
    global ask_choice_answer
    ask_choice_answer = ask_choice_options[index]

def ask_choice(text, options):
    global ask_choice_answer
    global ask_choice_options
    ask_choice_answer = None
    ask_choice_options = options
    tablet.show(top=text)
    tablet.choice(options, callback=ask_choice_callback)
    speech.say(text)
    while ask_choice_answer is None:
        time.sleep(0.1)
    tablet.buttons(magic_tablet.BLANK)
    tablet.show(magic_tablet.TICK, top=magic_tablet.BLANK)
    return ask_choice_answer

ask_number_answer = None
def ask_number_callback(value):
    global ask_number_answer
    ask_number_answer = value

def ask_number(text):
    global ask_number_answer
    ask_number_answer = None
    tablet.show(top=text)
    tablet.prompt(keyboard="numeric", callback=ask_number_callback)
    speech.say(text)
    while ask_number_answer is None:
        time.sleep(0.1)
    tablet.buttons(magic_tablet.BLANK)
    tablet.show(magic_tablet.TICK, top=magic_tablet.BLANK)
    return int(ask_number_answer)

def make_numpy_array_from_video(result):
    # Convert image to numpy array
    values = list(result[6])
    i = 0
    width, height = 320, 240
    image = np.zeros((height, width, 3), np.uint8)
    for y in range(0, height):
        for x in range(0, width):
            image.itemset((y, x, 0), values[i + 0])
            image.itemset((y, x, 1), values[i + 1])
            image.itemset((y, x, 2), values[i + 2])
            i += 3

yes_no_choices = ['Yes','No']
gender_choices = ['Female','Male','Other']



gesture.runBehavior('System/animations/Stand/Gestures/Hey_1')
show_and_tell("Hi! I'm Pepper!",)
show_and_tell("I'm still learning and I need your help")

consent_age = ask_choice("Can I guess your age?", yes_no_choices)
if consent_age == 'Yes':
    # Tablet: doubting emoji
    speech.say("Mmm...")
    result = videoDevice.getImageRemote(cap_id)

    if result is None:
        print 'cannot capture.'
    elif result[6] is None:
        print 'no image data string.'
    else:
        image = make_numpy_array_from_video(result)

        cv2.imwrite("predict.jpg", image)
        imageData = open("predict.jpg", "rb").read()
        payload = {"image": imageData}
        r = requests.post("http://localhost:4000/predict", files=payload).json()
        guessed_gender = "Female" if r["prediction"]["male"] <= r["prediction"]["female"] else "Male"
        guessed_age = 27 # TODO

        show_and_tell("Are you a {guessed_age} year old {guessed_gender}?".format(**locals()))

        actual_age = ask_number("What is your actual age?")

        if guessed_age == actual_age:
            show_and_tell("Yay! I guessed correctly", animation=magic_tablet.TICK, clear_animation=True)
        elif abs(guessed_age - actual_age) < 5:
            show_and_tell("I was close!", animation=magic_tablet.TICK, clear_animation=True)
        else:
            show_and_tell("Bummer! Sorry, I'm still learning", animation=magic_tablet.ERROR, clear_animation=True)

        actual_gender = ask_choice("Please select your gender:", gender_choices)

else:
    actual_gender = ask_choice("Please select your gender:", gender_choices)
    time.sleep(10)

consent_info = ask_choice("May I use your photo to train myself to recognise people better?", yes_no_choices)
if consent_info == 'Yes':
    # Tablet: shows user image w/ face detection square > takes photo
    gesture.runBehavior('System/animations/Stand/Gestures/Hey_1')
    show_and_tell("Thank you very much!")
    show_and_tell("Have a good day!", clear=True)
else:
    # Tablet: crying emoji
    gesture.runBehavior('System/animations/Stand/Gestures/Hey_1')
    show_and_tell("Oh! Mmm, okay...")
    show_and_tell("Hope you have a good day anyway!", clear=True)

videoDevice.unsubscribe(cap_id)
faceTracker.unregisterTarget('Face')
