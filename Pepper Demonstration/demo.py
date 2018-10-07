# Add time they took to reply/ choose - python time
# Use built-in tablet? maybe not!
# Always show the last thing the robot said at the bottom

from qimate import QiMate
import magic_tablet
import time

ipaddress = '192.168.1.21'
s = QiMate(ipaddress, connect_on_init=True)
tablet = magic_tablet.connect(s.session)
# tablet.start()
speech = s.ALTextToSpeech
gesture = s.ALBehaviorManager

actual_age = 21
guessed_age = 27
guessed_gender = 'M'
yes_no_choices = ['Yes','No']
gender_choices = ['Female','Male','Other']
consent_age = None
consent_info = None
actual_gender = None

# Call-back function for when a click occurs
def consent_age_click(index):
    global consent_age
    consent_age = yes_no_choices[index]
    print("selection: "+consent_age)
    tablet.buttons(magic_tablet.BLANK)
    tablet.show(animation=magic_tablet.BLANK, status_timeout=2.0)
    time.sleep(2)

def consent_info_click(index):
    global consent_info
    consent_info = yes_no_choices[index]
    print("selection_2: "+consent_info)
    tablet.buttons(magic_tablet.BLANK)
    tablet.show(animation=magic_tablet.BLANK, status_timeout=2.0)
    time.sleep(2)

def actual_gender_click(index):
    global actual_gender
    actual_gender = gender_choices[index]
    print("actual_gender: "+actual_gender)
    speech.say('Thanks!')
    tablet.buttons(magic_tablet.BLANK)
    tablet.show(animation=magic_tablet.BLANK, top="Thanks!", status_timeout=2.0)
    time.sleep(2)
    
tablet.show(animation=magic_tablet.BLANK, top="Hi! I'm Pepper")
gesture.runBehavior('System/animations/Stand/Gestures/Hey_1')
speech.say("Hi! I'm Pepper", _async=True)
speech.say("I'm still learning and I need your help", _async=True)

tablet.show(animation=magic_tablet.BLANK, top="Can I guess your age?")
speech.say("Can I guess your age?")
tablet.choice(yes_no_choices, consent_age_click)
while consent_age == None:
    time.sleep(1)
if consent_age == 'Yes':
    # Tablet: doubting emoji
    speech.say("Mmm...")
    tablet.show(animation=magic_tablet.BLANK, top=str(guessed_age)+"?") # Store guessed_age and guessed_gender
    speech.say(str(guessed_age)+"?")
    # Tablet: text box. ToDo: save as actual_age
    speech.say("What is your actual age?")
    if guessed_age == actual_age:
        speech.say("Yay!", _async=True)
    elif abs(guessed_age - actual_age) < 5:
        speech.say("Close!", _async=True)
    else:
        speech.say("Bummer! Sorry, I'm still learning")

    tablet.show(animation=magic_tablet.BLANK, top="Please select your gender:")
    speech.say("Please select your gender:")
    # Register three big buttons, with a callback
    tablet.choice(gender_choices, actual_gender_click)
    while actual_gender == None:
        time.sleep(1)

else:
    tablet.show(animation=magic_tablet.BLANK, top="Please select your gender:")
    speech.say("Please select your gender:")
    tablet.choice(gender_choices, actual_gender_click)
    time.sleep(10)

tablet.show(animation=magic_tablet.BLANK, top="May I use your photo to train myself to recognise people better?")
speech.say("May I use your photo to train myself to recognise people better?")
tablet.choice(yes_no_choices, consent_info_click)
while consent_info == None:
    time.sleep(1)
if consent_info == 'Yes':
    # Tablet: shows user image w/ face detection square > takes photo
    speech.say("Thank you very much!", _async=True)
    gesture.runBehavior('System/animations/Stand/Gestures/Hey_1')
    speech.say("Have a good day!", _async=True)
else:
    # Tablet: crying emoji
    speech.say("Oh! Mmm, okay...", _async=True)
    gesture.runBehavior('System/animations/Stand/Gestures/Hey_1')
    speech.say("Hope you have a good day anyway!", _async=True)