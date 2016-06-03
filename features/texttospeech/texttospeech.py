from __future__ import unicode_literals

import pyttsx


class TextToSpeech:

    def __init__(self):
        self.pyttsx = pyttsx.init()
 
    # convert text to speech
    def convert(self, text):
        print(text)

        try:
            self.pyttsx.say(text)
            self.pyttsx.runAndWait()
        except RuntimeError:
            print("Could not convert text to speech")