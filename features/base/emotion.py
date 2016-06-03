from __future__ import unicode_literals

from time import sleep


class Emotion:

    # initialize emotion
    def __init__(self):
        self.emotion = None

    # display emotion
    def _display_emotion(self, emotion):
        self.emotion = emotion
        sleep(2)
        self.emotion = None