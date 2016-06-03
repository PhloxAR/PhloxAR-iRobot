# -*- coding: utf-8 -*-

import cv2
from threading import Thread


class Webcam(object):

    def __init__(self):
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
        self.current_frame = self.video_capture.read()[1]

    def start(self):
        Thread(target=self._update_frame, args=()).start()

    def _update_frame(self):
        while True:
            self.current_frame = self.video_capture.read()[1]

    def get_current_frame(self):
        return self.current_frame
