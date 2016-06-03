from __future__ import unicode_literals

from threading import Thread


class Feature:
  
    # initialize features
    def __init__(self):
        self.thread = None
        self.is_stop = False

    # start thread
    def start(self, args=None):
        self.is_stop = False
        
        if self.thread and self.thread.is_alive(): return

        self.thread = Thread(target=self._thread, args=(args,))
        self.thread.start()

    # stop thread
    def stop(self):
        self.is_stop = True