
import time

class Watch(object):

    def __init__(self, msg='Time elapsed'):
        self.msg = msg

    def __enter__(self):
        self._start = time.time()

    def __exit__(self, *args, **kwargs):
        elapsed = (time.time() - self._start) * 1000
        print('%s: %s miliseconds' % (self.msg, elapsed))
