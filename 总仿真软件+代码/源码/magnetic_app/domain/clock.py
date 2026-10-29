import time


class PlaybackClock:
    def __init__(self, duration, now=time.perf_counter):
        self.duration = duration
        self.now = now
        self.position = 0.
        self.epoch = now()
        self.speed = 5.
        self.playing = False
        self.was_playing = False
        self.seeking = False

    def current(self):
        if self.playing:
            return min(self.duration, max(0., self.position+(self.now()-self.epoch)*self.speed))
        return self.position

    def seek(self, value):
        self.position = min(self.duration, max(0., value))
        self.epoch = self.now()

    def set_playing(self, playing):
        self.seek(self.current())
        self.playing = playing

    def set_speed(self, speed):
        self.seek(self.current())
        self.speed = speed

    def begin_seek(self):
        self.was_playing = self.playing
        self.set_playing(False)
        self.seeking = True

    def end_seek(self):
        self.seeking = False
        self.set_playing(self.was_playing)
