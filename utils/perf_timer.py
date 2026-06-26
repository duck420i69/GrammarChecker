from time import perf_counter

class PerfTimer:
    def __init__(self, func=None):
        self.start = perf_counter()
        self.func = func

    def start(self, *args, **kwargs):
        if self.func is not None:
            self.func(*args, **kwargs)
        self.start = perf_counter()

    def get_time(self):
        return perf_counter() - self.start