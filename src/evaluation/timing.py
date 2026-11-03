from time import perf_counter

class Timer:

    def __init__(self):

        self.start = None

    def begin(self):

        self.start = perf_counter()

    def stop(self):

        return (
            perf_counter()
            - self.start
        )