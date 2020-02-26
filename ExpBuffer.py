import numpy as np
import random

# stores tuples, which can be unzipped into individual numpy arrays of samples
class ExpBuffer:
    def __init__(self, maxBufferSize):
        self.BUFFER_LEN = maxBufferSize

        self.buffer = []
        self.count = 0

    def add(self, element):
        element = np.asarray(element)
        receivedShape = element.shape
        
        # initialize buffer if needed
        if (self.count == 0):
            self.elementShape = receivedShape

        else:
            if self.elementShape != receivedShape:
                print("Incorrect shape for buffer. Should be ", self.elementShape, ", received ", receivedShape, ".")
                return

        if (self.count < self.BUFFER_LEN):
            self.buffer.append(element)
            self.count += 1

        else:
            self.buffer.pop(0)
            self.buffer.append(element)

    def clear(self):
        self.buffer = []
        self.count = 0
        self.elementShape = None

    # samples sampleSize tuples randomly, then unzips and returns individual variable lists
    # returns the lists as multiple returns
    def sample(self, sampleSize):
        if count > sampleSize:
            sample = random.sample(self.buffer, sampleSize)
        else:
            sample = random.sample(self.buffer, self.count)

        result = list(zip(*self.buffer))
        return *result
        
