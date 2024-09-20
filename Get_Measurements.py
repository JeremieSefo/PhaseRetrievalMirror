import numpy as np

def operator(A, x):
        z = A @  np.conjugate(x)
        return z * np.conjugate(z)

class get_Measurements:

    def __init__(self, A, x, noise):
        self.A = A
        self.x = x
        self.noise = noise
    
    def magnitudes(self):
        return operator(self.A, self.x) + self.noise