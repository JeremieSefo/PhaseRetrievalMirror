import numpy as np
import SensingMatrix as sm

class noisy_meas:
    def __init__(self, x_true, noise_lvl, model):
        self.x_true =  x_true
        self.noise_lvl = noise_lvl
        self.model = model

class gauss_noisy_meas(noisy_meas):
    def __init__(self, x_true, noise_lvl, model):
        super().__init__(x_true, noise_lvl, model)
    
    def __call__(self,):
        self.z = self.model(self.x_true)
        #print('we are in')
        self.noise = (1. + 0.j) * np.abs( np.random.normal(loc = 0, scale= self.noise_lvl, size = (self.z).shape))
        self.synt_meas = self.z * np.conjugate(self.z) + self.noise
        return self.synt_meas, self.noise

class poiss_noisy_meas(noisy_meas):
    def __init__(self, x_true, noise_lvl, model):
        super().__init__(x_true, noise_lvl, model)
    def __call__(self,):
        self.z = self.model(self.x_true)
        self.noise = np.random.poisson(lam = 0, size = (self.z).shape)
        self.noise *= self.noise_lvl/np.linalg.norm(self.noise)
        self.synt_meas = self.z * np.conjugate(self.z) + (1. + 0.j) * self.noise 
        return self.synt_meas, self.noise

    
