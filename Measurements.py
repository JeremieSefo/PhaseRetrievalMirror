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
        self.noise = (1. + 0.j) * np.abs( np.random.normal(loc = 0, scale= 1, size = (self.z).shape)) #scale = self.noise_lvl
        self.noise *= self.noise_lvl/np.linalg.norm(self.noise)
        self.synt_meas = self.z * np.conjugate(self.z) + (1. + 0.j) * self.noise
        #print("self.z", self.z)
        return self.synt_meas, self.noise

class poiss_noisy_meas(noisy_meas):
    def __init__(self, x_true, noise_lvl, model):
        super().__init__(x_true, noise_lvl, model)
    def __call__(self,):
        self.z = self.model(self.x_true)
        self.noise = (1. + 0.j) * np.random.poisson(lam = 1, size = (self.z).shape)
        #self.noise *= (self.noise_lvl)/np.linalg.norm(self.noise)
        self.noise *= (1.)/np.linalg.norm(self.noise)
        
        self.noise_strength =  (self.noise_lvl**(0.5)) * self.z * np.conjugate(self.z)
        self.noise *= (self.noise_strength)

        self.synt_meas = self.z * np.conjugate(self.z) + (1. + 0.j) * self.noise 
        return self.synt_meas, self.noise

    
