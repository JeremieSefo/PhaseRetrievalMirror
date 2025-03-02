import numpy as np
import SensingMatrix as sm
def generate_multivariate_gaussian(mus, sigma, size=1):
    """
    Generate an instance of a multivariate Poisson distribution.

    Parameters:
    mus (list or array): Means of the individual Poisson distributions.
    size (int): Number of samples to generate.

    Returns:
    np.ndarray: An array of shape (size, len(mus)) representing the samples.
    """
    # Ensure lambdas is a numpy array
    mus = np.array(mus)
    # Generate samples for each Poisson distribution
    samples = ([np.abs(np.random.normal(mu, sigma, size)) for mu in mus]) #[np.random.poisson(lam, size) for lam in lambdas]
    
    # Combine samples into a multivariate array
    multivariate_samples = np.stack(samples, axis=-1)
    return multivariate_samples[0]

def generate_multivariate_poisson(lambdas, size=1):
    """
    Generate an instance of a multivariate Poisson distribution.

    Parameters:
    lambdas (list or array): Means of the individual Poisson distributions.
    size (int): Number of samples to generate.

    Returns:
    np.ndarray: An array of shape (size, len(lambdas)) representing the samples.
    """
    # Ensure lambdas is a numpy array
    lambdas = np.array(lambdas)
    
    # Generate samples for each Poisson distribution
    samples = [np.abs(np.random.poisson(lam, size)) for lam in lambdas]
    
    # Combine samples into a multivariate array
    multivariate_samples = np.stack(samples, axis=-1)
    
    return multivariate_samples[0]


class noisy_meas:
    def __init__(self, x_true, noise_lvl, model):
        self.x_true =  x_true
        self.noise_lvl = noise_lvl
        self.model = model

class gauss_noisy_meas(noisy_meas):
    def __init__(self, x_true, noise_lvl, model):
        super().__init__(x_true, noise_lvl, model)
    
    def __call__(self,):
        self.z = self.model(self.x_true).flatten()
        mus = np.abs(self.z)**2  # Means for the Poisson distributions
        num_samples = 1  # Number of samples to generate

        self.synt_meas = (1. + 0.j) *  generate_multivariate_gaussian(mus, self.noise_lvl, num_samples) #(1/self.noise_lvl) *
        self.noise = self.synt_meas - np.abs(self.z)**2

        # #print('we are in')
        # self.noise = (1. + 0.j) * np.abs( np.random.normal(loc = 0, scale= 1, size = (self.z).shape)) # random values in noise (generated here) are essentially ones. #scale = self.noise_lvl 
        # #self.noise *= self.noise_lvl/np.linalg.norm(self.noise)
        # #self.noise *= (1.)/np.linalg.norm(self.noise)
        
        # self.noise_strength =  (self.noise_lvl**(1)) * np.abs(self.z)**2 # np.linalg.norm(np.abs(self.z)**2)
        # self.noise *= (self.noise_strength)

        # self.synt_meas = np.abs(self.z)**2 + (1. + 0.j) * self.noise 

        self.NSR = np.linalg.norm(self.noise)**2 / ( 4 * np.linalg.norm(np.abs(self.z)**2)**2 )
        return self.synt_meas, self.noise, self.NSR

class poiss_noisy_meas(noisy_meas):
    def __init__(self, x_true, noise_lvl, model):
        super().__init__(x_true, noise_lvl, model)
    def __call__(self,):
        self.z = self.model(self.x_true).flatten()
        lambdas = self.noise_lvl * np.abs(self.z)**2  # Means for the Poisson distributions
        num_samples = 1  # Number of samples to generate

        self.synt_meas = (1. + 0.j) *  generate_multivariate_poisson(lambdas, num_samples) * (1/self.noise_lvl) 
        self.noise = self.synt_meas - np.abs(self.z)**2
        # self.noise = (1. + 0.j) * np.random.poisson(lam = 1, size = (self.z).shape)
        # #self.noise *= (self.noise_lvl)/np.linalg.norm(self.noise)
        # #self.noise *= (1.)/np.linalg.norm(self.noise)
        
        # self.noise_strength =  (self.noise_lvl**(1)) * np.abs(self.z)**2 # np.linalg.norm(np.abs(self.z)**2)
        # self.noise *= (self.noise_strength)

        #self.synt_meas = np.abs(self.z)**2 + (1. + 0.j) * self.noise 

        self.NSR = np.linalg.norm(self.noise)**2 / (4 * np.linalg.norm(np.abs(self.z)**2)**2 )
        return self.synt_meas, self.noise, self.NSR

    
