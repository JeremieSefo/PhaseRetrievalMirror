import numpy as np
import SensingMatrix as SM
from numpy.fft import fftn, ifftn, fftshift, ifftshift

class operators:
    
    def __init__(self, algo, meas, A, mask):
        self.mask = mask
        self.meas = meas
        self.A = A
        self.algo = algo
        self.a, self.b = 1., 0.
        if self.algo == 'real mirror' or self.algo == 'FIENUP':
            self.a, self.b = 1., 0.
        if self.algo =='complex mirror':
            self.a, self.b = 0., 1.

    def psi(self, x):
        self.x = x
        return  (0.25 * ( self.a * np.linalg.norm(self.x)**4 + self.b * (np.linalg.norm(self.x.real)**4 + np.linalg.norm(self.x.imag)**4)) + 0.5 * np.linalg.norm(self.x)**2 )

        '''
        if self.algo == 'mirror': 
            return 0.25 *  np.linalg.norm(self.x)**4 + 0.5 * np.linalg.norm(self.x)**2
        if self.algo == 'complex mirror':
            return 0.25 * ((np.linalg.norm(self.x.real)**4 + np.linalg.norm(self.x.imag)**4)) + 0.5 *  np.linalg.norm(x)**2
        '''
    def grad_psi(self, z): #Wirtinger derivatives
        self.z = z
        if self.algo == 'real mirror' or self.algo == 'FIENUP': 
            return 1 * (np.linalg.norm(self.z)**2 + 1) * z 
        if self.algo == 'complex mirror':
            return  self.z + ((np.linalg.norm(self.z.real))**2)*self.z.real + ((np.linalg.norm(self.z.imag))**2)*self.z.imag * 1j
        
    def breg_psi(self, x, u):
        self.x = x
        self.u = u
        #l = np.vdot(self.grad_psi(self.u), self.x-self.u)
        #print("look", np.vdot(self.grad_psi(self.u), self.x-self.u))
        return self.psi(x) - self.psi(u) - np.vdot(self.grad_psi(u), x-u)
    
    def f(self, x):
        s = np.linalg.norm(self.meas- np.abs(self.A(x).flatten())**2 )**2
        m = len(self.meas) 
        return (s/4/m)
    
    def grad_f(self, x): #Wirtinger derivative, fast matrix form #theirs _new
        z = (self.A(x)).flatten() # (fftn(x.reshape(self.mask.shape), s = self.mask.shape, norm = 'ortho')).flatten()            #self.A(x) #
        y = np.conjugate((self.A).Matrix) @ x
        a =  ( (self.A).Matrix).T @ ( y*( z * np.conjugate(z) - self.meas)) # (ifftn((y*( z * np.conjugate(z) - self.meas)).reshape(self.mask.shape), s = self.mask.shape, norm = 'ortho')).flatten()   #( (self.A).Matrix).T @ ( y*( z * np.conjugate(z) - self.meas))
        m = len(self.meas) 
        return a / (1*m)
    
    def breg_f(self, x, u):
        return self.f(x) - self.f(u) - np.vdot(self.grad_f(u), x-u)
    
    def grad_psi_star(self, z):
        b = np.linalg.norm(z)
        a = b**2
        print('a', a)
        
        if a == 0.:
            return z
        else: 
            '''
            c = b**3 * (3*(27*a+4))**(0.5) + 9* (a**2)
            t = (2 * c**2)**(1/3) - 2 * a * (3**(1/3))
            t /= a * ((36 * c)**(1/3))
            '''      
            p = [a, 0., 1., -1.]
            #print(p)
            t = np.roots(p)
            t = t.real[abs(t.imag) == 0. ][0]
            print('t', t)
            #import pdb
            #pdb.set_trace()
            return  t*z
    
def smoothnessPara_L(A, noise):

    z = np.linalg.norm(A, axis = -1)**2
    L = np.mean( z * ( 3*z + np.max(noise)))
    return L

def spectInit(meas, A):
    
    n = A.shape[1]
    m = A.shape[0]
    # Sum of measurements
    s1 = np.sum(meas)
    
    # Optimized sum of squared rows of A (avoiding the loop)
    s2 = np.sum(np.abs(A)**2, axis=1).sum()
    # Compute lambda directly
    lamda = np.sqrt(n * s1 / s2)
    '''
    Y = np.zeros((n,n)) #.reshape(n, n)

    for r in range(m):
        Y += meas[r] * A[r].reshape(1, n) @ np.conjugate(A[r]).reshape(1, n).T
    Y /= m 
    '''    
    # Optimized computation of Y
    Y = np.conjugate(A).T @ np.diag(meas) @ A
    # Efficient eigenvalue computation
    eigenval, eigenvect = np.linalg.eig(Y)
    
    # Extract the eigenvector corresponding to the largest eigenvalue
    x0 = eigenvect.real[eigenval.real == max(eigenval.real)][0]  #Complex eigenvalues
    y0 = eigenvect.imag[eigenval.imag == max(eigenval.imag)][0]  #Complex eigenvalues
    # x0 = eigenvect[:, np.argmax(eigenval)].real
    
    # Normalize the eigenvector
    x0 = x0 * (lamda / np.linalg.norm(x0))
    y0 = y0 * (lamda / np.linalg.norm(y0))
    return [x0 + 1j * y0, eigenval.real, eigenvect.real]

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
    samples = ([np.random.normal(mu, sigma, size) for mu in mus]) #[np.random.poisson(lam, size) for lam in lambdas]
    
    # Combine samples into a multivariate array
    multivariate_samples = np.stack(samples, axis=-1)
    return multivariate_samples[0]

def initialise(n, meas, A, type, real, imag, x_true_vect, mask, noise_lvl ):
    if type == 'spectral':
        x, eigenval_real, eigenvect_real = spectInit(meas, A)
        x = real * x.real + imag * 1j * x.imag 
    if type == 'Gaussian':
        mus = x_true_vect
        num_samples = 1
        #x = real * generate_multivariate_gaussian(mus.real, noise_lvl, num_samples) +  imag * 1.j * generate_multivariate_gaussian(mus.imag, noise_lvl, num_samples)

        x = real * ((1. + 0.j)*np.abs(np.random.normal(0.75, noise_lvl, size = (n, ) ))) + imag * (0. + 1.j)*np.abs(np.random.normal(0.75, noise_lvl, size = (n, ) ))
        #x = x * (noise_lvl/(1. * np.linalg.norm(x)))
    if type == 'close':
        guessNoise  = real * ((1. + 0.j)*np.random.normal(0, 1, size = x_true_vect.shape) + imag  *  (0. + 1.j)* np.random.normal(0, 1, size = x_true_vect.shape))

        guessNoise *= (1.)/np.linalg.norm(guessNoise)
        
        noise_strength =  (noise_lvl) * np.linalg.norm(x_true_vect)
        guessNoise *= (noise_strength)

        x = x_true_vect  + guessNoise
    indices = np.logical_not(mask.reshape((n,)))
    x[indices] = 0
    NSR = np.linalg.norm(x - x_true_vect) / np.linalg.norm(x_true_vect)
    return x, NSR

def soft_shrinkage(x, lamda):
    return np.maximum(np.abs(x)-lamda, 0.) * np.sign(x)