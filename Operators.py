import numpy as np
import SensingMatrix as SM

class operators:
    
    def __init__(self, algo, meas, A):
        self.meas = meas
        self.A = A
        self.algo = algo
        if self.algo == 'default':
            self.a, self.b = 1., 0.
        if self.algo =='complex mirror':
            self.a, self.b = 0., 1.

    def psi(self, x):
        self.x = x
        return 0.25 * ( self.a * np.linalg.norm(self.x)**4 + self.b * (np.linalg.norm(self.x.real)**4 + np.linalg.norm(self.x.imag)**4)) + 0.5 * np.linalg.norm(self.x)**2

        '''
        if self.algo == 'default': 
            return 0.25 *  np.linalg.norm(self.x)**4 + 0.5 * np.linalg.norm(self.x)**2
        if self.algo == 'complex mirror':
            return 0.25 * ((np.linalg.norm(self.x.real)**4 + np.linalg.norm(self.x.imag)**4)) + 0.5 *  np.linalg.norm(x)**2
        '''
    def grad_psi(self, z): #Wirtinger derivatives
        self.z = z
        if self.algo == 'default': 
            return 1 * (np.linalg.norm(self.z)**2 + 1) * z 
        if self.algo == 'complex mirror':
            return  self.z + ((np.linalg.norm(self.z.real))**2)*self.z.real + ((np.linalg.norm(self.z.imag))**2)*self.z.imag * 1j
    
    def breg_psi(self, x, u):
        self.x = x
        self.u = u
        return self.psi(self.x) - self.psi(self.u) - np.vdot(self.grad_psi(self.u), self.x-self.u)
    
    def f(self, x):
        s = np.linalg.norm(self.meas- np.abs(self.A(x))**2 )**2
        m = len(self.meas) 
        return (s/4/m)
    
    def grad_f(self, x): #Wirtinger derivative, fast matrix form #theirs _new
        z = self.A(x)
        y = np.conjugate((self.A).Matrix) @ x
        a = ( (self.A).Matrix).T @ ( y*( z * np.conjugate(z) - self.meas))
        m = len(self.meas) 
        return a / (1*m)
    
    def breg_f(self, x, u):
        return self.f(x) - self.f(u) - np.vdot(self.grad_f(u), x-u)