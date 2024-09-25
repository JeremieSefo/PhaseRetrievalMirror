import numpy as np
import SensingMatrix as SM

class operators:
    
    def __init__(self, algo, meas ):
        self.meas = meas
        if algo == 'default':
            self.a, self.b = 1., 0.
        if algo =='complex mirror':
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
        return operators.psi(self.x) - operators.psi(self.u) - np.vdot(operators.grad_psi(self.u), self.x-self.u)
    def f(self, x):
        s = np.linalg.norm(self.meas- operator(x) )**2
        m = len(meas) 
        return (s/4/m)

