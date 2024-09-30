import numpy as np
import SensingMatrix as SM

class operators:
    
    def __init__(self, algo, meas, A):
        self.meas = meas
        self.A = A
        self.algo = algo
        self.a, self.b = 1., 0.
        if self.algo == 'real mirror':
            self.a, self.b = 1., 0.
        if self.algo =='complex mirror':
            self.a, self.b = 0., 1.

    def psi(self, x):
        self.x = x
        return 0.25 * ( self.a * np.linalg.norm(self.x)**4 + self.b * (np.linalg.norm(self.x.real)**4 + np.linalg.norm(self.x.imag)**4)) + 0.5 * np.linalg.norm(self.x)**2

        '''
        if self.algo == 'mirror': 
            return 0.25 *  np.linalg.norm(self.x)**4 + 0.5 * np.linalg.norm(self.x)**2
        if self.algo == 'complex mirror':
            return 0.25 * ((np.linalg.norm(self.x.real)**4 + np.linalg.norm(self.x.imag)**4)) + 0.5 *  np.linalg.norm(x)**2
        '''
    def grad_psi(self, z): #Wirtinger derivatives
        self.z = z
        if self.algo == 'real mirror': 
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
    s1 = np.sum(meas)
    
    s2 = np.sum( [np.vdot( A[r, :], A[r, :]) for r in range(m)])
    lamda = np.sqrt(n*s1/s2)
    '''
    Y = np.zeros((n,n)) #.reshape(n, n)

    for r in range(m):
        Y += meas[r] * A[r].reshape(1, n) @ np.conjugate(A[r]).reshape(1, n).T
    Y /= m 
    '''
    Y = np.conjugate(A).T @ np.diag(meas) @ A
    eigenval, eigenvect = np.linalg.eig(Y)
    x0 = eigenvect.real[eigenval.real == max(eigenval.real)][0]  #Complex eigenvalues
    x0 = x0 * (lamda/np.linalg.norm(x0))
    return [x0, eigenval.real, eigenvect.real]


def initialise(n, meas, A, type, real, imag, x_true_vect, mask ):
    if type == 'spectral':
        x = spectInit(meas, A)
    if type == 'Gaussian':
        x = real * ((1. + 0.j)*np.random.normal(0, 1, size = (n, ) ))+ imag * (0. + 1.j)*np.random.normal(0, 1, size = (n, ) )
    if type == 'close':
        guessNoise  = ((1. + 0.j)*np.random.normal(0, 1, size = x_true_vect.shape) +  (0. + 1.j)* np.random.normal(0, 1, size = x_true_vect.shape))
        x = x_true_vect  + (1./(1. * np.linalg.norm(guessNoise)))*guessNoise

    return x * mask.reshape((n,))

