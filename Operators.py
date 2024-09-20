import numpy as np

class operators:
    
    def __init__(self, algo ):
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

'''
def grad_psi_num(x):#numerical real derivative
    return nd.Gradient(psi)(x)
def grad_psi_(x):#real derivative _real
    return (np.linalg.norm(x)**2 + 1) * x
def f_old(x): # f in slow loop form 
    s = 0
    for r in range(m):
        s += (meas[r] - np.dot(A[r], x)**2 )**2 
    return s/(4*m)
'''
def f(x): # f in fast matrix form
    s = np.linalg.norm(meas- operator(x) )**2 
    return (s/4/m)
def grad_f_num(x): #numerical real derivative
    return nd.Gradient(f)(x)
def grad_f_ (x): #real derivative, fast matrix form _real
    z = A @ x
    a = A.T @ ( z*( z**2 - meas))
    return a / m
def grad_f_true_real(x): #real derivative, slow loop form 
    s = 0
    for r in range(m):
        s +=  ( np.linalg.norm(np.dot(A[r], x))**2 - meas[r]) * np.conjugate(A[r]) * np.dot(A[r], x)
    return s / m

def grad_f_wrong(x): #Wirtinger derivative, fast matrix form #theirs  #_wrong
    z = np.conjugate(A) @ x
    a = A.T @ ( z*( z * np.conjugate(z) - meas))
    return a / (1*m)

def grad_f_mieux(x): #Wirtinger derivative, fast matrix form #theirs
    z = A @ np.conjugate(x)
    y = A @ x
    a = np.conjugate(A).T @ ( y*( z * np.conjugate(z) - meas))
    return a / (1*m)
def grad_f1(x): #Wirtinger derivative, fast matrix form #theirs
    z = A @ (x)
    y = np.conjugate(A) @ x
    a = (A).T @ ( y*( z * np.conjugate(z) - meas))
    return a / (1*m)
#'''
def grad_f(x): #Wirtinger derivative, fast matrix form #theirs _new
    z = A @ np.conjugate(x)
    y = np.conjugate(A) @ x
    a = (A).T @ ( y*( z * np.conjugate(z) - meas))
    return a / (1*m)
#'''
def grad_f_ours(x): #Wirtinger derivative, fast matrix form
    z = A @ x
    a = np.conjugate(A).T @ ( z*( z * np.conjugate(z) - meas))
    return a / (1*m)

def grad_f_true_compl(x): #Wirtinger derivative, slow loop form
    s = 0
    for r in range(m):
        s +=  ( np.linalg.norm(np.dot(A[r], x))**2 - meas[r]) * A[r] * np.dot(np.conjugate(A[r]), x)
    return s / (1*m)
def see1(x): #Wirtinger derivative, slow loop form
    s = 0
    for r in range(m):
        s +=  ( np.linalg.norm(np.dot(A[r], x))**2 - meas[r]) * (A[r].reshape((n,1)) @ (np.conjugate(A[r]).reshape((1,n)))) @ x
    return s / (1*m)
def see(x): #Wirtinger derivative, slow loop form
    s = 0
    for r in range(m):
        s +=  ( np.linalg.norm(np.vdot(A[r], x))**2 - meas[r]) * (A[r].reshape((n,1)) @ (np.conjugate(A[r]).reshape((1,n)))) @ x
    return s / (1*m)
'''
def grad_f_true_compl_flow(x): #Wirtinger flow, slow loop form
    s = 0
    for r in range(m):
        s +=  ( np.linalg.norm(np.dot(A[r], x))**2 - meas[r]) * np.conjugate(A[r]) * np.dot(A[r], x)
    return s / (2*m)
'''
def breg_f(x, u):
    return f(x) - f(u) - np.vdot(grad_f(u), x-u) 
#'''
x = (1 + 0j)*np.random.normal(2, 1, size = (n,)) + (0 + 1j)*np.random.normal(2, 1, size = (n,))
y = (1 + 0j)*np.random.normal(2, 1, size = (n,)) + (0 + 0j)*np.random.normal(2, 1, size = (n,))
#x = np.array([1 + 2j, 2 + 4j])
#y = 2 *x
print(x)
print('see',  grad_f(x))
#print(breg_f(y,x))
print(np.vdot(grad_f(x), y))
#print('look', np.linalg.norm(grad_psi(x)  - grad_psi_num(x)))
print('look', np.linalg.norm(grad_f(x)  - grad_f_num(x)))
print(np.linalg.norm(grad_f_true_compl(x)  - see1(x)))
print(np.linalg.norm(grad_f(x)-see(x)))
#print(grad_f(x), see(x))

#print(f(x_true_vect))
#'''