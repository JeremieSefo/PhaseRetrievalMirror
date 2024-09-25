from numpy.fft import fftn
import numpy as np



class sensing_model_matrix_based:
    def __init__(self, Nx, Ny):
        self.init_matrix(Nx, Ny)

    def init_matrix(self, Nx, Ny):
        pass

    def __call__(self, x):
        return self.A @ np.conjugate(x) 


class iid_stdd_Gauss(sensing_model_matrix_based):
    def __init__(self, Nx, Ny):
        super().__init__(Nx, Ny)
        
    def __call__(self, x):
        return self.A @ np.conjugate(x)

    def init_matrix(self, Nx, Ny):
        n = Nx * Ny
        m = n
        self.A = (1. + 0j) * np.random.normal(0,1, size = (m,n)) + (0. + 0j) * np.random.normal(0,1, size = (m,n)) # i.i.d. std Gaussian 
        #A = (1. + 0j)*np.eye(n)

#class FourierMatrix(sensing_model_matrix_based):


def FourierMatrix(Nx, Ny):
    n = Nx * Ny
    #n = self.Nx * self.Ny
    (rx, ry) = 1., 1.
    (Qx, Qy) = int(rx * Nx), int(ry * Ny)
    m = Qx * Qy

    A = (1 + 1j)*np.zeros((m,n))
    for k in range(n):
        j = int( k%Ny )
        i = int( (k-j)/ Ny )
        E = np.zeros((Nx,Ny))
        E[i,j] = 1. 
        #print(np.linalg.norm(E))
        A[0:m,k] = fftn(E, s = (Qx,Qy), norm = 'ortho').flatten()
    return A


def get_matrix(sensingModel):
    #if sensingModel ++ 
    return 1

class sense_via:
    def __init__(self, sensingModel): # sensingModel,
        self.sensingModel = sensingModel
        #self.Nx = Nx
        #self.Ny = Ny

    def __call__(self, Nx, Ny):
        if self.sensingModel == 'FourierMatrix':
            return FourierMatrix(Nx, Ny)
        if self.sensingModel == 'standard Gaussian':
            return self.iid_stdd_Gauss
    


    def iid_stdd_Gauss(self, Nx, Ny):
        n = Nx * Ny
        m = n
        A = (1. + 0j) * np.random.normal(0,1, size = (m,n)) + (0. + 0j) * np.random.normal(0,1, size = (m,n)) # i.i.d. std Gaussian 
        #A = (1. + 0j)*np.eye(n)
        return A
    

    def syntMeas(self, x_true, noise):
        A = sense_via(self.sensingModel)
        z = A @  np.conjugate(x_true)

        return z * np.conjugate(z) + noise
    
    def measure(self, x):
        A = sense_via(self.sensingModel)
        z = A @  np.conjugate(x)

        return z * np.conjugate(z)


        
