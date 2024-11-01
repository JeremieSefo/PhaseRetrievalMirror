
import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift



class sensing_model_matrix_based:
    def __init__(self, Nx, Ny, rx, ry):
        self.init_matrix(Nx, Ny, rx, ry)
        (self.Qx, self.Qy) = int(rx * Nx), int(ry * Ny)
    def init_matrix(self, Nx, Ny):
        pass #print('sum',Nx + Ny )

    def __call__(self, x):
        return (fftn((x.reshape((self.Nx, self.Ny))), s = (self.Qx, self.Qy), norm = 'ortho')) #self.Matrix @ np.conjugate(x) #(self.Nx, self.Ny)


class iid_stdd_Gauss(sensing_model_matrix_based):
    def __init__(self, Nx, Ny, rx, ry):
        super().__init__(Nx, Ny, rx, ry)
        
    def init_matrix(self, Nx, Ny, rx, ry):
        n = Nx * Ny
        (Qx, Qy) = int(rx * Nx), int(ry * Ny)
        m = Qx * Qy
        self.Matrix = (1. + 0j) * np.random.normal(0,1, size = (m,n)) + (0. + 0j) * np.random.normal(0,1, size = (m,n)) # i.i.d. std Gaussian 
        #print('product',Nx * Ny )
        #return self.Matrix
        #self.Matrix = (1. + 0j)*np.eye(n)

class FourierMatrix(sensing_model_matrix_based):
    def __init__(self, Nx, Ny, rx, ry):
        super().__init__(Nx, Ny, rx, ry)

    def init_matrix(self, Nx, Ny, rx, ry):
        self.Nx = Nx
        self.Ny = Ny
        self.rx = rx
        self.ry = ry
        n = Nx * Ny
        #n = self.Nx * self.Ny
        #(rx, ry) = 2., 2.
        (Qx, Qy) = int(rx * Nx), int(ry * Ny)
        m = Qx * Qy
        self.Matrix = (1 + 1j)*np.zeros((m,n))
        for k in range(n):
            j = int( k%Ny )
            i = int( (k-j)/ Ny )
            E = np.zeros((Nx,Ny))
            E[i,j] = 1. 
            #print(np.linalg.norm(E))
            self.Matrix[0:m,k] = fftn(E, s = (Qx,Qy), norm = 'ortho').flatten()
        


def syntMeas(self, x_true, noise):
    A = sense_via(self.sensingModel)
    z = A @  np.conjugate(x_true)

    return z * np.conjugate(z) + noise

def measure(self, x):
    A = sense_via(self.sensingModel)
    z = A @  np.conjugate(x)

    return z * np.conjugate(z)


        
