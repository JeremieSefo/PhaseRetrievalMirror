from numpy.fft import fftn
import numpy as np
class sense_via:
    def __init__(self, sensingModel, Nx, Ny):
        self.sensingModel = sensingModel
        self.Nx = Nx
        self.Ny = Ny

        
    def FourierMatrix(self):
        n = self.Nx * self.Ny
        (rx, ry) = 1., 1.
        (Qx, Qy) = int(rx * self.Nx), int(ry * self.Ny)
        m = Qx * Qy

        A = (1 + 1j)*np.zeros((m,n))
        for k in range(n):
            j = int( k%self.Ny )
            i = int( (k-j)/ self.Ny )
            E = np.zeros((self.Nx,self.Ny))
            E[i,j] = 1. 
            #print(np.linalg.norm(E))
            A[0:m,k] = fftn(E, s = (Qx,Qy), norm = 'ortho').flatten()
        return A 

    def iid_stdd_Gauss(m,n):
        A = (1. + 0j) * np.random.normal(0,1, size = (m,n)) + (0. + 0j) * np.random.normal(0,1, size = (m,n)) # i.i.d. std Gaussian 
        #A = (1. + 0j)*np.eye(n)
        return A

    
        
