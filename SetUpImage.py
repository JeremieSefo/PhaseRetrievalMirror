import numpy as np
from PIL import Image
import skimage as ski
import matplotlib.pyplot as plt
import scipy.misc

def get_phantom(dim):
    phantom = ski.img_as_float(ski.data.shepp_logan_phantom())
    return ski.transform.resize(phantom, (dim, dim))

class setUpImage:
    def __init__(self, Nx, Ny):
        self.Nx = Nx
        self.Ny = Ny

    def __call__(self,):
 
        NumPix = 0 + 1 * np.floor(0.5 * (1-2**(-1)) *self.Nx)
        bord = NumPix/self.Nx
        #x_true = im[500:516, 500:516]
        mask = (0 + 0j) * np.zeros((self.Nx,self.Ny))
        mask[int(bord * self.Nx):int((1-bord)*self.Nx),int(bord * self.Ny):int((1-bord)*self.Ny)] = ((1)*1 + (0) *1j) * np.ones((int((1-2*bord)*self.Nx),int((1-2*bord)*self.Ny)))
        lower = int(bord * self.Nx)
        upper = int((1-bord)*self.Nx)

        x_true = (0 + 0j) * np.zeros((self.Nx,self.Ny))

        i, k = np.meshgrid(np.arange(int(self.Nx/2)), np.arange(int(self.Ny/2)))
        #omega = np.exp( - 2 * np.pi * 1j /int(self.Nx/2) * int(self.Ny/2) ) # (i + k *0j) *  #i+k**2-k*i   #k + i**2 - i*k
        grd_truths = []
        x_true[int(bord * self.Nx):int((1-bord)*self.Nx),int(bord * self.Ny):int((1-bord)*self.Ny)] = ((1)*1 + (1) *1j) * np.ones((int((1-2*bord)*self.Nx),int((1-2*bord)*self.Ny))) #(7 + 0j)* np.random.normal(0, 1, size = (int(self.Nx/2),int(self.Ny/2))) + (0 + 5j)* np.random.normal(0, 1, size = (int(self.Nx/2),int(self.Ny/2))) #(i + k *1j) * np.ones((int(self.Nx/2),int(self.Ny/2)))
        x_true = x_true.real / np.max(np.abs(x_true.real)) + x_true.imag / np.max(np.abs(x_true.imag)) *1j
        #x_true = np.rot90(x_true, -1)  # Change axis convention
        x_true = np.rot90(x_true, -1)
        x_true = np.rot90(x_true, -1)
        grd_truths.append(x_true)
        #x_true[1,1] = 2 + 1j
        #print(x_true)
        #x_true[0, 1:self.Ny-1] = np.ones((1,self.Ny-2))
        img = Image.open('ISIC_0000004_cancer.jpg')
        #img = iio.v2.imread('ISIC_0000004_cancer.jpg')
        #x_true = Image.rgb2gray(img)
        x_true3 = np.array( img.resize((self.Nx, self.Ny)))
        #x_true = x_true/np.max(np.abs(x_true))
        x_true = x_true3[:, :,0] + (1j) * x_true3[:, :,2]# np.zeros((self.Nx,self.Ny))
        x_true = x_true.real / np.max(np.abs(x_true.real)) + x_true.imag / np.max(np.abs(x_true.imag)) *1j
        x_true *= mask 
        x_true = np.rot90(x_true, -1)
        x_true = np.rot90(x_true, -1)
        grd_truths.append(x_true)
        #x_true *= mask 
        img.save('resized_image.jpg')
        #print(x_true[0])
        #print(np.shape(x_true))
        #x_true = img
        #plt.imshow(img)
        img = ski.img_as_float(ski.data.camera())
        img_res = ski.transform.resize(img, (self.Nx, self.Ny))
        x_true = 2. * img_res + 2.j * (- img_res)
        x_true = x_true.real / np.max(np.abs(x_true.real)) + x_true.imag / np.max(np.abs(x_true.imag)) *1j
        
        #x_true = (1 + 1j) * np.ones(x_true.shape)
        x_true *= mask 
        x_true = np.rot90(x_true, -1)
        x_true = np.rot90(x_true, -1)
        grd_truths.append(x_true)

        x_true[int(bord * self.Nx):int((1-bord)*self.Nx),int(bord * self.Ny):int((1-bord)*self.Ny)] = get_phantom(int((1-2*bord)*self.Nx)) + get_phantom(int((1-2*bord)*self.Nx)) * 1.j
        x_true = x_true.real / np.max(np.abs(x_true.real)) + x_true.imag / np.max(np.abs(x_true.imag)) *1j
        x_true *= mask 
        x_true = np.rot90(x_true, -1)
        x_true = np.rot90(x_true, -1)
        grd_truths.append(x_true)

        image = scipy.datasets.ascent().astype('complex').reshape((512, 512)) #resize((int((1-2*bord)*self.Nx)*int((1-2*bord)*self.Ny))) #.
        image /= image.max()
        image = np.rot90(image, 1)
        image = np.rot90(image, 1)
    
    # Change axis convention
        #image /= np.max(np.abs(x_true))
        x_true = (0 + 0j) * np.zeros((self.Nx,self.Ny))

        x_true[int(bord * self.Nx):int((1-bord)*self.Nx),int(bord * self.Ny):int((1-bord)*self.Ny)] = np.resize(image, (int((1-2*bord)*self.Nx),int((1-2*bord)*self.Ny))) + np.resize(image, (int((1-2*bord)*self.Nx),int((1-2*bord)*self.Ny))) *1.j # image.resize((int((1-2*bord)*self.Nx),int((1-2*bord)*self.Ny))) #+ image.T.copy().resize((int((1-2*bord)*self.Nx),int((1-2*bord)*self.Nx))) * 1.j
        x_true = x_true.real / np.max(np.abs(x_true.real)) + x_true.imag / np.max(np.abs(x_true.imag)) *1j
        #x_true = image.T.copy() + image.T.copy() * 1.j
        #x_true = x_true.resize((self.Nx, self.Ny))
        #x_true *= mask 
        grd_truths.append(x_true)


        x_true = (0 + 0j) * np.zeros((self.Nx,self.Ny))
        grd_truths.append(x_true)
        '''
        plt.imshow(x_true.real, cmap='gray')
        plt.colorbar()
        '''
        x_true_real = np.real(x_true)#.reshape((self.Nx, self.Ny))
        x_true_imag = np.imag(x_true)#.reshape((self.Nx, self.Ny))
        #print(x_true_real)
        #x_true = x_true_real
        '''
        fig, ((ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13)) = plt.subplots(nrows = 2, ncols=4, figsize=(10,5))
        
        ax00.imshow(x_true_imag, cmap='gray') 
        ax00.set_title("image ")
        ax00.axis('off')
        '''
        self.mask = mask
        self.grd_truths = grd_truths
        return grd_truths, mask
        ##plt.imshow(-1j * x_true)
        #plt.colorbar()    

