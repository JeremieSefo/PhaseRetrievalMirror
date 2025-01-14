import numpy as np
from PIL import Image
import skimage as ski
import matplotlib.pyplot as plt
import scipy.misc
import cv2
import imageio

def crop_center(img, crop_width, crop_height):
    # Load the image
    #img = cv2.imread(img_path)
    height, width = img.shape

    # Calculate the center
    center_x, center_y = width // 2, height // 2

    # Calculate the cropping coordinates
    x1 = max(0, center_x - crop_width // 2)
    y1 = max(0, center_y - crop_height // 2)
    x2 = min(width, center_x + crop_width // 2)
    y2 = min(height, center_y + crop_height // 2)

    # Crop the image
    cropped_img = img[y1:y2, x1:x2]

    return cropped_img

def get_phantom(dim):
    phantom = ski.img_as_float(ski.data.shepp_logan_phantom())
    return ski.transform.resize(phantom, (dim, dim))

class setUpImage:
    def __init__(self, Nx, Ny, Kx, Ky, tx, ty):
        self.Nx = Nx
        self.Ny = Ny
        self.Kx = Kx
        self.Ky = Ky
        self.tx = tx
        self.ty = ty

    def __call__(self,):
 
        NumPix = -0 + 0 * np.floor(0.5 * (1-2**(-1)) *self.Nx)
        bord = 1 * NumPix/self.Nx
        #x_true = im[500:516, 500:516]

        mask = (0 + 0j) * np.zeros((self.Nx, self.Ny))
        #mask[int(bord * self.Nx):int((1-bord)*self.Nx),int(bord * self.Ny):int((1-bord)*self.Ny)] = ((1)*1 + (0) *1j) * np.ones((int((1-2*bord)*self.Nx),int((1-2*bord)*self.Ny)))
        Lower = int(bord * self.Nx)
        Upper = int((1-bord)*self.Nx)

        # object support
        # Kx = (self.Nx - 1)//4 #any natural integer below (self.Nx - 1)//2.  
        # Ky = (self.Ny - 1)//4 #any natural integer below (self.Ny - 1)//2.  
        sx = (2 * self.Kx + 1) / self.Nx #ratio: object length/ full image length,  between 0 and 1
        sy = (2 * self.Ky + 1) / self.Ny #ratio: object length/ full image length,  between 0 and 1
        sNx = int(sx * self.Nx) #int is optional
        sNy = int(sy * self.Ny) #int is optional
        # mask support
        # tx = .05 #parameter between 0 and 1  for e(t) # 0.35  failed already
        # ty = .05 #parameter between 0 and 1  for e(t) # 0.35  failed already
        qx = int(self.Kx + self.tx * ((self.Nx - 1)//2 - self.Kx)) #any natural integer above kx and below (self.Nx - 1)//2.
        qy = int(self.Ky + self.ty * ((self.Ny - 1)//2 - self.Ky)) #any natural integer above ky and below (self.Ny - 1)//2.
        #ex = (1 - tx) * 1 + tx * (1/sx) # e(t) between 1 and (1/s) is the ratio = estimated support length /  object length
        ex = (2 * qx + 1)/ sNx # e(t) between 1 and (1/s) is the ratio = estimated support length /  object length
        ey = (2 * qy + 1)/ sNy # e(t) between 1 and (1/s) is the ratio = estimated support length /  object length
        esNx = int(ex * sx * self.Nx) #int is optional
        esNy = int(ey * sy * self.Ny) #int is optional
        #mask = (0 + 0j) * np.zeros((self.Nx, self.Ny))
        # int(0.5 * (1 - e * s) * self.Nx):int(0.5 * (1 + e * s) * self.Nx), int(0.5 * (1 - e * s) * self.Ny):int(0.5 * (1 + e * s) * self.Ny)
        mask[(self.Nx - esNx)//2  : (self.Nx - esNx)//2 + esNx, (self.Ny - esNy)//2  : (self.Ny - esNy)//2 + esNy] = ((1)*1 + (0) *1j) * np.ones((esNx, esNy))

        true_support = (0 + 0j) * np.zeros((self.Nx, self.Ny))
        #true_support[ int(0.5 * (1 -  s) * self.Nx)  : int(0.5 * (1 +  s) * self.Nx) , int(0.5 * (1 -  s) * self.Ny) : int(0.5 * (1 +  s) * self.Ny) ] = ((1)*1 + (0) *1j) * np.ones((int(s * self.Nx), int(s * self.Ny)))
        true_support[(self.Nx - sNx)//2  : (self.Nx - sNx)//2 + sNx, (self.Ny - sNy)//2  : (self.Ny - sNy)//2 + sNy ] = ((1)*1 + (0) *1j) * np.ones((sNx, sNy))
        lowerX = (self.Nx - sNx)//2
        upperX = (self.Nx - sNx)//2 + sNx
        lowerY = (self.Ny - sNy)//2
        upperY = (self.Ny - sNy)//2 + sNy

        


        x_true = (0 + 0j) * np.zeros((self.Nx,self.Ny))

        # (ones + i ones) image

        i, k = np.meshgrid(np.arange(int(self.Nx/2)), np.arange(int(self.Ny/2)))
        #omega = np.exp( - 2 * np.pi * 1j /int(self.Nx/2) * int(self.Ny/2) ) # (i + k *0j) *  #i+k**2-k*i   #k + i**2 - i*k
        grd_truths = []
        #x_true[int(bord * self.Nx):int((1-bord)*self.Nx),int(bord * self.Ny):int((1-bord)*self.Ny)] = ((1)*1 + (1) *1j) * np.ones((int((1-2*bord)*self.Nx),int((1-2*bord)*self.Ny))) #(7 + 0j)* np.random.normal(0, 1, size = (int(self.Nx/2),int(self.Ny/2))) + (0 + 5j)* np.random.normal(0, 1, size = (int(self.Nx/2),int(self.Ny/2))) #(i + k *1j) * np.ones((int(self.Nx/2),int(self.Ny/2)))
        im = np.ones((self.Nx, self.Ny))
        im_res = cv2.resize(im, ((sNx), (sNy)), interpolation=cv2.INTER_AREA)
        x_true[lowerX : upperX, lowerY : upperY] = (1 + 0.j) * np.ones(((sNx), (sNy))) #  im_res / np.max(np.abs(im_res)) + 1.j * (- im_res/ np.max(np.abs(im_res)))
        #x_true = x_true.real / np.max(np.abs(x_true.real)) + (x_true.imag / np.max(np.abs(x_true.imag))) * 1.j
        x_true = np.rot90(x_true, -1)
        x_true = np.rot90(x_true, -1)
        grd_truths.append(x_true)

        # ring + i disk of Gaussians balls 

        image = imageio.imread('ring.png', mode='F')
        image = np.array(image)
        image = cv2.resize(image, ((sNx), (sNy)), interpolation=cv2.INTER_AREA)
        image_padded = np.pad(image, (self.Nx - sNx)//2, 'constant')
        x_true =  image_padded / np.max(np.abs(image_padded)) + (image_padded / np.max(np.abs(image_padded))) * 1.j
        grd_truths.append(x_true)
        # Define the size of the image
        size = 1000
        radius = 700
        # Create a grid of points
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        # Define the centered disk
        disk = (X**2 + Y**2) <= (radius / size)**2
        # Create the image with ones in the disk and zeros outside
        imagedisk = np.zeros((size, size))
        imagedisk[disk] = 1
        image = cv2.resize(imagedisk, ((sNx), (sNy)), interpolation=cv2.INTER_AREA)
        imagedisk_padded = np.pad(image, (self.Nx - sNx)//2, 'constant')

        x_true =  image_padded / np.max(np.abs(image_padded)) + (imagedisk_padded / np.max(np.abs(imagedisk_padded))) * 1.j
        grd_truths.append(x_true)




        # cancer cell

        x_true = (0 + 0j) * np.zeros((self.Nx,self.Ny))
        img = Image.open('ISIC_0000004_cancer.jpg')
        #img = iio.v2.imread('ISIC_0000004_cancer.jpg')
        #x_true = Image.rgb2gray(img)
        #print('true_support.shape', true_support.shape)
        x_true3 = np.array(img) #.resize((self.Nx, self.Ny))
        x_true3 = cv2.resize(x_true3, ((sNx), (sNy)), interpolation=cv2.INTER_AREA)
        #x_true3_imag = cv2.resize(x_true3.imag, (int(s * self.Nx), int(s * self.Ny)), interpolation=cv2.INTER_AREA
        #x_true = x_true/np.max(np.abs(x_true))
        x_true[lowerX : upperX, lowerY : upperY] = x_true3[:, :,0] / np.max(np.abs(x_true3[:, :,0])) + (1j) * (x_true3[:, :,2]/ np.max(np.abs(x_true3[:, :,2])))# np.zeros((self.Nx,self.Ny))
        x_true = x_true.real / np.max(np.abs(x_true.real)) + (x_true.imag / np.max(np.abs(x_true.imag))) * 1j
        #x_true *= mask 
        x_true = np.rot90(x_true, -1)
        x_true = np.rot90(x_true, -1)
        #x_true *= mask 
        img.save('resized_image.jpg')
        grd_truths.append(x_true)
        
        #complex cameraman

        x_true = (0 + 0j) * np.zeros((self.Nx,self.Ny))
        img = ski.img_as_float(ski.data.camera())
        img_res = ski.transform.resize(img, (self.Nx, self.Ny))
        img_res = cv2.resize(img_res, ((sNx), (sNy)), interpolation=cv2.INTER_AREA)
        x_true[lowerX : upperX, lowerY : upperY] = 1. * img_res + 1.j * (- img_res)
        x_true = x_true.real / np.max(np.abs(x_true.real)) + (x_true.imag / np.max(np.abs(x_true.imag))) * 1.j
        #x_true *= mask 
        # x_true = np.rot90(x_true, -1)
        # x_true = np.rot90(x_true, -1)
        grd_truths.append(x_true)

        #real cameraman

        x_true = (0 + 0j) * np.zeros((self.Nx,self.Ny))
        img = ski.img_as_float(ski.data.camera())
        img_res = ski.transform.resize(img, (self.Nx, self.Ny))
        img_res = cv2.resize(img_res, ((sNx), (sNy)), interpolation=cv2.INTER_AREA)
        x_true[lowerX : upperX, lowerY : upperY] = 1. * img_res + 0.j# * (- img_res)
        x_true = x_true.real / np.max(np.abs(x_true.real)) + 0.j #* (x_true.imag / np.max(np.abs(x_true.imag))) * 
        #x_true *= mask 
        # x_true = np.rot90(x_true, -1)
        # x_true = np.rot90(x_true, -1)
        grd_truths.append(x_true)

        # cameraman

        x_true = (0 + 0j) * np.zeros((self.Nx,self.Ny))
        image = imageio.imread('cameraman.png', mode='F')
        image = np.array(image)
        image = cv2.resize(image, ((sNx), (sNy)), interpolation=cv2.INTER_AREA)
        image_padded = np.pad(image, (self.Nx - sNx)//2, 'constant')
        x_true =  image_padded / np.max(np.abs(image_padded)) + (image_padded / np.max(np.abs(image_padded))) * 1.j
        # x_true = np.rot90(x_true, -1)
        # x_true = np.rot90(x_true, -1)
        grd_truths.append(x_true)
        
        #x_true[int(bord * self.Nx):int((1-bord)*self.Nx),int(bord * self.Ny):int((1-bord)*self.Ny)] = get_phantom(int((1-2*bord)*self.Nx)) + get_phantom(int((1-2*bord)*self.Nx)) * 1.j
        
        #complex shepp logan

        x_true = (0 + 0j) * np.zeros((self.Nx,self.Ny))
        img = get_phantom(self.Nx)
        img_res = cv2.resize(img.real, ((sNx), (sNy)), interpolation=cv2.INTER_AREA)
        x_true[lowerX : upperX, lowerY : upperY] = 1. * img_res + 1.j * ( img_res)
        x_true = x_true.real / np.max(np.abs(x_true.real)) + (x_true.imag / np.max(np.abs(x_true.imag))) * 1.j
        # x_true = np.rot90(x_true, -1)
        # x_true = np.rot90(x_true, -1)
        grd_truths.append(x_true)

        #real shepp logan

        x_true = (0 + 0j) * np.zeros((self.Nx,self.Ny))
        img = get_phantom(self.Nx)
        img_res = cv2.resize(img.real, ((sNx), (sNy)), interpolation=cv2.INTER_AREA)
        x_true[lowerX : upperX, lowerY : upperY] = 1. * img_res + 0.j #* ( img_res)
        x_true = x_true.real / np.max(np.abs(x_true.real)) + 0.j # (x_true.imag / np.max(np.abs(x_true.imag))) *
        # x_true = np.rot90(x_true, -1)
        # x_true = np.rot90(x_true, -1)
        grd_truths.append(x_true)

        #ascent

        x_true = (0 + 0j) * np.zeros((self.Nx,self.Ny))
        image = scipy.datasets.ascent().astype('complex').reshape((512, 512)) #resize((int((1-2*bord)*self.Nx)*int((1-2*bord)*self.Ny))) #.
        img_res = cv2.resize(image.real, ((sNx), (sNy)), interpolation=cv2.INTER_AREA)
        x_true[lowerX : upperX, lowerY : upperY] = 1. * img_res + 1.j * (- img_res)
        image = x_true.real / np.max(np.abs(x_true.real)) + (x_true.imag / np.max(np.abs(x_true.imag))) * 1.j
        # image = np.rot90(image, 1)
        # image = np.rot90(image, 1)
        grd_truths.append(image)
    
    # Change axis convention
       
        #x_true = (0 + 0j) * np.zeros((self.Nx,self.Ny))
        #x_true[int(bord * self.Nx):int((1-bord)*self.Nx),int(bord * self.Ny):int((1-bord)*self.Ny)] = np.resize(image, (int((1-2*bord)*self.Nx),int((1-2*bord)*self.Ny))) + np.resize(image, (int((1-2*bord)*self.Nx),int((1-2*bord)*self.Ny))) *1.j # image.resize((int((1-2*bord)*self.Nx),int((1-2*bord)*self.Ny))) #+ image.T.copy().resize((int((1-2*bord)*self.Nx),int((1-2*bord)*self.Nx))) * 1.j
        #x_true = x_true.real / np.max(np.abs(x_true.real)) + x_true.imag / np.max(np.abs(x_true.imag)) *1j
        
        #x_true = image.T.copy() + image.T.copy() * 1.j
        #x_true = x_true.resize((self.Nx, self.Ny))
        #x_true *= mask 
        #grd_truths.append(x_true)

        # a single centered dot

        x_true = (0 + 0j) * np.zeros((self.Nx,self.Ny))
        grd_truths.append(x_true)

        im1 = (1 + 0.j) * np.zeros((sNx, sNy))
        im1[0, 0] = 1. + 0j
        im1 = np.fft.fftshift(im1)
        #self.im1 = im1
        x_true[lowerX : upperX, lowerY : upperY] = 1. * im1 + 0.j * (im1)
        x_true = np.fft.ifftshift(x_true)
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

