import numpy as np
from PIL import Image
import skimage as ski
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import scipy.misc
import cv2
import imageio
from scipy.ndimage import binary_dilation

def extend_support_with_holes(image, dilation_radius=1):
    """
    Extend the support (boundary) of the shape in the image by ones using dilation,
    while preserving the holes inside the support shape.
    
    Args:
    - image (np.ndarray): The binary image with the support shape.
    - dilation_radius (int): The radius by which to extend the support region.
    
    Returns:
    - np.ndarray: The padded image with extended support and preserved holes.
    """
    # Perform binary dilation with a square structuring element
    structure = np.ones((dilation_radius * 2 + 1, dilation_radius * 2 + 1), dtype=bool)
    dilated_image = binary_dilation(image, structure=structure).astype(int)
    
    # Mask out the holes: keep original zero regions that were zeros in the input
    extended_image = np.where(image == 0, dilated_image, image)
    
    return extended_image


def exact_support(image): 
    # Step 1: Find the indices of non-zero elements
    non_zero_indices = (image != 0 )
    support = np.zeros(image.shape)
    support[non_zero_indices] =  1.
    return support
    
def rect_support(image): 
    # Step 1: Find the indices of non-zero elements
    non_zero_indices = np.argwhere(image != 0 )
    
    # Step 2: Get the bounding box (min and max indices)
    min_row, min_col = non_zero_indices.min(axis=0)
    max_row, max_col = non_zero_indices.max(axis=0)

    box = np.zeros(image.shape)
    box[min_row : max_row + 1, min_col : max_col + 1 ] = 1.
    return box
    
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

class setUpImage_padding:
    def __init__(self, Nx, Ny, Kx, Ky, tx, ty):
        self.Nx = Nx
        self.Ny = Ny
        self.Kx = Kx
        self.Ky = Ky
        self.tx = tx
        self.ty = ty

    def __call__(self,):

        sNx = 2 * self.Kx  # int(sx * self.Nx) #int is optional
        sNy = 2 * self.Ky  # int(sy * self.Ny) #int is optional

        NumPix = -0 + 0 * np.floor(0.5 * (1-2**(-1)) * (self.Nx + sNx))
        bord = 1 * NumPix/(self.Nx + sNx )
        #x_true = im[500:516, 500:516]

        #mask[int(bord * self.Nx):int((1-bord)*self.Nx),int(bord * self.Ny):int((1-bord)*self.Ny)] = ((1)*1 + (0) *1j) * np.ones((int((1-2*bord)*self.Nx),int((1-2*bord)*self.Ny)))
        Lower = int(bord * (self.Nx + sNx))
        Upper = int((1-bord)*(self.Nx + sNy))

        # object support
        # Kx = (self.Nx - 1)//4 #any natural integer below (self.Nx - 1)//2.  
        # Ky = (self.Ny - 1)//4 #any natural integer below (self.Ny - 1)//2.  

        mask = (0 + 0j) * np.zeros((self.Nx + sNx, self.Ny + sNx))
        
        sx = (self.Nx ) / (self.Nx + sNx) #ratio: object length/ full image length,  between 0 and 1
        sy = (self.Ny ) / (self.Ny + sNx) #ratio: object length/ full image length,  between 0 and 1

        # mask support
        # tx = .05 #parameter between 0 and 1  for e(t) # 0.35  failed already
        # ty = .05 #parameter between 0 and 1  for e(t) # 0.35  failed already
        qx = int((self.Nx - 1)//2 + self.tx * ((self.Nx + sNx)//2 - self.Nx)) #any natural integer above kx and below (self.Nx - 1)//2.
        qy = int((self.Ny - 1)//2 + self.ty * ((self.Ny + sNy)//2 - self.Ny)) #any natural integer above ky and below (self.Ny - 1)//2.
        #ex = (1 - tx) * 1 + tx * (1/sx) # e(t) between 1 and (1/s) is the ratio = estimated support length /  object length
        ex = (2 * qx + 1)/ (self.Nx) # e(t) between 1 and (1/s) is the ratio = estimated support length /  object length
        ey = (2 * qy + 1)/ (self.Ny) # e(t) between 1 and (1/s) is the ratio = estimated support length /  object length
        esNx = int(ex * sx * (self.Nx + sNx)) #int is optional
        esNy = int(ey * sy * (self.Ny + sNy)) #int is optional
        #mask = (0 + 0j) * np.zeros((self.Nx, self.Ny))
        # int(0.5 * (1 - e * s) * self.Nx):int(0.5 * (1 + e * s) * self.Nx), int(0.5 * (1 - e * s) * self.Ny):int(0.5 * (1 + e * s) * self.Ny)
        
        mask[(self.Nx + sNx - esNx)//2  : (self.Nx + sNx - esNx)//2 + esNx, (self.Ny + sNy - esNy)//2  : (self.Ny + sNy - esNy)//2 + esNy] = ((1)*1 + (0) *1j) * np.ones((esNx, esNy))


        true_support = (0 + 0j) * np.zeros((self.Nx + sNx, self.Ny + sNy))
        #true_support[ int(0.5 * (1 -  s) * self.Nx)  : int(0.5 * (1 +  s) * self.Nx) , int(0.5 * (1 -  s) * self.Ny) : int(0.5 * (1 +  s) * self.Ny) ] = ((1)*1 + (0) *1j) * np.ones((int(s * self.Nx), int(s * self.Ny)))
        true_support[(sNx )//2  : (sNx )//2 + self.Nx, (sNy  )//2  : (sNy  )//2 + self.Ny ] = ((1)*1 + (0) *1j) * np.ones((self.Nx, self.Ny))
        lowerX = (self.Nx)//2
        upperX = (self.Nx)//2 + self.Nx
        lowerY = (self.Ny)//2
        upperY = (self.Ny)//2 + self.Ny

        


        x_true = (0 + 0j) * np.zeros((self.Nx + sNx,self.Ny + sNy))

        # (ones + i ones) image

        i, k = np.meshgrid(np.arange(int((self.Nx + sNx )/2)), np.arange(int((self.Ny + sNy)/2)))
        #omega = np.exp( - 2 * np.pi * 1j /int(self.Nx/2) * int(self.Ny/2) ) # (i + k *0j) *  #i+k**2-k*i   #k + i**2 - i*k
        grd_truths = []
        #x_true[int(bord * self.Nx):int((1-bord)*self.Nx),int(bord * self.Ny):int((1-bord)*self.Ny)] = ((1)*1 + (1) *1j) * np.ones((int((1-2*bord)*self.Nx),int((1-2*bord)*self.Ny))) #(7 + 0j)* np.random.normal(0, 1, size = (int(self.Nx/2),int(self.Ny/2))) + (0 + 5j)* np.random.normal(0, 1, size = (int(self.Nx/2),int(self.Ny/2))) #(i + k *1j) * np.ones((int(self.Nx/2),int(self.Ny/2)))
        im = np.ones((self.Nx + sNx, self.Ny + sNy))
        im_res = cv2.resize(im, ((self.Ny), (self.Nx)), interpolation=cv2.INTER_AREA)
        # x_true[lowerX : upperX, lowerY : upperY] = (1 + 0.j) * np.ones(((sNx), (sNy))) #  im_res / np.max(np.abs(im_res)) + 1.j * (- im_res/ np.max(np.abs(im_res)))
        #x_true = x_true.real / np.max(np.abs(x_true.real)) + (x_true.imag / np.max(np.abs(x_true.imag))) * 1.j
        image_padded = np.pad(im_res, ( ((sNx)//2, (sNx)//2), ((sNy)//2, (sNy)//2)), 'constant')
        grd_truths.append(image_padded)

        # ring + 0 i  Gaussians balls 

        image = imageio.imread('ring.png', mode='F')
        image = np.array(image)
        imagem = cv2.resize(image, ((self.Ny), (self.Nx)), interpolation=cv2.INTER_AREA)
        image_padded = np.pad(imagem, ( ((sNx)//2, (sNx)//2), ((sNy)//2, (sNy)//2)) , 'constant')
        x_true =  image_padded / np.max(np.abs(image_padded)) + (image_padded / np.max(np.abs(image_padded))) * 1.j
        grd_truths.append(x_true)


        # ring + i disk of Gaussians balls 

        image = imageio.imread('ring.png', mode='F')
        image = np.array(image)
        image = cv2.resize(image, ((self.Ny), (self.Nx)), interpolation=cv2.INTER_AREA)
        image_padded = np.pad(image, ( ((sNx)//2, (sNx)//2), ((sNy)//2, (sNy)//2)), 'constant')
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
        Imagedisk = imagedisk
        image = cv2.resize(Imagedisk, ((self.Ny), (self.Nx)), interpolation=cv2.INTER_AREA)
        imagedisk_padded = np.pad(image, ( ((sNx)//2, (sNx)//2), ((sNy)//2, (sNy)//2)), 'constant')

        x_true =  (imagedisk_padded / np.max(np.abs(imagedisk_padded))) + (image_padded / np.max(np.abs(image_padded))) * 0.j
        grd_truths.append(x_true)




        # cancer cell

        x_true = (0 + 0j) * np.zeros((self.Nx + sNx, self.Ny + sNy))
        img = Image.open('ISIC_0000004_cancer.jpg') # ISIC_0000004_cancer
        #img = iio.v2.imread('ISIC_0000004_cancer.jpg')
        #x_true = Image.rgb2gray(img)
        #print('true_support.shape', true_support.shape)
        x_true3 = np.array(img) #.resize((self.Nx, self.Ny))
        rect_sup = rect_support(x_true3[:, :,0])
        x_true3 = cv2.resize(x_true3, ((self.Ny), (self.Nx)), interpolation=cv2.INTER_AREA)
        #x_true3_imag = cv2.resize(x_true3.imag, (int(s * self.Nx), int(s * self.Ny)), interpolation=cv2.INTER_AREA
        #x_true = x_true/np.max(np.abs(x_true))
        x_true = np.pad( x_true3[:, :,0] + (1j) * x_true3[:, :,2] , ( ((sNx)//2, (sNx)//2), ((sNy)//2, (sNy)//2)), 'constant')
        # x_true[lowerX : upperX, lowerY : upperY] = x_true3[:, :,0] / np.max(np.abs(x_true3[:, :,0])) + (1j) * (x_true3[:, :,2]/ np.max(np.abs(x_true3[:, :,2])))# np.zeros((self.Nx,self.Ny))
        x_truecanc = x_true.real / np.max(np.abs(x_true.real)) + (x_true.imag / np.max(np.abs(x_true.imag))) * 1j
        #x_true *= mask 
        x_true = np.rot90(x_truecanc, -1)
        x_true = np.rot90(x_truecanc, -1)
        #x_true *= mask 
        img.save('resized_image.jpg')
        ma = cv2.resize(rect_sup, ((esNy), (esNx)), interpolation=cv2.INTER_AREA)
        mask_cancer = (0 + 0j) * np.zeros((self.Nx + sNx, self.Ny + sNy))
        mask_cancer[(self.Nx + sNx - esNx)//2  : (self.Nx + sNx - esNx)//2 + esNx, (self.Ny + sNy - esNy)//2  : (self.Ny + sNy - esNy)//2 + esNy] = ((1)*1 + (0) *1j) * ma
        
        grd_truths.append(x_truecanc)
        
        #complex cameraman

        x_true = (0 + 0j) * np.zeros((self.Nx + sNx,self.Ny + sNy))
        img = ski.img_as_float(ski.data.camera())
        img_res = ski.transform.resize(img, (self.Nx, self.Ny))
        img_res = cv2.resize(img_res, ((self.Ny), (self.Nx)), interpolation=cv2.INTER_AREA)
        x_true = np.pad( 1. * img_res + 1.j * (- img_res) , ( ((sNx)//2, (sNx)//2), ((sNy)//2, (sNy)//2)), 'constant')
        # x_true[lowerX : upperX, lowerY : upperY] = 1. * img_res + 1.j * (- img_res)
        x_true = x_true.real / np.max(np.abs(x_true.real)) + (x_true.imag / np.max(np.abs(x_true.imag))) * 1.j
        #x_true *= mask 
        # x_true = np.rot90(x_true, -1)
        # x_true = np.rot90(x_true, -1)
        grd_truths.append(x_true)

        #real cameraman

        x_true = (0 + 0j) * np.zeros((self.Nx + sNx, self.Ny + sNy))
        img = ski.img_as_float(ski.data.camera())
        img_res = ski.transform.resize(img, (self.Nx, self.Ny))
        img_res = cv2.resize(img_res, ((self.Ny), (self.Nx)), interpolation=cv2.INTER_AREA)
        x_true = np.pad( 1. * img_res + 0.j ,( ((sNx)//2, (sNx)//2), ((sNy)//2, (sNy)//2)), 'constant')
        # x_true[lowerX : upperX, lowerY : upperY] = 1. * img_res + 0.j# * (- img_res)
        x_true = x_true.real / np.max(np.abs(x_true.real)) + 0.j #* (x_true.imag / np.max(np.abs(x_true.imag))) * 
        #x_true *= mask 
        # x_true = np.rot90(x_true, -1)
        # x_true = np.rot90(x_true, -1)
        grd_truths.append(x_true)

        # cameraman

        x_truec = (0 + 0j) * np.zeros((self.Nx + sNx, self.Ny + sNy))
        image = imageio.imread('cameraman.png', mode='F')
        image = np.array(image)
        image = cv2.resize(image, ((self.Ny), (self.Nx)), interpolation=cv2.INTER_AREA)
        image_padded = np.pad(image, ( ((sNx)//2, (sNx)//2), ((sNy)//2, (sNy)//2)), 'constant')
        x_truec =  image_padded / np.max(np.abs(image_padded)) + (image_padded / np.max(np.abs(image_padded))) * 1.j
        # x_true = np.rot90(x_true, -1)
        # x_true = np.rot90(x_true, -1)
        grd_truths.append(x_truec)
        
        x_true[int(bord * (self.Nx + sNx)):int((1-bord)* (self.Nx + sNx )), int(bord * (self.Ny + sNy) ):int((1-bord)* (self.Ny + sNy)) ] = get_phantom(int((1-2*bord)*( self.Nx + sNx) )) + get_phantom(int( (1-2*bord)* ( self.Nx + sNx) )) * 1.j
        
        #complex shepp logan

        x_true = (0 + 0j) * np.zeros((self.Nx + sNx , self.Ny + sNy ))
        img = get_phantom(self.Nx + sNx )
        rect_sup = rect_support(img) #to define a box that is a tight to the object. Not just a rectangle tight on the heigth but not on the weith
        img_res = cv2.resize(img.real, ((self.Ny - 2), (self.Nx - 2)), interpolation=cv2.INTER_AREA)
        # img_res = cv2.resize(img.real, ((sNy - 34), (sNx - 34)), interpolation=cv2.INTER_AREA)
        # img_res = np.pad( img_res , ((16, 16), (16, 16)), mode='constant', constant_values=0) #+ 0.j * np.pad( (img_res) , ((1, 1), (1, 1)), mode='constant', constant_values=1)
        img_res = np.pad( img_res , ((1, 1), (1, 1)), mode='constant', constant_values=1) #+ 0.j * np.pad( (img_res) , ((1, 1), (1, 1)), mode='constant', constant_values=1)
        x_true = np.pad( 1. * img_res + 1.j * ( img_res) ,  ( ((sNx)//2, (sNx)//2), ((sNy)//2, (sNy)//2)), 'constant')
        # x_true[lowerX : upperX, lowerY : upperY] = 1. * img_res + 1.j * ( img_res)
        x_true = x_true.real / np.max(np.abs(x_true.real)) + (x_true.imag / np.max(np.abs(x_true.imag))) * 1.j
        # rect_sup = rect_support(x_true)
        ma = cv2.resize(rect_sup, ((esNy), (esNx)), interpolation=cv2.INTER_AREA)
        mask_shepp = (0 + 0j) * np.zeros((self.Nx + sNx, self.Ny + sNy))
        mask_shepp[(self.Nx + sNx - esNx)//2  : (self.Nx + sNx - esNx)//2 + esNx, (self.Ny + sNy  - esNy)//2  : (self.Ny + sNy - esNy)//2 + esNy] = ((1)*1 + (0) *1j) * ma

        grd_truths.append(x_true)


        #real shepp logan

        x_true = (0 + 0j) * np.zeros((self.Nx + sNx , self.Ny + sNy))
        img = get_phantom(self.Nx + sNx)
        img_res = cv2.resize(img.real, ((self.Ny), (self.Nx)), interpolation=cv2.INTER_AREA)
        x_true = np.pad( 1. * img_res + 1.j * ( img_res) ,  ( ((sNx)//2, (sNx)//2), ((sNy)//2, (sNy)//2)), 'constant')
        # x_true[lowerX : upperX, lowerY : upperY] = 1. * img_res + 0.j * ( img_res)
        x_true = x_true.real / np.max(np.abs(x_true.real)) + 1.j * (x_true.imag / np.max(np.abs(x_true.imag))) 
        # get exact image support
        exact_mask_shepp = exact_support(x_true)
        grd_truths.append(x_true)

        extended_mask_shepp = extend_support_with_holes(exact_mask_shepp, dilation_radius=2)
        
        #shepp + 1j * (cancer * support_shepp) : real and imaginary partshave same support
        imag = x_truecanc.copy()
        imag[x_true == 0] = 0 
        grd_truths.append(x_true.real + 1j * imag.imag)

        #shepp logan + 1j * cameraman
        
        x_true = x_true.real / np.max(np.abs(x_true.real)) + 1.j * (x_truec.imag / np.max(np.abs(x_truec.imag))) 
        grd_truths.append(x_true)

        #cameraman + i barbara

        x_true = (0 + 0j) * np.zeros((self.Nx + sNx, self.Ny + sNy ))
        image = scipy.datasets.ascent().astype('complex').reshape((512, 512)) #resize((int((1-2*bord)*self.Nx)*int((1-2*bord)*self.Ny))) #.
        image = imageio.imread('barbara.jpg', mode='F')
        img_res = cv2.resize(image.real, ((self.Ny), (self.Nx)), interpolation=cv2.INTER_AREA)
        x_true = np.pad( -1. * img_res + 1.j * ( + img_res) , ( ((sNx)//2, (sNx)//2), ((sNy)//2, (sNy)//2)), 'constant')
        # x_true[lowerX : upperX, lowerY : upperY] = 1. * img_res + 1.j * (- img_res)
        image = x_truec.real / np.max(np.abs(x_truec.real)) + (x_true.imag / np.max(np.abs(x_true.imag))) * 1.j
        # image = np.rot90(image, 1)
        # image = np.rot90(image, 1)
        grd_truths.append(image)

        # cameraman + i astronaut

        x_true = (0 + 0j) * np.zeros((self.Nx + sNx,self.Ny + sNy))
        image = rgb2gray( ski.data.astronaut() ).reshape((512, 512)) #resize((int((1-2*bord)*self.Nx)*int((1-2*bord)*self.Ny))) #.
        img_res = cv2.resize(image.real, ((self.Ny), (self.Nx)), interpolation=cv2.INTER_AREA)
        x_true = np.pad( -1. * img_res + 1.j * ( + img_res) ,  ( ((sNx)//2, (sNx)//2), ((sNy)//2, (sNy)//2)), 'constant')
        # x_true[lowerX : upperX, lowerY : upperY] = 1. * img_res + 1.j * (- img_res)
        image = x_truec.real / np.max(np.abs(x_truec.real)) + (x_true.imag / np.max(np.abs(x_true.imag))) * 1.j
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
        im1 = (1 + 0.j) * np.zeros((self.Nx, self.Ny))
        im1[0, 0] = 0. + 1j
        im1 = np.fft.fftshift(im1)
        #self.im1 = im1
        x_true = np.pad( 1. * im1 + 0.j * (im1) ,  ( ((sNx)//2, (sNx)//2), ((sNy)//2, (sNy)//2)), 'constant')
        # x_true[lowerX : upperX, lowerY : upperY] = 1. * im1 + 0.j * (im1)
        # x_true = np.fft.ifftshift(x_true)
        grd_truths.append(x_true)
        
        # zero image

        x_true = (0 + 0j) * np.zeros((self.Nx + sNx , self.Ny + sNy))
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
        # cameraman + i ascent
        
        x_true = (0 + 0j) * np.zeros((self.Nx + sNx , self.Ny + sNy))
        image = scipy.datasets.ascent().astype('complex').reshape((512, 512)) #resize((int((1-2*bord)*self.Nx)*int((1-2*bord)*self.Ny))) #.
        # image = imageio.imread('barbara.jpg', mode='F')
        img_res = cv2.resize(image.real, ((self.Ny), (self.Nx)), interpolation=cv2.INTER_AREA)
        x_true = np.pad( -1. * img_res + 1.j * ( + img_res) ,  ( ((sNx)//2, (sNx)//2), ((sNy)//2, (sNy)//2)), 'constant')
        # x_true[lowerX : upperX, lowerY : upperY] = 1. * img_res + 1.j * (- img_res)
        image = x_truec.real / np.max(np.abs(x_truec.real)) + (x_true.imag / np.max(np.abs(x_true.imag))) * 1.j
        # image = np.rot90(image, 1)
        # image = np.rot90(image, 1)
        grd_truths.append(image)
        
        self.mask = mask
        self.grd_truths = grd_truths
        return grd_truths, mask, mask_shepp,exact_mask_shepp, extended_mask_shepp,  mask_cancer #
        ##plt.imshow(-1j * x_true)
        #plt.colorbar()    

