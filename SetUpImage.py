import numpy as np
from PIL import Image
import skimage as ski
import matplotlib.pyplot as plt

def get_phantom(dim):
    phantom = ski.img_as_float(ski.data.shepp_logan_phantom())
    return ski.transform.resize(phantom, (dim, dim))

def setUpImage(Nx, Ny):
    
    NumPix = 0 + np.floor(0.5 * (1-2**(-1)) *Nx)
    bord = NumPix/Nx
    #x_true = im[500:516, 500:516]
    mask = (0 + 0j) * np.zeros((Nx,Ny))
    mask[int(bord * Nx):int((1-bord)*Nx),int(bord * Ny):int((1-bord)*Ny)] = ((1)*1 + (0) *1j) * np.ones((int((1-2*bord)*Nx),int((1-2*bord)*Ny)))
    x_true = (0 + 0j) * np.zeros((Nx,Ny))

    i, k = np.meshgrid(np.arange(int(Nx/2)), np.arange(int(Ny/2)))
    #omega = np.exp( - 2 * np.pi * 1j /int(Nx/2) * int(Ny/2) ) # (i + k *0j) *  #i+k**2-k*i   #k + i**2 - i*k
    grd_truths = []
    x_true[int(bord * Nx):int((1-bord)*Nx),int(bord * Ny):int((1-bord)*Ny)] = ((1)*1 + (1) *1j) * np.ones((int((1-2*bord)*Nx),int((1-2*bord)*Ny))) #(7 + 0j)* np.random.normal(0, 1, size = (int(Nx/2),int(Ny/2))) + (0 + 5j)* np.random.normal(0, 1, size = (int(Nx/2),int(Ny/2))) #(i + k *1j) * np.ones((int(Nx/2),int(Ny/2)))
    x_true = x_true / np.max(np.abs(x_true))
    grd_truths.append(x_true)
    #x_true[1,1] = 2 + 1j
    #print(x_true)
    #x_true[0, 1:Ny-1] = np.ones((1,Ny-2))
    img = Image.open('ISIC_0000004_cancer.jpg')
    #img = iio.v2.imread('ISIC_0000004_cancer.jpg')
    #x_true = Image.rgb2gray(img)
    x_true3 = np.array( img.resize((Nx, Ny)))
    #x_true = x_true/np.max(np.abs(x_true))
    x_true = x_true3[:, :,0] + (1j) * x_true3[:, :,2]# np.zeros((Nx,Ny))
    x_true = x_true / np.max(np.abs(x_true))
    x_true *= mask 
    grd_truths.append(x_true)
    #x_true *= mask 
    img.save('resized_image.jpg')
    #print(x_true[0])
    #print(np.shape(x_true))
    #x_true = img
    #plt.imshow(img)
    img = ski.img_as_float(ski.data.camera())
    img_res = ski.transform.resize(img, (Nx, Ny))
    x_true = 2. * img_res + 2.j * (- img_res)
    x_true = x_true / np.max(np.abs(x_true))
    
    #x_true = (1 + 1j) * np.ones(x_true.shape)
    x_true *= mask 
    grd_truths.append(x_true)

    x_true[int(bord * Nx):int((1-bord)*Nx),int(bord * Ny):int((1-bord)*Ny)] = get_phantom(int((1-2*bord)*Nx)) + get_phantom(int((1-2*bord)*Nx)) * 1.j
    x_true *= mask 
    grd_truths.append(x_true)
    '''
    plt.imshow(x_true.real, cmap='gray')
    plt.colorbar()
    '''
    x_true_real = np.real(x_true)#.reshape((Nx, Ny))
    x_true_imag = np.imag(x_true)#.reshape((Nx, Ny))
    #print(x_true_real)
    #x_true = x_true_real
    '''
    fig, ((ax00, ax01, ax02, ax03), (ax10, ax11, ax12, ax13)) = plt.subplots(nrows = 2, ncols=4, figsize=(10,5))
    
    ax00.imshow(x_true_imag, cmap='gray') 
    ax00.set_title("image ")
    ax00.axis('off')
    '''

    return grd_truths, mask
    ##plt.imshow(-1j * x_true)
    #plt.colorbar()