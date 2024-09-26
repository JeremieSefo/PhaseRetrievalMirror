import numpy as np
import matplotlib.pyplot as plt

def phase_retrie_plots(idx, image):
    
    fig = plt.figure(figsize=(28,12))
    axs00 = plt.subplot2grid((5,4), (0,0))
    axs01 = plt.subplot2grid((5,4), (0,1))
    axs02 = plt.subplot2grid((5,4), (0,2))
    axs03 = plt.subplot2grid((5,4), (0,3)) 
    axs10 = plt.subplot2grid((5,4), (1,0))
    axs11 = plt.subplot2grid((5,4), (1,1))
    axs12 = plt.subplot2grid((5,4), (1,2))
    axs13 = plt.subplot2grid((5,4), (1,3)) 
    axs21 = plt.subplot2grid((5,4), (2,1))
    axs22 = plt.subplot2grid((5,4), (2,2))
    axs23 = plt.subplot2grid((5,4), (2,3))
    axs31 = plt.subplot2grid((5,4), (3,1))
    axs32 = plt.subplot2grid((5,4), (3,2))
    axs33 = plt.subplot2grid((5,4), (3,3))
    axs41 = plt.subplot2grid((5,4), (4,1))
    axs42 = plt.subplot2grid((5,4), (4,2))
    axs43 = plt.subplot2grid((5,4), (4,3))
    x = grd_truths[idx]
    im00 = axs00.imshow(x.real, cmap='gray', vmin=np.min(x.real), vmax=np.max(x.real))
    axs00.axis('off')
    axs00.set_title('Real Part-Ground Truth')
    plt.colorbar(im00, ax = axs00)
    im10 = axs10.imshow(x.imag, cmap='gray', vmin = np.min(x.imag), vmax = np.max(x.imag))
    axs10.axis('off')
    axs10.set_title('Imaginary Part-Ground Truth')
    plt.colorbar(im10, ax = axs10)