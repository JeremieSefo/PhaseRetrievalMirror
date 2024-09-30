import numpy as np
import matplotlib.pyplot as plt

def phase_retrie_plots(idx, grd_truths, X_sols, map, A, meas, maxiter):
    
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
    Nx, Ny = x.shape
    im00 = axs00.imshow(x.real, cmap='gray', vmin=np.min(x.real), vmax=np.max(x.real))
    axs00.axis('off')
    axs00.set_title('Real Part-Ground Truth')
    plt.colorbar(im00, ax = axs00)
    im10 = axs10.imshow(x.imag, cmap='gray', vmin = np.min(x.imag), vmax = np.max(x.imag))
    axs10.axis('off')
    axs10.set_title('Imaginary Part-Ground Truth')
    plt.colorbar(im10, ax = axs10)

    x = X_sols[0][-1].reshape(Nx, Ny)
    im01 = axs01.imshow(x.real, cmap='gray', vmin=np.min(x.real), vmax=np.max(x.real))
    axs01.axis('off')
    axs01.set_title('Real Part-FIENUP')
    plt.colorbar(im01, ax = axs01)
    im11 = axs11.imshow(x.imag, cmap='gray', vmin = np.min(x.imag), vmax = np.max(x.imag))
    axs11.axis('off')
    axs11.set_title('Imaginary Part-FIENUP')
    plt.colorbar(im11, ax = axs11)

    #f_x_sols  = [map.f(x) for x in X_sols[1]]
    axs21.plot(np.arange(maxiter+1), [meas - np.linalg.norm(A(x))**2 for x in X_sols[0]])

    x = X_sols[1][-1].reshape(Nx, Ny)
    im02 = axs02.imshow(x.real, cmap='gray', vmin=np.min(x.real), vmax=np.max(x.real))
    axs02.axis('off')
    axs02.set_title('Real Part-MD')
    plt.colorbar(im02, ax = axs02)
    im12 = axs12.imshow(x.imag, cmap='gray', vmin = np.min(x.imag), vmax = np.max(x.imag))
    axs12.axis('off')
    axs12.set_title('Imaginary Part-MD')
    plt.colorbar(im12, ax = axs12)

    axs22.plot(np.arange(maxiter+1), [meas - np.linalg.norm(A(x))**2 for x in X_sols[1]])

    x = X_sols[2][-1].reshape(Nx, Ny)
    im03 = axs03.imshow(x.real, cmap='gray', vmin=np.min(x.real), vmax=np.max(x.real))
    axs03.axis('off')
    axs03.set_title('Real Part-CMD')
    plt.colorbar(im03, ax = axs03)
    im13 = axs13.imshow(x.imag, cmap='gray', vmin = np.min(x.imag), vmax = np.max(x.imag))
    axs13.axis('off')
    axs13.set_title('Imaginary Part-CMD')
    plt.colorbar(im13, ax = axs13)

    axs23.plot(np.arange(maxiter+1), [meas - np.linalg.norm(A(x))**2 for x in X_sols[2]])