import numpy as np
import matplotlib.pyplot as plt
import odl
import math
import scipy

def phase_retrie_plots(idx, grd_truths, X_sols, map, A, meas, maxiter, space, Algos):

    
    op = odl.FlatteningOperator(space)

    fig = plt.figure(figsize=(32,32))
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
    m = len((A(x.flatten()).flatten()))
    im00 = axs00.imshow(x.real, cmap='gray', vmin=np.min(x.real), vmax=np.max(x.real)) #, origin='lower'
    axs00.axis('off')
    axs00.set_title('Real Part-Ground Truth')
    plt.colorbar(im00, ax = axs00)
    im10 = axs10.imshow(x.imag, cmap='gray', vmin = np.min(x.imag), vmax = np.max(x.imag)) #, origin='lower'
    axs10.axis('off')
    axs10.set_title('Imaginary Part-Ground Truth')
    plt.colorbar(im10, ax = axs10)
    
    x = op.inverse(X_sols[0][-1])#.reshape(Nx, Ny)
    im01 = axs01.imshow(x.real, cmap='gray', vmin=np.min(x.real), vmax=np.max(x.real)) #, origin='lower'
    axs01.axis('off')
    axs01.set_title('Real Part - '+str(Algos[0])) # No Object Domain Constraints
    plt.colorbar(im01, ax = axs01)
    im11 = axs11.imshow(x.imag, cmap='gray', vmin = np.min(x.imag), vmax = np.max(x.imag)) #, origin='lower'
    axs11.axis('off')
    axs11.set_title('Imaginary Part - '+str(Algos[0]))# No Object Domain Constraints
    plt.colorbar(im11, ax = axs11)
    f_x_sols  = [map.f(x) for x in X_sols[0]]
    #FMPE0 = [((meas - np.linalg.norm(A(x).flatten())**2)**(1))/((2 * np.sqrt(m))**(1)) for x in X_sols[0]]
    axs21.plot(np.arange(maxiter+1), [((meas - np.abs(A(x).flatten())**2)**(1))/((2 * np.sqrt(m))**(1)) for x in X_sols[0]]) #np.sum(((A(x)*np.conjugate(A(x))) + np.log(math.factorial(meas)) - meas * np.log((A(x)*np.conjugate(A(x)))))) 
    axs21.set_title('Fourier Magnitude Pixels Error')
    K = np.arange(len(f_x_sols))
    axs31.loglog(K, f_x_sols)
    axs31.set_title('Fourier error decay')
    x_tru = grd_truths[idx]
    axs41.loglog(K, [np.linalg.norm(x-x_tru.flatten()) for x in X_sols[0]])
    axs41.set_title('Object domain error')
   
    x = op.inverse(X_sols[1][-1]) #.reshape(Nx, Ny)
    im02 = axs02.imshow(x.real, cmap='gray', vmin=np.min(x.real), vmax=np.max(x.real)) #, origin='lower'
    axs02.axis('off')
    axs02.set_title('Real Part - '+str(Algos[1])) # Gerchberg-Saxton
    #axs02.set_title('Real Part-MD')
    plt.colorbar(im02, ax = axs02)
    im12 = axs12.imshow(x.imag, cmap='gray', vmin = np.min(x.imag), vmax = np.max(x.imag)) #, origin='lower'
    axs12.axis('off')
    axs12.set_title('Imaginary Part - '+str(Algos[1])) #Gerchberg-Saxton
    #axs12.set_title('Imaginary Part-MD')
    plt.colorbar(im12, ax = axs12)
    f_x_sols  = [map.f(x)  for x in X_sols[1]] #Poisson likelihood np.sum(((A(x)*np.conjugate(A(x))) + np.log(scipy.special.gamma(meas + np.ones_like(meas))) - meas * np.log((A(x)*np.conjugate(A(x))))))
    #FMPE1 = [((meas - np.linalg.norm(A(x).flatten())**2)**(1))/((2 * np.sqrt(m))**(1)) for x in X_sols[1]]
    axs22.plot((np.arange(maxiter+1)), [((meas - np.abs(A(x).flatten())**2)**(1))/((2 * np.sqrt(m))**(1)) for x in X_sols[1]]) #Poisson likelihood ((A(x)*np.conjugate(A(x))) + np.log(scipy.special.gamma(meas + np.ones_like(meas))) - meas * np.log((A(x)*np.conjugate(A(x)))))
    axs22.set_title('Fourier Magnitude Pixels Error')
    K = np.arange(len(f_x_sols))
    axs32.loglog(K, f_x_sols)
    axs32.set_title('Fourier error decay')
    x_tru = grd_truths[idx]
    axs42.loglog(K, [np.linalg.norm(x-x_tru.flatten()) for x in X_sols[1]])
    axs42.set_title('Object domain error')
    
    x = op.inverse(X_sols[2][-1]) #.reshape(Nx, Ny)
    im03 = axs03.imshow(x.real, cmap='gray', vmin=np.min(x.real), vmax=np.max(x.real)) #, origin='lower'
    axs03.axis('off')
    axs03.set_title('Real Part - '+str(Algos[2])) #ER
    #axs03.set_title('Real Part-CMD')
    plt.colorbar(im03, ax = axs03)
    im13 = axs13.imshow(x.imag, cmap='gray', vmin = np.min(x.imag), vmax = np.max(x.imag)) #, origin='lower'
    axs13.axis('off')
    axs13.set_title('Imaginary Part - '+str(Algos[2])) #ER
    #axs13.set_title('Imaginary Part-CMD')
    plt.colorbar(im13, ax = axs13)
    f_x_sols  = [map.f(x) for x in X_sols[2]]
    #FMPE2 = [((meas - np.linalg.norm(A(x).flatten())**2)**(1))/((2 * np.sqrt(m))**(1)) for x in X_sols[2]]
    axs23.plot(np.arange(maxiter+1), [((meas - np.abs(A(x).flatten())**2)**(1))/((2 * np.sqrt(m))**(1)) for x in X_sols[2]])
    axs23.set_title('Fourier Magnitude Pixels Error')
    K = np.arange(len(f_x_sols))
    axs33.loglog(K, f_x_sols)
    axs33.set_title('Fourier error decay')
    x_tru = grd_truths[idx]
    axs43.loglog(K, [np.linalg.norm(x-x_tru.flatten()) for x in X_sols[2]])
    axs43.set_title('Object domain error')
    #return FMPE0, FMPE1, FMPE2 
    #'''