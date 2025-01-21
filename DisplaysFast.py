import numpy as np
import matplotlib.pyplot as plt
import odl
import math
import scipy

def phase_retrie_plots(idx, grd_truths, X_sols, map, A, meas, maxiter, space, Algos):
    op = odl.FlatteningOperator(space)

    fig, axs = plt.subplots(5, 4, figsize=(32, 32))
    axs = axs.flatten()
    
    x = grd_truths[idx]
    Nx, Ny = x.shape
    m = len(A(x.flatten()).flatten())
    
    # Helper function to plot images
    def plot_image(ax, img, title, cmap='gray', vmin=None, vmax=None):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis('off')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    
    plot_image(axs[0], x.real, 'Real Part-Ground Truth', vmin=np.min(x.real), vmax=np.max(x.real))
    plot_image(axs[4], x.imag, 'Imaginary Part-Ground Truth', vmin=np.min(x.imag), vmax=np.max(x.imag))
    
    # Loop over test images and the algorithms 
    for i, algo in enumerate(Algos):
        x_sol = op.inverse(X_sols[i][-1])
        plot_image(axs[i + 1], x_sol.real, f'Real Part - {algo}', vmin=np.min(x_sol.real), vmax=np.max(x_sol.real))
        plot_image(axs[i + 5], x_sol.imag, f'Imaginary Part - {algo}', vmin=np.min(x_sol.imag), vmax=np.max(x_sol.imag))
        
        #correct global phase : close form of the 1D optimizer
        lambd = np.vdot( X_sols[i][-1], x)
        phi = np.arctan(lambd.imag/lambd.real)
        X_sols[i][-1] = np.exp(phi * 1.j) * X_sols[i][-1]

        #compute errors
        f_x_sols = [map.f(x) for x in X_sols[i]]
        axs[i + 9].plot(np.arange(maxiter + 1), [(meas - np.abs(A(x).flatten())**2) / (2 * np.sqrt(m)) for x in X_sols[i]])
        axs[i + 9].set_title('Fourier Magnitude Pixels Error')
        
        K = np.arange(len(f_x_sols))
        axs[i + 13].loglog(K, f_x_sols)
        axs[i + 13].set_title('Fourier error decay')
        
        axs[i + 17].loglog(K, [np.linalg.norm(x - grd_truths[idx].flatten()) for x in X_sols[i]])
        axs[i + 17].set_title('Object domain error')

    plt.tight_layout()
    plt.show()
