import numpy as np
import matplotlib.pyplot as plt
import odl
import math
import scipy

def phase_retrie_plots(idx, grd_truths, X_sols, map, A, meas, maxiter, space, Algos):
    op = odl.FlatteningOperator(space)

    fig, axs = plt.subplots(5, 4, figsize=(32, 32))
    axs = axs.flatten()
    axs[8].axis('off')
    axs[12].axis('off')
    axs[16].axis('off')
    
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
    f_X_sols = []
    Err = []
    # Loop over test images and the algorithms 
    for i, algo in enumerate(Algos):
        
        # #correct global phase : close form of the 1D optimizer
        # lambd = np.vdot( X_sols[i][-1], x)
        # phi = np.arctan(lambd.imag/lambd.real)
        # X_sols[i][-1] = np.exp(phi * 1.j) * X_sols[i][-1]

        x_sol = op.inverse(X_sols[i][-1])
        plot_image(axs[i + 1], x_sol.real, f'Real Part - {algo}', vmin=np.min(x_sol.real), vmax=np.max(x_sol.real))
        plot_image(axs[i + 5], x_sol.imag, f'Imaginary Part - {algo}', vmin=np.min(x_sol.imag), vmax=np.max(x_sol.imag))
        
        #compute errors
        f_x_sols = [map.f(x) for x in X_sols[i]]
        f_X_sols.append(f_x_sols)
        axs[i + 13].plot(np.arange(maxiter + 1), [(meas - np.abs(A(x).flatten())**2) / (2 * np.sqrt(m)) for x in X_sols[i]])
        axs[i + 13].set_title('Fourier magnitude pixels error')
        axs[i + 13].set_xlabel('Iterations')
        axs[i + 13].set_ylabel('Fourier magnitude error for different frequencies ')
        
        K = np.arange(1, len(f_x_sols) + 1)
        axs[i + 9].loglog(K, f_x_sols)
        axs[i + 9].set_title('total Fourier error')
        axs[i + 9].set_xlabel('Iterations')
        axs[i + 9].set_ylabel('Fourier error')

        err = [np.linalg.norm(x - grd_truths[idx].flatten()) for x in X_sols[i]]
        Err.append(err)
        axs[i + 17].loglog(K, err)
        axs[i + 17].set_title('Error in object domain')
        axs[i + 17].set_xlabel('Iterations')
        axs[i + 17].set_ylabel('L2-error')

    plt.tight_layout()
    plt.show()

def phase_retrie_plots_objects(idx, grd_truths, X_sols, space, Algos):
    op = odl.FlatteningOperator(space)
    
    fig = plt.figure(figsize=(32, 32))
    
    x = grd_truths[idx]
    Nx, Ny = x.shape
    # m = len(A(x.flatten()).flatten())
    
    # plot_image(axs[0], x.real, 'Real Part-Ground Truth', vmin=np.min(x.real), vmax=np.max(x.real))
    # plot_image(axs[4], x.imag, 'Imaginary Part-Ground Truth', vmin=np.min(x.imag), vmax=np.max(x.imag))
    N = len(Algos) + 1
    ax = fig.add_subplot(5, N, 1)
    im = ax.imshow(x.real, vmin=np.min(x.real), vmax=np.max(x.real), interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title('Real Part-Ground Truth', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax1 = fig.add_subplot(5, N, 1 + N)
    im = ax1.imshow(x.imag, vmin=np.min(x.imag), vmax=np.max(x.imag), interpolation="nearest", cmap=plt.cm.gray)
    ax1.set_title('Imaginary Part-Ground Truth', fontsize=10)
    ax1.axis('off')
    plt.colorbar(im, ax=ax1)

    # f_X_sols = []
    # Err = []

    # Loop over test images and the algorithms 
    for i, algo in enumerate(Algos):
        x_sol = op.inverse(X_sols[i][-1])
        # #correct global phase : close form of the 1D optimizer
        # lambd = np.vdot( X_sols[i][-1], x)
        # phi = np.arctan(lambd.imag/lambd.real)
        # X_sols[i][-1] = np.exp(phi * 1.j) * X_sols[i][-1]
        ax = fig.add_subplot(5, N, i + 2)
        im = ax.imshow(x_sol.real, vmin=np.min(x_sol.real), vmax=np.max(x_sol.real), interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(f'Real Part - {algo}', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax1 = fig.add_subplot(5, N, i + 2 + N)
        im = ax1.imshow(x_sol.imag, vmin=np.min(x_sol.imag), vmax=np.max(x_sol.imag), interpolation="nearest", cmap=plt.cm.gray)
        ax1.set_title(f'Imaginary Part - {algo}', fontsize=10)
        ax1.axis('off')
        plt.colorbar(im, ax=ax1)

        # #compute errors
        # f_x_sols = [map.f(x) for x in X_sols[i]]
        # f_X_sols.append(f_x_sols)
        # K = np.arange(len(f_x_sols))

        # ax = fig.add_subplot(5, N, i + 2 + 2*N)
        # ax.loglog(K, f_x_sols)
        # ax.set_title('total Fourier error')
        # ax.set_xlabel('Iterations')
        # ax.set_ylabel('Fourier error')
        
        # ax = fig.add_subplot(5, N, i + 2 + 3*N)
        # ax.plot(np.arange(maxiter + 1), [(meas - np.abs(A(x).flatten())**2) / (2 * np.sqrt(m)) for x in X_sols[i]])
        # ax.set_title('Fourier magnitude pixels error')
        # ax.set_xlabel('Iterations')
        # ax.set_ylabel('Fourier magnitude error for different frequencies ')


        # err = [np.linalg.norm(x - grd_truths[idx].flatten()) for x in X_sols[i]]
        # Err.append(err)
        
        # ax = fig.add_subplot(5, N, i + 2 + 4*N)
        # ax.loglog(K, err)
        # ax.set_title('Error in object domain')
        # ax.set_xlabel('Iterations')
        # ax.set_ylabel('L2-error')


    plt.tight_layout()
    plt.show()
    # return f_X_sols, Err


def phase_retrie_plots_desk(idx, grd_truths, X_sols, map, A, meas, maxiter, space, Algos):
    op = odl.FlatteningOperator(space)

    fig = plt.figure(figsize=(32, 32))
    #axs = axs.flatten()
    #axs[8].axis('off')
    #axs[12].axis('off')
    #axs[16].axis('off')
    
    x = grd_truths[idx]
    Nx, Ny = x.shape
    m = len(A(x.flatten()).flatten())
    
    # Helper function to plot images
    def plot_image(ax, img, title, cmap='gray', vmin=None, vmax=None):
        im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis('off')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    
    # plot_image(axs[0], x.real, 'Real Part-Ground Truth', vmin=np.min(x.real), vmax=np.max(x.real))
    # plot_image(axs[4], x.imag, 'Imaginary Part-Ground Truth', vmin=np.min(x.imag), vmax=np.max(x.imag))
    N = len(Algos) + 1
    ax = fig.add_subplot(5, N, 1)
    im = ax.imshow(x.real, vmin=np.min(x.real), vmax=np.max(x.real), interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title('Real Part-Ground Truth', fontsize=10)
    ax.axis('off')
    plt.colorbar(im, ax=ax)
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax1 = fig.add_subplot(5, N, 1 + N)
    im = ax1.imshow(x.imag, vmin=np.min(x.imag), vmax=np.max(x.imag), interpolation="nearest", cmap=plt.cm.gray)
    ax1.set_title('Imaginary Part-Ground Truth', fontsize=10)
    ax1.axis('off')
    plt.colorbar(im, ax=ax1)

    f_X_sols = []
    Err = []

    # Loop over test images and the algorithms 
    for i, algo in enumerate(Algos):
        x_sol = op.inverse(X_sols[i][-1])
        # #correct global phase : close form of the 1D optimizer
        # lambd = np.vdot( X_sols[i][-1], x)
        # phi = np.arctan(lambd.imag/lambd.real)
        # X_sols[i][-1] = np.exp(phi * 1.j) * X_sols[i][-1]
        ax = fig.add_subplot(5, N, i + 2)
        im = ax.imshow(x_sol.real, vmin=np.min(x_sol.real), vmax=np.max(x_sol.real), interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(f'Real Part - {algo}', fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax)
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax1 = fig.add_subplot(5, N, i + 2 + N)
        im = ax1.imshow(x_sol.imag, vmin=np.min(x_sol.imag), vmax=np.max(x_sol.imag), interpolation="nearest", cmap=plt.cm.gray)
        ax1.set_title(f'Imaginary Part - {algo}', fontsize=10)
        ax1.axis('off')
        plt.colorbar(im, ax=ax1)

        # x_sol = op.inverse(X_sols[i][-1])
        # plot_image(axs[i + 1], x_sol.real, f'Real Part - {algo}', vmin=np.min(x_sol.real), vmax=np.max(x_sol.real))
        # plot_image(axs[i + 5], x_sol.imag, f'Imaginary Part - {algo}', vmin=np.min(x_sol.imag), vmax=np.max(x_sol.imag))
        
        #compute errors
        f_x_sols = [map.f(x) for x in X_sols[i]]
        f_X_sols.append(f_x_sols)
        K = np.arange(1, len(f_x_sols)+1)

        ax = fig.add_subplot(5, N, i + 2 + 2*N)
        ax.loglog(K, f_x_sols)
        ax.set_title('total Fourier error')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Fourier error')
        
        ax = fig.add_subplot(5, N, i + 2 + 3*N)
        ax.plot(np.arange(maxiter + 1), [(meas - np.abs(A(x).flatten())**2) / (2 * np.sqrt(m)) for x in X_sols[i]])
        ax.set_title('Fourier magnitude pixels error')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Fourier magnitude error for different frequencies ')


        err = [np.linalg.norm(x - grd_truths[idx].flatten()) for x in X_sols[i]]
        Err.append(err)
        
        ax = fig.add_subplot(5, N, i + 2 + 4*N)
        ax.loglog(K, err)
        ax.set_title('Error in object domain')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('L2-error')


    plt.tight_layout()
    plt.show()
    return f_X_sols, Err

def P_S(x, mask):
    n = len(x)
    x = x.copy()
    indices = np.logical_not(mask.reshape((n,)))
    x[indices] = 0
    return x 

#correct global phase : close form of the 1D optimizer
def correct_phase(x, x_true):
    lambd = np.vdot(x, x_true)
    phi = np.arctan(lambd.imag/lambd.real)
    return np.exp(phi * 1.j) * x

def phase_retrie_plots_noPlot(idx, grd_truths, X_sols, map, A, meas, maxiter, space, Algos, mask):
    op = odl.FlatteningOperator(space)

    # fig = plt.figure(figsize=(32, 32))
    
    x_true = grd_truths[idx]
    # Nx, Ny = x.shape
    # m = len(A(x.flatten()).flatten())

    # N = len(Algos) + 1
    # ax = fig.add_subplot(5, N, 1)
    # im = ax.imshow(x.real, vmin=np.min(x.real), vmax=np.max(x.real), interpolation="nearest", cmap=plt.cm.gray)
    # ax.set_title('Real Part-Ground Truth', fontsize=10)
    # ax.axis('off')
    # plt.colorbar(im, ax=ax)

    # ax1 = fig.add_subplot(5, N, 1 + N)
    # im = ax1.imshow(x.imag, vmin=np.min(x.imag), vmax=np.max(x.imag), interpolation="nearest", cmap=plt.cm.gray)
    # ax1.set_title('Imaginary Part-Ground Truth', fontsize=10)
    # ax1.axis('off')
    # plt.colorbar(im, ax=ax1)

    f_X_sols = []
    RR = []
    Err = []
    RE = []

    # Loop over test images and the algorithms 
    for i, algo in enumerate(Algos):
        # x_sol = op.inverse(X_sols[i][-1])
        # #correct global phase : close form of the 1D optimizer
        # lambd = np.vdot( X_sols[i][-1], x)
        # phi = np.arctan(lambd.imag/lambd.real)
        # X_sols[i][-1] = np.exp(phi * 1.j) * X_sols[i][-1]
        
        # ax = fig.add_subplot(5, N, i + 2)
        # im = ax.imshow(x_sol.real, vmin=np.min(x_sol.real), vmax=np.max(x_sol.real), interpolation="nearest", cmap=plt.cm.gray)
        # ax.set_title(f'Real Part - {algo}', fontsize=10)
        # ax.axis('off')
        # plt.colorbar(im, ax=ax)

        # ax1 = fig.add_subplot(5, N, i + 2 + N)
        # im = ax1.imshow(x_sol.imag, vmin=np.min(x_sol.imag), vmax=np.max(x_sol.imag), interpolation="nearest", cmap=plt.cm.gray)
        # ax1.set_title(f'Imaginary Part - {algo}', fontsize=10)
        # ax1.axis('off')
        # plt.colorbar(im, ax=ax1)

        #compute errors
        f_x_sols = [map.f(P_S(xt, mask)) for xt in X_sols[i]]
        o = np.zeros(x_true.flatten().shape)
        rr = [e / map.f(o) for e in f_x_sols]
        f_X_sols.append(f_x_sols)
        RR.append(rr)
        
        # K = np.arange(len(f_x_sols))

        # ax = fig.add_subplot(5, N, i + 2 + 2*N)
        # ax.loglog(K, f_x_sols)
        # ax.set_title('total Fourier error')
        # ax.set_xlabel('Iterations')
        # ax.set_ylabel('Fourier error')
        
        # ax = fig.add_subplot(5, N, i + 2 + 3*N)
        # ax.plot(np.arange(maxiter + 1), [(meas - np.abs(A(x).flatten())**2) / (2 * np.sqrt(m)) for x in X_sols[i]])
        # ax.set_title('Fourier magnitude pixels error')
        # ax.set_xlabel('Iterations')
        # ax.set_ylabel('Fourier magnitude error for different frequencies ')


        err = [np.linalg.norm(correct_phase(P_S(xt, mask).reshape(x_true.shape), x_true) - x_true) for xt in X_sols[i]]
        re = [e / np.linalg.norm(x_true) for e in err]
        # err = [np.linalg.norm(xt - x_true.flatten()) for xt in X_sols[i]]
        Err.append(err)
        RE.append(re)
        
        # ax = fig.add_subplot(5, N, i + 2 + 4*N)
        # ax.loglog(K, err)
        # ax.set_title('Error in object domain')
        # ax.set_xlabel('Iterations')
        # ax.set_ylabel('L2-error')


    # plt.tight_layout()
    # plt.show()
    return f_X_sols, RR, Err, RE
def phase_retrie_plots_noPlot_full_error(idx, grd_truths, X_sols, map, A, meas, maxiter, space, Algos, mask):
    op = odl.FlatteningOperator(space)
  
    x_true = grd_truths[idx]

    f_X_sols = []
    RR = []
    Err = []
    RE = []

    # Loop over test images and the algorithms 
    for i, algo in enumerate(Algos):

        #compute errors
        f_x_sols = [map.f(xt) for xt in X_sols[i]]
        o = np.zeros(x_true.flatten().shape)
        rr = [e / map.f(o) for e in f_x_sols]
        f_X_sols.append(f_x_sols)
        RR.append(rr)


        err = [np.linalg.norm(correct_phase(xt.reshape(x_true.shape), x_true) - x_true) for xt in X_sols[i]]
        re = [e / np.linalg.norm(x_true) for e in err]
        # err = [np.linalg.norm(xt - x_true.flatten()) for xt in X_sols[i]]
        Err.append(err)
        RE.append(re)

    return f_X_sols, RR, Err, RE