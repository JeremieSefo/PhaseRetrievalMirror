import numpy as np
import matplotlib.pyplot as plt
import odl

def phase_retrie_plots_alternate(idx, grd_truths, X_sols, map, A, meas, space, Algos):
    maxiter = len(X_sols)
    op = odl.FlatteningOperator(space)
    x_tru = grd_truths[idx]
    K = np.arange(1, maxiter + 1)
    f_x_sols = [map.f(x) for x in X_sols]
    m = len(A(x_tru.flatten()).flatten())

    fig, axs = plt.subplots(5, 4, figsize=(32, 32))
    axs = axs.flatten()
    axs[8].axis('off')
    axs[16].axis('off')
    # Function to plot images
    def plot_image(ax, image, title, vmin, vmax):
        im = ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax, origin='lower')
        ax.axis('off')
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    # Plot ground truth
    plot_image(axs[0], x_tru.real, 'Real Part-Ground Truth', np.min(x_tru.real), np.max(x_tru.real))
    plot_image(axs[4], x_tru.imag, 'Imaginary Part-Ground Truth', np.min(x_tru.imag), np.max(x_tru.imag))

    # Plot solution
    x = op.inverse(X_sols[-1])
    plot_image(axs[1], x.real, 'Real Part - '+Algos[0]+' alternate with '+Algos[1], np.min(x.real), np.max(x.real))
    plot_image(axs[5], x.imag, 'Imaginary Part - '+Algos[0]+' alternate with '+Algos[1], np.min(x.imag), np.max(x.imag))

    # Plot errors
    axs[13].plot(K, [(meas - np.abs(A(x).flatten())**2) / (2 * np.sqrt(m)) for x in X_sols])
    axs[13].set_title('Fourier Magnitude Pixels Error')
    axs[9].loglog(K, f_x_sols)
    axs[9].set_title('Fourier error')
    axs[17].loglog(K, [np.linalg.norm(x-x_tru.flatten()) for x in X_sols])
    axs[17].set_title('Object domain error')

    plt.show()

# # Example usage with dummy data
# idx = 0
# grd_truths = [np.random.rand(64, 64) + 1j * np.random.rand(64, 64)]
# X_sols = [np.random.rand(64, 64) + 1j * np.random.rand(64, 64) for _ in range(10)]
# map = odl.OperatorNorm(odl.IdentityOperator(odl.uniform_discr([0, 0], [64, 64], [64, 64], dtype=np.complex64)))
# A = odl.IdentityOperator(odl.uniform_discr([0, 0], [64, 64], [64, 64], dtype=np.complex64))
# meas = np.random.rand(64*64)
# space = odl.uniform_discr([0, 0], [64, 64], [64, 64], dtype=np.complex64)
# Algos = ['Algorithm1', 'Algorithm2']

# phase_retrie_plots_alternate(idx, grd_truths, X_sols, map, A, meas, space, Algos)
