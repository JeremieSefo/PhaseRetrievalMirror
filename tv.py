# matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import odl
import scipy.misc
import primal_dual_hybrid_gradient_support as PDHGS

class support(odl.solvers.Functional):

    def __init__(self, mask, X):
        self.mask = mask
        self.X = X
        super(support, self).__init__(self.X)

    def __call__(self, x):
        M = np.ones(self.mask.shape)
        M -= self.mask
        func = odl.solvers.IndicatorZero(self.X)
        return func(x * M)

    def proximal(self):
        """Return the proximal factory of the functional.

        This is the zero operator.
        """
        def support_proximal(space, mask):

            class ProximalSupport(odl.Operator):

                def __init__(self, sigma):
                    super(ProximalSupport, self).__init__(
                        domain=space, range=space, linear=True)
                    if np.isscalar(sigma):
                        self.sigma = float(sigma)
                    else:
                        self.sigma = space.element(sigma)

                def __call__(self, primal_tmp, out):
                    out.assign(primal_tmp * mask)

            return ProximalSupport

        return support_proximal(self.X, self.mask)

# Usage
import SensingMatrix as sm

def TVregularize(y, alpha, mask, x, X):
    grad = odl.Gradient(X)
    I = odl.IdentityOperator(X)
    L = odl.BroadcastOperator(I, grad)

    f_1 = odl.solvers.L2NormSquared(X).translated(y)
    f_2 = alpha * odl.solvers.L1Norm(grad.range)
    f = odl.solvers.SeparableSum(f_1, f_2)

    g = support(mask, X)
    L_norm = 1.1 * odl.power_method_opnorm(L, xstart=y, maxiter=20)

    tau = 1.0 / L_norm
    sigma = 1.0 / L_norm

    PDHGS.pdhgs(x, g, f, L, tau=tau, sigma=sigma, niter=20)

image = scipy.datasets.ascent().astype('complex').reshape((512, 512))
image /= image.max()
image = np.rot90(image, -1)

Nx, Ny = image.shape
X = odl.uniform_discr(min_pt=[0, 0], max_pt=image.shape, shape=image.shape)

x_true = X.element(image)
y = x_true + 0.1 * odl.phantom.white_noise(X, seed=42)
x_true.show('true image')
y.show('noisy image')

x = X.zero()

from SetUpImage import setUpImage
s = setUpImage(Nx, Ny)
true_images, mask = s()
mask = X.element(mask)
TVregularize(y, 0.15, mask, x, X)
x.show('Denoised image');
(x_true - x).show('Difference true - denoised');
#x.show('Denoised image');
from odl.contrib import fom
print('Noisy')
print('-----')
print('Mean squared error:', fom.mean_squared_error(y, x_true))
print('PSNR:', fom.psnr(y, x_true))
print('SSIM:', fom.ssim(y, x_true))
print('')

print('Denoised')
print('--------')
print('Mean squared error:', fom.mean_squared_error(x, x_true))
print('PSNR:', fom.psnr(x, x_true))
print('SSIM:', fom.ssim(x, x_true))