#matplotlib inline
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
        M = (1. + 0.j) * np.ones(self.mask.shape)
        M -= self.mask
        func = odl.solvers.IndicatorZero(self.X)
        return func(x*M)
    '''
    def proximal(self, x):
        """Return the proximal factory of the functional.

        This is the zero operator.
        """
        def support_proximal(space, out):
         
            class support_operator(odl.Operator):

                def __init__(self,):
                    
                    super(support_operator, self).__init__(
                        domain=space, range=space, linear=True)
                def __call__(self, x, out):
                    out.assign( x*self.mask)
                    
        #     """Proximal factory for zero operator.

        #     Parameters
        #     ----------
        #     sigma : positive float, optional
        #         Step size parameter.
        #     """
        #     return support_operator(self.X, self.mask)

        return support_proximal
        '''
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
                    out.assign( primal_tmp*mask)
            return ProximalSupport       
       
        return support_proximal(self.X, self.mask)    

import SensingMatrix as sm


def TVregularize(y, alpha,  mask, x, X, niter, supportPrior):
    # `min_pt` corresponds to `a`, `max_pt` to `b`
    #X = odl.uniform_discr(min_pt=[0, 0], max_pt=y.shape, shape=y.shape)
    #print('Pixel size:', X.cell_sides)
    grad = odl.Gradient(X)
    #print('Gradient domain X:', grad.domain)
    #print('Gradient range X^d:', grad.range)
    #A = sm.FourierMatrix(mask.shape[0], mask.shape[1])
    #B = X.element(A.Matrix)
    #op = odl.MultiplyOperator(B)
    I = odl.IdentityOperator(X)
    L = odl.BroadcastOperator(I, grad)
    #print('L domain X:', L.domain)
    #print('L range X x X^d:', L.range)
    # `.translated(y)` takes care of the `. - y` part in the function
    f_1 = odl.solvers.L2NormSquared(X).translated(y)
    # The regularization parameter `alpha` is multiplied with the L1 norm.
    # The L1 norm must be defined on X^d, the range of the gradient.
    #alpha = 0.15
    f_2 = alpha * odl.solvers.L1Norm(grad.range)
    f = odl.solvers.SeparableSum(f_1, f_2)

    # We can test whether everything makes sense by evaluating `f(L(x))`
    # at some arbitrary `x` in `X`. It should produce a scalar.
    #print(f(L(X.zero())))
    #print(g(np.array([2, 0, 0, 0]).reshape(y.shape)))
    L_norm = 1.1 * odl.power_method_opnorm(L, xstart=y, maxiter=20)

    tau = 1.0 / L_norm
    sigma = 1.0 / L_norm
    
    #print('||L|| =', L_norm)
    if supportPrior == 'yes':
        g = support(mask, X)
        PDHGS.pdhgs(x, g, f, L, tau=tau, sigma=sigma, niter=niter)
    if supportPrior == 'no':
        g = odl.solvers.ZeroFunctional(X)
        odl.solvers.pdhg(x, g, f, L, tau=tau, sigma=sigma, niter=niter)
        # for _ in range(niter): 
        #     odl.solvers.accelerated_proximal_gradient( L, [f_1, f_2], x, niter=1, gamma=sigma )
    #g = odl.solvers.IndicatorNonnegativity(X)

    #PDHGS.pdhgs(x, g, f, L, tau=tau, sigma=sigma, niter=niter)
    
