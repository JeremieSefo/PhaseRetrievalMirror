import numpy as np
import odl
import scipy.misc
from Operators import soft_shrinkage
from TVRegularise import TVregularize
from scipy.signal import convolve2d
from numpy.fft import fftn, ifftn, fftshift, ifftshift



def phase_retrieval(L, kappa, xi, Algo, map, mask, n, Af, A, A_pinv, meas, maxiter, x, x_true_vect, IO_OO_HIO_beta, TvIter, TvAlpha, rho_Gau_Poi ):

    IO_beta = IO_OO_HIO_beta[0]
    OO_beta = IO_OO_HIO_beta[1]
    HIO_beta = IO_OO_HIO_beta[2]
    outsideMask = (1. + 0.j)*np.ones(mask.shape)
    outsideMask = outsideMask - mask
    space = odl.uniform_discr(min_pt=[0, 0], max_pt=mask.shape, shape=mask.shape, dtype='complex64')
    x0 = space.zero() 
    # add = space.element(y.real.reshape(mask.shape)) y.real
    # x0 = x0 + add
    Mask = space.element(mask)
    op = odl.FlatteningOperator(space)

    iterates = []
    iterates.append(x)
    
    beta = .7
    def P_S(x):
        return x * mask.reshape((n,))
    def P_M(x):
        X = Af(x).flatten() # (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
        X = (meas**(0.5)) * np.exp(1j* np.angle(X))
        return A_pinv(X) #(ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() # np.conjugate(A.T) @ (X)
    gamma_M = 1 #/(-beta)
    def R_M(x):
        return (1+gamma_M)*P_M(x) - gamma_M*x
    gamma_S = 1#/beta
    def R_S(x):
        return (1+gamma_S)*P_S(x) - gamma_S*x
    

    if Algo == 'real mirror' or Algo == 'complex mirror':
        if Algo == 'real mirror': a, b = 1., 0.
        if Algo == 'complex mirror': a, b = 0., 1.
        for k in range(maxiter):
            j = 0
            while  True:   
                L /= xi
                gamma = (1-kappa)/(L)#*mag 
                #print(gamma)
                x = iterates[k]
                grad_psi_x = np.asarray(map.grad_psi(x))
                grad_f_x = np.asarray(map.grad_f(x))
                z = grad_psi_x - gamma * grad_f_x

                x_temp  = a * map.grad_psi_star(z) + b * ( map.grad_psi_star(z.real) +  map.grad_psi_star(z.imag) * 1j ) # map.grad_psi_star(z.real) +  map.grad_psi_star(z.imag) * 1j #map.grad_psi_star(z)
                              
                '''
                print('iterate x: ', iterates[k], 'x_temp ', x_temp )
                print(f"A: {(map.breg_f(x_temp, iterates[k]))} B: { (L * map.breg_psi(x_temp, iterates[k]))}")
                print('while loop', j, "L", L, "gap", map.breg_f(x_temp, iterates[k]) - L * map.breg_psi(x_temp, iterates[k]))
                '''
                if (map.breg_f(x_temp, iterates[k])) >  (L  * map.breg_psi(x_temp, iterates[k])):
                    #print(f"A: {map.breg_f(x_temp, iterates[k])} B: { L * map.breg_psi(x_temp, iterates[k])}")
                    #print("hi")               
                    L *= xi
                    gamma = (1-kappa)/L 
                    x = iterates[k]
                    grad_psi_x = np.asarray(map.grad_psi(x))
                    grad_f_x = np.asarray(map.grad_f(x))
                    z = grad_psi_x - gamma * grad_f_x
                    x = a * map.grad_psi_star(z) + b * ( map.grad_psi_star(z.real) +  map.grad_psi_star(z.imag) * 1j )  # map.grad_psi_star(z)

                    ########## x = x * mask.reshape((n,))
                    ############ TV Support Regularization ############
                    # y = space.element(x.reshape(mask.shape))
                    # TVregularize(y, 0.15, Mask, x0, space, niter = 10, supportPrior = 'yes')
                    # x = op(x0) 

                    #x = x * op(mask)
                    #x = x * mask.reshape((n,))
                    #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
                    #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
                    
                    
                         


                    #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x 
                    '''
                    X = A @ np.conjugate(x)
                    X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #/(np.abs(X))) * X
                    x = np.conjugate(A.T) @ (X)
                    '''
                    iterates.append(x)
                    L *= (xi**j) # else L gets smaller than needed and the step size gets inappropiately large, 
                                 #sothat we get pulled away from the actual true minimum and the error jumps high again
                    break  
                #print("ho")
                j += 1 
                #if j % 10 == 0:
                    
            if k % 100 == 0:
            
               print('iterate', k+1)
    if Algo == 'Poisson-DRS': #In comparison to AAR, DRS allows a relaxation parameter on the magnitudes constraints, using proximal operators
                               #pseudoinverse of A discards any vector component which is orthogonal to the range of A, and by so doing annihilates the kernel of A.
        rho = rho_Gau_Poi[1] ############## rho = 0 gives AAR ##################
        for k in range(maxiter):
            
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #'''
            #A_pinv = np.linalg.pinv(A)
            ######################## Self-derived and adapted Poisson DRS, do not diverge as in the paper. Reconstruction gets however more blurred with larger rho and fails to converge locally #################
            #'''
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()
            #X = (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #A @ np.conjugate(x)
            Y = Af((x * mask.reshape((n,)))).flatten() # (fftn((x * mask.reshape((n,))).reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            RX_Y = 2 * Y - X
            Z = ( (-1) * (rho/(2*(rho + 2))) * np.linalg.norm(RX_Y) + (1/(2)) * ((((rho**2/((rho + 2)**2)) * np.linalg.norm(RX_Y)**2 + ((8)/(rho + 2)) * meas))**(0.5))) * ( np.exp(1j* np.angle(RX_Y)) )#amplitude-based Gaussian loss L
            X = (1/(rho + 1)) * (X) + ((rho - 1)/(rho + 1)) * Y + (1/(rho + 1)) * (Z)
            x = A_pinv(X) #(ifftn(X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten()
            #'''

            #X = (1/(rho + 1)) * (meas**(0.5)) * np.exp(1j* np.angle(X)) + (rho/(rho + 1)) * X           #/(np.abs(X))) * X
            #X = A @ ((A_pinv) @ X) #proximal point relative to the range of A. Seems to be without much effect

            ############## from the paper (not reproducible). iterates explode for rho larger  than zero ################
            '''
            X = (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #A @ np.conjugate(x)
            Y = (fftn((x * mask.reshape((n,))).reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #projection on the object constraints. But an element in the Fourier space
            R_X = (2*Y - X)

            #Z = (rho/(2*(rho + 2))) * R_X + (1/(2)) * ((((rho**2/((rho + 2)**2)) * np.linalg.norm(R_X)**2 + ((8)/(rho + 2)) * meas))**(0.5)) * np.exp(1j* np.angle(R_X)) #projection of R_X onto Fourier data constraints
            #X = 0.5 * X +   ((rho - 1)/(2*(rho + 1))) * R_X + (1/(rho + 1)) * Z
            
            #X = 0.5 * X +  ((rho**2 + 2*rho - 2)/(2*(rho + 1)*(rho + 2))) * (R_X) - (1/(2*(rho + 1))) * ((((rho**2/((rho + 2)**2)) * np.linalg.norm(R_X)**2 + ((8)/(rho + 2)) * meas))**(0.5)) * np.exp(1j* np.angle(R_X)) #explicit form
              
            X = 0.5 * X - (1/(rho + 2)) * R_X + (1/(2)) * ((((rho**2/((rho + 2)**2)) * np.linalg.norm(R_X)**2 + ((8)/(rho + 2)) * meas))**(0.5)) * np.exp(1j* np.angle(R_X)) #from paper
            x = (ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #(A_pinv) @ X #np.conjugate(A.T) @ (X)
            '''

            #'''
            ########## TV Regularizaton ####################
            y_real = space.element(x.real.reshape(mask.shape))
            y_imag = space.element(x.imag.reshape(mask.shape))
            xr = x0
            xi = x0
            TVregularize(y_real, TvAlpha, Mask, xr, space, niter = TvIter, supportPrior = 'no') 
            TVregularize(y_imag, TvAlpha, Mask, xi, space, niter = TvIter, supportPrior = 'no') 
            lincomb = odl.LinCombOperator(space, 1, 1.j)
            XX = odl.ProductSpace(space, space)
            xx = XX.element([xr, xi])
            x = space.element(x.reshape(mask.shape))
            lincomb(xx, out = x) 
            x = op(x)
            x = x.__array__()
            #'''

            #x = x * mask.reshape((n,))
            #x = .5 * x + .5 * R_S( R_M(x) )

            ########## TV Support Regularizaton ####################
            # y = space.element(x.reshape(mask.shape))
            # TVregularize(y, 0.15, Mask, x0, space, niter = 10, support = 'yes')
            # x = op(x0)

            #x = x * op(mask)
            #x = x * mask.reshape((n,))
            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
            

            #'''                                             #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)  
    if Algo == 'Gaussian-DRS': #rho = 0 boils down to AAR. Other values of rho converge slowly and ressemble error reduction, the more rho gets to 1
                               
                               #pseudoinverse of A discards any vector component which is orthogonal to the range of A, and by so doing annihilates the kernel of A. Though A here is unital
        (rho, alpha) = rho_Gau_Poi[0]
        for k in range(maxiter):
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #'''
            #A_pinv = np.linalg.pinv(A)
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            
            (Qx, Qy) = X.shape
            X = X.flatten()
            X_save = X
            #X = (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            Y = Af((x * mask.reshape((n,)))).flatten() #(fftn((x * mask.reshape((n,))).reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            RX_Y = 2 * Y - X
            Z = (1) * (1/(rho + 1)) * (meas**(0.5)) * np.exp(1j* np.angle(RX_Y)) + (rho/(rho + 1)) * RX_Y #amplitude-based Gaussian loss L. This is sign dependent
            X = (1/(rho + 1)) * (X) + ((rho - 1)/(rho + 1)) * Y + (1/(rho + 1)) * (Z)   #bad if prho >0        # (X) + 2 * alpha * (Z - Y) # performs better IEEE
            #X = .5 * X  + ((rho - 1)/2 * (rho + 1)) * Y + (1/(rho + 1)) * (Z)
            #X = alpha * X + (1 - alpha) * ((1/(rho + 1)) * (meas**(0.5)) * np.exp(1j* np.angle(X_save)) + (rho/(rho + 1)) * X_save) RAAR
            x = A_pinv(X) #Af.pinv(X) # #(ifftn(X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten()
            
            #x = (x * mask.reshape((n,)))
            '''
            X = (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #A @ np.conjugate(x)
            X = (1/(rho + 1)) * (meas**(0.5)) * np.exp(1j* np.angle(X)) + (rho/(rho + 1)) * X           #/(np.abs(X))) * X
            #X = A @ ((A_pinv) @ X) #proximal point relative to the range of A. Seems to be without much effect 
            x = (ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #(A_pinv) @ X # np.conjugate(A.T) @ (X)         
            '''
            '''
            ########## TV Regularizaton ####################
            y_real = space.element(x.real.reshape(mask.shape))
            y_imag = space.element(x.imag.reshape(mask.shape))
            xr = x0
            xi = x0
            TVregularize(y_real, TvAlpha, Mask, xr, space, niter = TvIter, supportPrior = 'no') 
            TVregularize(y_imag, TvAlpha, Mask, xi, space, niter = TvIter, supportPrior = 'no') 
            lincomb = odl.LinCombOperator(space, 1, 1.j)
            XX = odl.ProductSpace(space, space)
            xx = XX.element([xr, xi])
            x = space.element(x.reshape(mask.shape))
            lincomb(xx, out = x) 
            x = op(x)
            x = x.__array__()
            '''

            #x = 
            #x = .5 * x + .5 * R_S( R_M(x) )

            ########## TV Support Regularizaton ####################
            # y = space.element(x.reshape(mask.shape))
            # TVregularize(y, 0.15, Mask, x0, space, niter = 10, support = 'yes')
            # x = op(x0)

            #x = x * op(mask)
            #x = x * mask.reshape((n,))
            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
            

            #'''                                             #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)  
    if Algo == 'Peaceman-Rachford':

        for k in range(maxiter):
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #'''
            #X = A @ np.conjugate(x)
            #X = (meas**(0.5)) * np.exp(1j* np.angle(X))              

            #x = R_S( R_M(x.real) )  + R_S( R_M(x.imag) ) * 1j
            x = R_S( R_M(x) )
            #x = .5 * x + .5 * R_S( R_M(x) ) #AAR

            ########## TV Support Regularizaton ####################
            # y = space.element(x.reshape(mask.shape))
            # TVregularize(y, 0.15, Mask, x0, space, niter = 10, supportPrior = 'yes')
            # x = op(x0)

            #x = x * op(mask)
            #x = x * mask.reshape((n,))
            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
            

            #'''                                             #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'Averaged Alternating Reflections RSRM': 
                                                   # classical Douglas-Rachford algorithm #Problem with phase retrieval : Fourier Magnitude equality constraint defines a non convex set. 
                                                   # Propo 2.1 It would converge to a fix point if constraints sets were all convex closed.
        for k in range(maxiter):
           
           ######  x = .5 * x + .5 * R_S( R_M(x) ) ###### also works ########
            #'''
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten() # (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            P_M_X = (meas**(0.5)) * np.exp(1j* np.angle(X))  #Fourier amplitude fitting
            R_M_X = 2 * P_M_X - X     
            #R_rangeA_R_M_X = R_M_X #A is invertible with inverse A_pinv. So no need for A @ ((A_pinv) @ R_M_X) #proximal point relative to the range of A.    
            x_new = A_pinv(R_M_X)  # (ifftn(R_M_X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten() #back into object domain
            
            r_s_R_M_x = 2 * x_new * mask.reshape((n,)) - x_new # reflexion for support projection
            x = .5 * x + .5 *  r_s_R_M_x
            #'''
            #x = .5 * x + .5 * R_S( R_M(x) )  #x = .5 * x + .5 * R_M( R_S(x) ) # #AAR in object domain with support prior (not range(A) prior), performs worse.
                                            #One could maybe formulate an equivalent of range(A) prior in the object domain
           
            
            ########## TV Regularizaton #################### Please, don't try this at home
            '''
            y_real = space.element(x.real.reshape(mask.shape))
            y_imag = space.element(x.imag.reshape(mask.shape))
            xr = x0
            xi = x0
            TVregularize(y_real, TvAlpha, Mask, xr, space, niter = TvIter, supportPrior = 'no') 
            TVregularize(y_imag, TvAlpha, Mask, xi, space, niter = TvIter, supportPrior = 'no') 
            lincomb = odl.LinCombOperator(space, 1, 1.j)
            XX = odl.ProductSpace(space, space)
            xx = XX.element([xr, xi])
            x = space.element(x.reshape(mask.shape))
            lincomb(xx, out = x) 
            x = op(x)
            x = x.__array__()
            '''

            ########## TV Support Regularizaton ####################
            # y = space.element(x.reshape(mask.shape))
            # TVregularize(y, 0.15, Mask, x0, space, niter = 10, supportPrior = 'yes')
            # x = op(x0)

            #x = x * op(mask)
            #x = x * mask.reshape((n,))
            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
            

            #'''                                             #But for noiseless measurements, it does help getting the right scale

            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'Averaged Alternating Reflections RMRS': 
                                                   # classical Douglas-Rachford algorithm #Problem with phase retrieval : Fourier Magnitude equality constraint defines a non convex set. 
                                                   # Propo 2.1 It would converge to a fix point if constraints sets were all convex closed.
        for k in range(maxiter):
           
           ######  x = .5 * x + .5 * R_S( R_M(x) ) ###### also works ########
           
            '''X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten() # (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            P_M_X = (meas**(0.5)) * np.exp(1j* np.angle(X))  #Fourier amplitude fitting
            R_M_X = 2 * P_M_X - X     
            #R_rangeA_R_M_X = R_M_X #A is invertible with inverse A_pinv. So no need for A @ ((A_pinv) @ R_M_X) #proximal point relative to the range of A.    
            x_new = (ifftn(R_M_X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten() #back into object domain
            
            r_s_R_M_x = 2 * x_new * mask.reshape((n,)) - x_new # reflexion for support projection
            x = .5 * x + .5 *  r_s_R_M_x
            '''
            x = .5 * x + .5 * R_M( R_S(x) ) # #AAR in object domain with support prior (not range(A) prior), performs worse.
                                            #One could maybe formulate an equivalent of range(A) prior in the object domain
           
            
            ########## TV Regularizaton #################### Please, don't try this at home
            '''
            y_real = space.element(x.real.reshape(mask.shape))
            y_imag = space.element(x.imag.reshape(mask.shape))
            xr = x0
            xi = x0
            TVregularize(y_real, TvAlpha, Mask, xr, space, niter = TvIter, supportPrior = 'no') 
            TVregularize(y_imag, TvAlpha, Mask, xi, space, niter = TvIter, supportPrior = 'no') 
            lincomb = odl.LinCombOperator(space, 1, 1.j)
            XX = odl.ProductSpace(space, space)
            xx = XX.element([xr, xi])
            x = space.element(x.reshape(mask.shape))
            lincomb(xx, out = x) 
            x = op(x)
            x = x.__array__()
            '''

            ########## TV Support Regularizaton ####################
            # y = space.element(x.reshape(mask.shape))
            # TVregularize(y, 0.15, Mask, x0, space, niter = 10, supportPrior = 'yes')
            # x = op(x0)

            #x = x * op(mask)
            #x = x * mask.reshape((n,))
            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
            

            #'''                                             #But for noiseless measurements, it does help getting the right scale

            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'Hybrid Input-Output': ############### beta = 1 gives AAR #################

        for k in range(maxiter):
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #''' ########## An Attempt for HIO  in Fourier domain - How to include object support prior in Fourier space ########## Convolution theorem for IFT not well implemented?
            '''
            x_bef = iterates[-1]
            X_bef = A @ (x_bef)
            X = A @ (x)
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))  
                     
            X_rangeA = A @ ((A_pinv) @ X)  
            
         
            ########## TV Support Regularizaton ####################
            # y = space.element(x.reshape(mask.shape))
            # TVregularize(y, 0.15, Mask, x0, space, niter = 10)
            # x = op(x0)

            #x = x * op(mask)
            MASK = (A @ (mask.flatten())).reshape(mask.shape)
            conv_XrangeA_MASK = (convolve2d(X_rangeA.reshape(mask.shape), MASK, mode='same', boundary='wrap')).flatten()
            
            OUTSIDEMASK = (A @ (outsideMask.flatten())).reshape(mask.shape)
            conv_OUTSIDEMASK = (convolve2d((X_bef - HIO_beta * X_rangeA).reshape(mask.shape), OUTSIDEMASK, mode='same', boundary='wrap')).flatten()


            X = conv_XrangeA_MASK + conv_OUTSIDEMASK #HIO in Fourier domain
            x = np.conjugate(A.T) @ (X)

            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
            '''


            #''' ########## HIO in object domain - support prior ##########

            x_bef = iterates[-1]
            X = Af(x).flatten() #(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))               
            #X = A @ ((A_pinv) @ X)
            x = A_pinv(X) #(ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
         
            ########## TV Support Regularizaton ####################
            # y = space.element(x.reshape(mask.shape))
            # TVregularize(y, 0.15, Mask, x0, space, niter = 10)
            # x = op(x0)

            #x = x * op(mask)
            x = x * mask.reshape((n,)) + (x_bef - HIO_beta * x) * outsideMask.reshape((n,)) 
            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
            

                                                        #But for noiseless measurements, it does help getting the right scale
            #'''
            '''
            ########## TV Regularizaton ####################
            y_real = space.element(x.real.reshape(mask.shape))
            y_imag = space.element(x.imag.reshape(mask.shape))
            xr = x0
            xi = x0
            TVregularize(y_real, TvAlpha, Mask, xr, space, niter = TvIter, supportPrior = 'no') 
            TVregularize(y_imag, TvAlpha, Mask, xi, space, niter = TvIter, supportPrior = 'no') 
            lincomb = odl.LinCombOperator(space, 1, 1.j)
            XX = odl.ProductSpace(space, space)
            xx = XX.element([xr, xi])
            x = space.element(x.reshape(mask.shape))
            lincomb(xx, out = x) 
            x = op(x)
            x = x.__array__()
            '''


            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'Output-Output':

        for k in range(maxiter):
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #'''
            x_bef = iterates[-1]
            X = Af(x).flatten() #(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #A @ np.conjugate(x)
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #/(np.abs(X))) * X
            x = A_pinv(X) #(ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #np.conjugate(A.T) @ (X)
         
            ########## TV Support Regularizaton ####################
            # y = space.element(x.reshape(mask.shape))
            # TVregularize(y, 0.15, Mask, x0, space, niter = 10)
            # x = op(x0)

            #x = x * op(mask)
            x = x * mask.reshape((n,)) + (x - OO_beta * x) * outsideMask.reshape((n,))
            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
            

            #'''                                             #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'basic Input-Output':

        for k in range(maxiter):
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #'''
            x_bef = iterates[-1]
            X = Af(x).flatten() #(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #A @ np.conjugate(x)
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #/(np.abs(X))) * X
            x = A_pinv(X) #(ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #np.conjugate(A.T) @ (X)
         
            ########## TV Support Regularizaton ####################
            # y = space.element(x.reshape(mask.shape))
            # TVregularize(y, 0.15, Mask, x0, space, niter = 10)
            # x = op(x0)

            #x = x * op(mask)
            x = x_bef * mask.reshape((n,)) + (x_bef - IO_beta * x) * outsideMask.reshape((n,))
            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
            

            #'''                                             #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'Error Reduction : support prior':

        for k in range(maxiter):
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #'''
            X = Af(x).flatten() #(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() # A @(x)

            X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #attribute measured Fourier Intensities
            #X = (fftn((ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')), s = mask.shape, norm = 'ortho')) 
                                                                           #A @ ((A_pinv) @ X) #proximal point relative to the range of A. Seems to be without much effect 
            
            x = A_pinv(X).flatten() #(ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #np.conjugate(A.T) @ (X)

            ########## TV Support Regularizaton ####################
            # y = space.element(x.reshape(mask.shape))
            # TVregularize(y, 0.15, Mask, x0, space, niter = 10, supportPrior = 'yes')
            # x = op(x0)

            #x = x * op(mask)
            x = x * mask.reshape((n,))
            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
            

            #'''                                             #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'Error Reduction + TV-support':

        for k in range(maxiter):
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #'''
            X = Af(x).flatten() # (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #A @ np.conjugate(x)
            #x = np.conjugate(A.T) @ (X)
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #/(np.abs(X))) * X
            #z = space.zero()
            y = A_pinv(X) # (ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #np.conjugate(A.T) @ (X)
            #z = space.element(z)
            #t = space.zero()
            z = 2 * y  - x # (2 * z - x)
            #print(type(t))
            ########## TV Support Regularizaton ####################
            t = space.element(z.reshape(mask.shape)) #y = space.element(x.real.reshape(mask.shape)) + space.element(x.imag.reshape(mask.shape)) *1j
        
            xr = x0.real
            xi = x0.real
            TVregularize(t, TvAlpha, Mask, x0, space, niter = TvIter, supportPrior = 'no') 
            #TVregularize(y.imag, 0.15, Mask, xi, space, niter = TvIter) 
            x = x + op(x0) - y
            x = x.__array__()

            #x = x * op(mask)
            ############################################################################################################################x = x * mask.reshape((n,))
            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
            

            #'''                                             #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k) 
    if Algo == 'Gerchberg-Saxton':

        for k in range(maxiter):
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #'''
            X = Af(x).flatten() # (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #A @ np.conjugate(x)
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #/(np.abs(X))) * X
            x = A_pinv(X) # (ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #np.conjugate(A.T) @ (X)
            x = np.abs(x_true_vect) * np.exp(1j* np.angle(x))

            ########## TV Support Regularizaton ####################
            # y = space.element(x.reshape(mask.shape))
            # TVregularize(y, 0.15, Mask, x0, space, niter = 10, supportPrior = 'yes')
            # x = op(x0)

            #x = x * op(mask)
            ################# x = x * mask.reshape((n,))
            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
            

            #'''                                             #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                print('iteration k', k) 
    if Algo == 'Gerchberg-Saxton + TV-support':

        for k in range(maxiter):
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #'''
            X = Af(x).flatten() #A @ np.conjugate(x)
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #/(np.abs(X))) * X
            x = A_pinv(X) #np.conjugate(A.T) @ (X)
            x = np.abs(x_true_vect) * np.exp(1j* np.angle(x))

            ########## TV Support Regularizaton ####################
            y = space.element(x.reshape(mask.shape)) #y = space.element(x.real.reshape(mask.shape)) + space.element(x.imag.reshape(mask.shape)) *1j
            xr = x0.real
            xi = x0.imag
            TVregularize(y, TvAlpha, Mask, x0, space, niter = TvIter, supportPrior = 'yes') 
            #TVregularize(space.element(y.imag), 0.15, Mask, xi, space, niter = TvIter) 
            x = op(x0)

            #x = x * op(mask)
            ################# x = x * mask.reshape((n,))
            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
            

            #'''                                             #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                print('iteration k', k)            
    if Algo == 'No Support prior - fast':

        for k in range(maxiter):
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #'''
            X = Af(x).flatten() #(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #/(np.abs(X))) * X
            #X = A @ ((A_pinv) @ X) #proximal point relative to the range of A. It finds the true support better than a direct support estimation prior appplication. Why?
            x = A_pinv(X) #(ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            '''
            ########## TV Regularizaton #################### Don't TV regularise. Bad idea here
            y_real = space.element(x.real.reshape(mask.shape))
            y_imag = space.element(x.imag.reshape(mask.shape))
            xr = x0
            xi = x0
            TVregularize(y_real, TvAlpha, Mask, xr, space, niter = TvIter, supportPrior = 'no') 
            TVregularize(y_imag, TvAlpha, Mask, xi, space, niter = TvIter, supportPrior = 'no') 
            lincomb = odl.LinCombOperator(space, 1, 1.j)
            XX = odl.ProductSpace(space, space)
            xx = XX.element([xr, xi])
            x = space.element(x.reshape(mask.shape))
            lincomb(xx, out = x) 
            x = op(x)
            x = x.__array__()
            '''
           
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'No Support prior - slow':

        for k in range(maxiter):
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #'''
            X = A @ np.conjugate(x)
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #/(np.abs(X))) * X
            x = np.conjugate(A.T) @ (X)

            #x = space.element(x)

            ########## TV Support Regularizaton ####################
            # y = space.element(x.reshape(mask.shape))
            # TVregularize(y, 0.15, Mask, x0, space, niter = 10, supportPrior = 'yes')
            # x = op(x0)

            #x = x * op(mask)
            ################# x = x * mask.reshape((n,))
            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
            

            #'''                                             #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                print('iteration k', k)
    if Algo == 'No Support prior + TV-support real_imag_separated':

        for k in range(maxiter):
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #'''
            X = Af(x).flatten() #A @ np.conjugate(x)
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #/(np.abs(X))) * X
            x = A_pinv(X) #np.conjugate(A.T) @ (X)

            ########## TV Support Regularizaton ####################
            #y = space.element(x.real.reshape(mask.shape))  + space.element(x.imag.reshape(mask.shape)) *1j
            y_real = space.element(x.real.reshape(mask.shape))
            y_imag = space.element(x.imag.reshape(mask.shape))
            xr = x0
            xi = x0
            TVregularize(y_real, TvAlpha, Mask, xr, space, niter = TvIter, supportPrior = 'yes') 
            TVregularize(y_imag, TvAlpha, Mask, xi, space, niter = TvIter, supportPrior = 'yes') 
            lincomb = odl.LinCombOperator(space, 1, 1.j)
            XX = odl.ProductSpace(space, space)
            xx = XX.element([xr, xi])
            x = space.element(x.reshape(mask.shape))
            lincomb(xx, out = x) 
            x = op(x)
            #x = op(xr).__array__() + op(xi).__array__() * 1j

            #x = x * op(mask)
            ################# x = x * mask.reshape((n,))
            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
            

            #'''                                             #But for noiseless measurements, it does help getting the right scale
            x = x.__array__()
            iterates.append(x)
            #x = x.flatten()
            if k % 100 == 0: 
                print('iteration k', k)
    if Algo == 'No Support prior + TV-support':

        for k in range(maxiter):
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #'''
            X = Af(x).flatten() # A @ np.conjugate(x)
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #/(np.abs(X))) * X
            x = A_pinv(X) # np.conjugate(A.T) @ (X)

            ########## TV Support Regularizaton ####################
            #y = space.element(x.real.reshape(mask.shape))  + space.element(x.imag.reshape(mask.shape)) *1j
            y = space.element(x.reshape(mask.shape))
            #y_imag = space.element(x.imag.reshape(mask.shape))
            #xr = x0
            #xi = x0
            TVregularize(y, TvAlpha, Mask, x0, space, niter = TvIter, supportPrior = 'yes') 
            #TVregularize(y_imag, 1., Mask, xi, space, niter = TvIter) 
            x = op(x0).__array__() #+ op(xi).__array__() * 1j

            #x = x * op(mask)
            ################# x = x * mask.reshape((n,))
            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
            

            #'''                                             #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                print('iteration k', k)       
    if Algo == 'stable':
        for k in range(maxiter):
            j = 0

            x_temp = x
            while  True:   
                
                gamma = (1-kappa)/L 

                z = map.grad_psi(x) - gamma * map.grad_f(x)
                x_prev = x_temp
                
                x_temp  = map.grad_psi_star(z)


                print(np.linalg.norm(x_prev-x_temp))
                L /= xi

                print(f"A: {map.breg_f(x_temp, iterates[k])} B: {xi * L * map.breg_psi(x_temp, iterates[k])} C: {map.breg_psi(x_temp, iterates[k])}")
                    
                if map.breg_f(x_temp, iterates[k]) <= xi * L  * map.breg_psi(x_temp, iterates[k]):
                    print(f"A: {map.breg_f(x_temp, iterates[k])} B: {xi * L * map.breg_psi(x_temp, iterates[k])}")
                    #print("hi")
                    x = x_temp
                    L *= xi
                    break
                
                #print("ho")
                j += 1 
                #if j % 10 == 0:
                print('while loop', j, "L", L, "gap", map.breg_f(x_temp, iterates[k]) - xi * L * map.breg_psi(x_temp, iterates[k]))
        
           
            
            iterates.append(x)
            
            if k % 100 == 0:
            
               print('iterate', k+1)
    if Algo == 'none':
        for k in range(maxiter):
            j = 0

            while  True:   
                L /= xi
                gamma = (1-kappa)/L 

                z = map.grad_psi(x) - gamma * map.grad_f(x)
                x_temp  = map.grad_psi_star(z)

                
                if map.breg_f(x_temp, iterates[k]) <= xi * L  * map.breg_psi(x_temp, iterates[k]):
                    print(f"A: {map.breg_f(x_temp, iterates[k])} B: {xi * L * map.breg_psi(x_temp, iterates[k])}")
                    #print("hi")
                    x = x_temp
                    break
                
                #print("ho")
                j += 1 
                #if j % 10 == 0:
                print('while loop', j, "L", L, "gap", map.breg_f(x_temp, iterates[k]) - xi * L * map.breg_psi(x_temp, iterates[k]))
        
            L *= xi
            
            iterates.append(x)
            
            if k % 100 == 0:
            
               print('iterate', k+1)        

    return iterates, space