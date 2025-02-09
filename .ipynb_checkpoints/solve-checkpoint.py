import numpy as np
import odl
import scipy.misc
import Operators as op
from Operators import soft_shrinkage
from TVRegularise import TVregularize
from scipy.signal import convolve2d
from numpy.fft import fftn, ifftn, fftshift, ifftshift



def phase_retrieval(noise, kappa, xi, Algo, map, mask, n, Af, A, A_pinv, meas, maxiter, x, x_true_vect, IO_OO_HIO_newHIO_beta, RAAR_AAR_beta, TvIter, TvAlpha, rho_Gau_Poi ):
    
    AAR_lambda = RAAR_AAR_beta[1]
    RAAR_beta = RAAR_AAR_beta[0]
    IO_beta = IO_OO_HIO_newHIO_beta[0]
    OO_beta = IO_OO_HIO_newHIO_beta[1]
    HIO_beta = IO_OO_HIO_newHIO_beta[2]
    [alph, bet, gamm] = IO_OO_HIO_newHIO_beta[3]
    outsideMask = (1. + 0.j)*np.ones(mask.shape)
    outsideMask = outsideMask - mask
    space = odl.uniform_discr(min_pt=[0, 0], max_pt=mask.shape, shape=mask.shape, dtype='complex64')
    x0 = space.zero() 
    # add = space.element(y.real.reshape(mask.shape)) y.real
    # x0 = x0 + add
    Mask = space.element(mask)
    op = odl.FlatteningOperator(space)

    iterates = []
    iterates.append(x) #np.real(x)
    
    
    beta = .7
    def P_S(x):
        x = x.copy()
        indices = np.logical_not(mask.reshape((n,)))
        x[indices] = 0
        return x #* mask.reshape((n,))
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
        x = P_S(x)
        L = 1 * op.smoothnessPara_L(Af.Matrix, noise)
        if Algo == 'real mirror': a, b = 1., 0.
        if Algo == 'complex mirror': a, b = 0., 1.
        for k in range(maxiter):
            j = 0
            l = k
            if l == k : #while   True:   
                L /= xi
                gamma = (1+kappa)#/(L)#*mag 
                #print(gamma)
                x = iterates[k]
                grad_psi_x = np.asarray(map.grad_psi(x))
                grad_f_x = np.asarray(map.grad_f(x))
                #x_temp = iterates[k] - gamma * grad_f_x # gradient descent step
                z = grad_psi_x - gamma * grad_f_x #mirror gradient descent step
                x_temp  = a * map.grad_psi_star(z) + b * ( map.grad_psi_star(z.real) +  map.grad_psi_star(z.imag) * 1j ) #mirror step
                                                                                     # map.grad_psi_star(z.real) +  map.grad_psi_star(z.imag) * 1j #map.grad_psi_star(z)
                              
                '''
                print('iterate x: ', iterates[k], 'x_temp ', x_temp )
                print(f"A: {(map.breg_f(x_temp, iterates[k]))} B: { (L * map.breg_psi(x_temp, iterates[k]))}")
                print('while loop', j, "L", L, "gap", map.breg_f(x_temp, iterates[k]) - L * map.breg_psi(x_temp, iterates[k]))
                '''
                
                if 1 < 0: #(map.breg_f(x_temp, iterates[k])) >  (L  * map.breg_psi(x_temp, iterates[k])):
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
                    #X = A @ np.conjugate(x)
                    #X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #/(np.abs(X))) * X
                    #x = np.conjugate(A.T) @ (X)
                    '''
                    iterates.append(x)
                    L *= (xi**(int(j/maxiter)) + 1) # int(j/2) else L gets smaller than needed and the step size gets inappropiately large, 
                                 #sothat we get pulled away from the actual true minimum and the error jumps high again
                    break  
                x = x_temp
                iterates.append(x)

                #print("ho")
                j += 1 
                #if j % 10 == 0:
                    
            if k % 1 == 0:
            
               print('iterate', k+1)

    if Algo == 'Poisson-DRS': #In comparison to AAR, DRS allows a relaxation parameter on the magnitudes constraints, using proximal operators
                               #pseudoinverse of A discards any vector component which is orthogonal to the range of A, and by so doing annihilates the kernel of A.
        (rho, alpha) = rho_Gau_Poi[1] ############## rho = 0 gives AAR ##################
        for k in range(maxiter):
        
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()
            Y = Af(P_S(x)).flatten() # (fftn((x * mask.reshape((n,))).reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            RX_Y = 2 * Y - X
            Z = ( (-1) * (rho/(2*(rho + 2))) * np.abs(RX_Y) + (1/(2)) * ( ( ((rho**2/((rho + 2)**2)) * np.abs(RX_Y)**2 
                                                                    + ((8)/(rho + 2)) * meas) )**(0.5) ) ) * ( np.exp(1j* np.angle(RX_Y)) )#amplitude-based Gaussian loss L
            
            #X = (1/(rho + 1)) * (X) + ((rho - 1)/(rho + 1)) * Y + (1/(rho + 1)) * (Z) 
            X = (X) + 2 * alpha * (Z - Y) #alternative

            x = A_pinv(X) #(ifftn(X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten()
                                                       #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)  
    if Algo == 'Poisson-DRS in OS': # object space 
                               #In comparison to AAR, DRS allows a relaxation parameter on the magnitudes constraints, using proximal operators
                               #pseudoinverse of A discards any vector component which is orthogonal to the range of A, and by so doing annihilates the kernel of A.
        (rho, alpha) = rho_Gau_Poi[1] ############## rho = 0 gives AAR ##################
        for k in range(maxiter):
        
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()
            y = (P_S(x))#.flatten() # (fftn((x * mask.reshape((n,))).reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            r_S_x = 2 * y - x 
            RX_Y = Af(r_S_x).flatten()
            Z = ( (-1) * (rho/(2*(rho + 2))) * np.abs(RX_Y) + (1/(2)) * ( ( ((rho**2/((rho + 2)**2)) * np.abs(RX_Y)**2 
                                                                    + ((8)/(rho + 2)) * meas) )**(0.5) ) ) * ( np.exp(1j* np.angle(RX_Y)) )#amplitude-based Gaussian loss L
            z = A_pinv(Z) #crucial difference
            #X = (1/(rho + 1)) * (X) + ((rho - 1)/(rho + 1)) * Y + (1/(rho + 1)) * (Z) 
            x = (x) + 2 * alpha * (z - y) #alternative

            #x = A_pinv(X) #(ifftn(X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten()
                                                       #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)  
    if Algo == 'Poisson-DRS + TVS_M': #In comparison to AAR, DRS allows a relaxation parameter on the magnitudes constraints, using proximal operators
                               #pseudoinverse of A discards any vector component which is orthogonal to the range of A, and by so doing annihilates the kernel of A.
        (rho, alpha) = rho_Gau_Poi[1] ############## rho = 0 gives AAR ##################
        for k in range(maxiter):
      
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()

            Y = ( (-1) * (rho/(2*(rho + 2))) * np.abs(X) + (1/(2)) * ((((rho**2/((rho + 2)**2)) * np.abs(X)**2 
                                                                        + ((8)/(rho + 2)) * meas))**(0.5))) * ( np.exp(1j* np.angle(X)) )#amplitude-based Gaussian loss L
            R_M_X = 2 * Y - X
            x = A_pinv(R_M_X)
            #'''
            ########## TV Regularizaton ####################
            y_real = space.element(x.real.reshape(mask.shape))
            y_imag = space.element(x.imag.reshape(mask.shape))
            xr = x0
            xi = x0
            TVregularize(y_real, TvAlpha, Mask, xr, space, niter = TvIter, supportPrior = 'yes') 
            if np.linalg.norm(y_imag) > 1e-3:
                TVregularize(y_imag, TvAlpha, Mask, xi, space, niter = TvIter, supportPrior = 'yes') 
            
            lincomb = odl.LinCombOperator(space, 1, 1.j)
            XX = odl.ProductSpace(space, space)
            xx = XX.element([xr, xi])
            x = space.element(x.reshape(mask.shape))
            lincomb(xx, out = x) 
            x = op(x)
            x = x.__array__()

            Z = (Af(x.flatten())).flatten()
            X = (X) + 2 * alpha * (Z - Y) #alternative
            x = A_pinv(X) #(ifftn(X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten()
                                                 #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k) 
    if Algo == 'Poisson-DRS in OS + TVS_M': #In comparison to AAR, DRS allows a relaxation parameter on the magnitudes constraints, using proximal operators
                               #pseudoinverse of A discards any vector component which is orthogonal to the range of A, and by so doing annihilates the kernel of A.
        (rho, alpha) = rho_Gau_Poi[1] ############## rho = 0 gives AAR ##################
        for k in range(maxiter):
      
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()

            Y = ( (-1) * (rho/(2*(rho + 2))) * np.abs(X) + (1/(2)) * ((((rho**2/((rho + 2)**2)) * np.abs(X)**2 
                                                                        + ((8)/(rho + 2)) * meas))**(0.5))) * ( np.exp(1j* np.angle(X)) )#amplitude-based Gaussian loss L
            #R_M_X = 2 * Y - X
            y = A_pinv(Y)
            r_M_x = 2 * y - x
            x_bef = x
            x = r_M_x
            #x = A_pinv(R_M_X)
            #'''
            ########## TV Regularizaton ####################
            y_real = space.element(x.real.reshape(mask.shape))
            y_imag = space.element(x.imag.reshape(mask.shape))
            xr = x0
            xi = x0
            TVregularize(y_real, TvAlpha, Mask, xr, space, niter = TvIter, supportPrior = 'yes') 
            if np.linalg.norm(y_imag) > 1e-3:
                TVregularize(y_imag, TvAlpha, Mask, xi, space, niter = TvIter, supportPrior = 'yes') 
            
            lincomb = odl.LinCombOperator(space, 1, 1.j)
            XX = odl.ProductSpace(space, space)
            xx = XX.element([xr, xi])
            x = space.element(x.reshape(mask.shape))
            lincomb(xx, out = x) 
            x = op(x)
            x = x.__array__()

            #Z = (Af(x.flatten())).flatten()
            z = x
            x = x_bef + 2 * alpha * (z - y)
            #X = (X) + 2 * alpha * (Z - Y) #alternative
            #x = A_pinv(X) #(ifftn(X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten()
                                                 #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k) 
    if Algo == 'Poisson-DRS + M_TVS': #In comparison to AAR, DRS allows a relaxation parameter on the magnitudes constraints, using proximal operators
                               #pseudoinverse of A discards any vector component which is orthogonal to the range of A, and by so doing annihilates the kernel of A.
        (rho, alpha) = rho_Gau_Poi[1] ############## rho = 0 gives AAR ##################
        for k in range(maxiter):
      
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()
            #'''
            ########## TV Regularizaton ####################
            y_real = space.element(x.real.reshape(mask.shape))
            y_imag = space.element(x.imag.reshape(mask.shape))
            xr = x0
            xi = x0
            TVregularize(y_real, TvAlpha, Mask, xr, space, niter = TvIter, supportPrior = 'yes') 
            if np.linalg.norm(y_imag) > 1e-3:
                TVregularize(y_imag, TvAlpha, Mask, xi, space, niter = TvIter, supportPrior = 'yes') 
            
            lincomb = odl.LinCombOperator(space, 1, 1.j)
            XX = odl.ProductSpace(space, space)
            xx = XX.element([xr, xi])
            x = space.element(x.reshape(mask.shape))
            lincomb(xx, out = x) 
            x = op(x)
            x = x.__array__()

            Y = (Af(x.flatten())).flatten() # * mask.reshape((n,)) (fftn((x * mask.reshape((n,))).reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()            
            RX_Y = 2 * Y - X
            Z = ( (-1) * (rho/(2*(rho + 2))) * np.abs(RX_Y) + (1/(2)) * ((((rho**2/((rho + 2)**2)) * np.abs(RX_Y)**2 
                                                                           + ((8)/(rho + 2)) * meas))**(0.5))) * ( np.exp(1j* np.angle(RX_Y)) )#amplitude-based Gaussian loss L
            
            #X = (1/(rho + 1)) * (X) + ((rho - 1)/(rho + 1)) * Y + (1/(rho + 1)) * (Z)
            X = (X) + 2 * alpha * (Z - Y) #alternative
            x = A_pinv(X) #(ifftn(X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten()
                                                 #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)  
    if Algo == 'Poisson-DRS in OS + M_TVS': #In comparison to AAR, DRS allows a relaxation parameter on the magnitudes constraints, using proximal operators
                               #pseudoinverse of A discards any vector component which is orthogonal to the range of A, and by so doing annihilates the kernel of A.
        (rho, alpha) = rho_Gau_Poi[1] ############## rho = 0 gives AAR ##################
        for k in range(maxiter):
            x_bef = x
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()
            #'''
            ########## TV Regularizaton ####################
            y_real = space.element(x.real.reshape(mask.shape))
            y_imag = space.element(x.imag.reshape(mask.shape))
            xr = x0
            xi = x0
            TVregularize(y_real, TvAlpha, Mask, xr, space, niter = TvIter, supportPrior = 'yes') 
            if np.linalg.norm(y_imag) > 1e-3:
                TVregularize(y_imag, TvAlpha, Mask, xi, space, niter = TvIter, supportPrior = 'yes') 
            
            lincomb = odl.LinCombOperator(space, 1, 1.j)
            XX = odl.ProductSpace(space, space)
            xx = XX.element([xr, xi])
            x = space.element(x.reshape(mask.shape))
            lincomb(xx, out = x) 
            x = op(x)
            x = x.__array__()
            
            y = x
            r_s_x = 2 *  y - x_bef
            #Y = (Af(x.flatten())).flatten() # * mask.reshape((n,)) (fftn((x * mask.reshape((n,))).reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()            
            RX_Y = (Af(r_s_x.flatten())).flatten()
            Z = ( (-1) * (rho/(2*(rho + 2))) * np.abs(RX_Y) + (1/(2)) * ((((rho**2/((rho + 2)**2)) * np.abs(RX_Y)**2 
                                                                           + ((8)/(rho + 2)) * meas))**(0.5))) * ( np.exp(1j* np.angle(RX_Y)) )#amplitude-based Gaussian loss L
            z = A_pinv(Z)
            #X = (1/(rho + 1)) * (X) + ((rho - 1)/(rho + 1)) * Y + (1/(rho + 1)) * (Z)
            #X = (X) + 2 * alpha * (Z - Y) #alternative
            x = x_bef + 2 * alpha * (z - y)
            #x = A_pinv(X) #(ifftn(X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten()
                                                 #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)  
    if Algo == 'Poisson-DRS + TV1': #In comparison to AAR, DRS allows a relaxation parameter on the magnitudes constraints, using proximal operators
                               #pseudoinverse of A discards any vector component which is orthogonal to the range of A, and by so doing annihilates the kernel of A.
        (rho, alpha) = rho_Gau_Poi[1] ############## rho = 0 gives AAR ##################
        for k in range(maxiter):
      
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()

            Y = (Af(P_S(x))).flatten() # * mask.reshape((n,)) (fftn((x * mask.reshape((n,))).reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            
            RX_Y = 2 * Y - X


            Z = ( (-1) * (rho/(2*(rho + 2))) * np.abs(RX_Y) + (1/(2)) * ((((rho**2/((rho + 2)**2)) * np.abs(RX_Y)**2 + ((8)/(rho + 2)) * meas))**(0.5))) * ( np.exp(1j* np.angle(RX_Y)) )#amplitude-based Gaussian loss L
            
            #X = (1/(rho + 1)) * (X) + ((rho - 1)/(rho + 1)) * Y + (1/(rho + 1)) * (Z)
            X = (X) + 2 * alpha * (Z - Y) #alternative
            x = A_pinv(X) #(ifftn(X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten()
                                                 #But for noiseless measurements, it does help getting the right scale

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
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)  
    if Algo == 'Poisson-DRS in OS + TV1': #In comparison to AAR, DRS allows a relaxation parameter on the magnitudes constraints, using proximal operators
                               #In comparison to AAR, DRS allows a relaxation parameter on the magnitudes constraints, using proximal operators
                               #pseudoinverse of A discards any vector component which is orthogonal to the range of A, and by so doing annihilates the kernel of A.
        (rho, alpha) = rho_Gau_Poi[1] ############## rho = 0 gives AAR ##################
        for k in range(maxiter):
        
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()
            y = (P_S(x))#.flatten() # (fftn((x * mask.reshape((n,))).reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            r_S_x = 2 * y - x 
            RX_Y = Af(r_S_x).flatten()
            Z = ( (-1) * (rho/(2*(rho + 2))) * np.abs(RX_Y) + (1/(2)) * ( ( ((rho**2/((rho + 2)**2)) * np.abs(RX_Y)**2 
                                                                    + ((8)/(rho + 2)) * meas) )**(0.5) ) ) * ( np.exp(1j* np.angle(RX_Y)) )#amplitude-based Gaussian loss L
            z = A_pinv(Z) #crucial difference
            #X = (1/(rho + 1)) * (X) + ((rho - 1)/(rho + 1)) * Y + (1/(rho + 1)) * (Z) 
            x = (x) + 2 * alpha * (z - y) #alternative
                                                 #But for noiseless measurements, it does help getting the right scale

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
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)   
    if Algo == 'Gaussian-DRS': #rho = 0 boils down to AAR. Other values of rho converge slowly and ressemble error reduction, the more rho gets to 1
                               
                               #pseudoinverse of A discards any vector component which is orthogonal to the range of A, and by so doing annihilates the kernel of A. Though A here is unital
        (rho, alpha) = rho_Gau_Poi[0]
        for k in range(maxiter):
         
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            
            (Qx, Qy) = X.shape
            X = X.flatten()
            X_save = X
            #X = (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            Y = Af(P_S(x)).flatten() #(fftn((x * mask.reshape((n,))).reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            RX_Y = 2 * Y - X
            Z = (1) * (1/(rho + 1)) * (meas**(0.5)) * np.exp(1j* np.angle(RX_Y)) + (rho/(rho + 1)) * RX_Y #amplitude-based Gaussian loss L. This is sign dependent
            X = (X) + 2 * alpha * (Z - Y) # (1/(rho + 1)) * (X) + ((rho - 1)/(rho + 1)) * Y + (1/(rho + 1)) * (Z)   #bad if prho >0        # (X) + 2 * alpha * (Z - Y) # performs better IEEE
            #X = .5 * X  + ((rho - 1)/2 * (rho + 1)) * Y + (1/(rho + 1)) * (Z)
            #X = alpha * X + (1 - alpha) * ((1/(rho + 1)) * (meas**(0.5)) * np.exp(1j* np.angle(X_save)) + (rho/(rho + 1)) * X_save) RAAR
            x = A_pinv(X) #Af.pinv(X) # #(ifftn(X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten()
                                                       #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k) 
    if Algo == 'Gaussian-DRS + TVS_M': #In comparison to AAR, DRS allows a relaxation parameter on the magnitudes constraints, using proximal operators
                               #pseudoinverse of A discards any vector component which is orthogonal to the range of A, and by so doing annihilates the kernel of A.
        (rho, alpha) = rho_Gau_Poi[1] ############## rho = 0 gives AAR ##################
        for k in range(maxiter):
      
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()

            Y = (1) * (1/(rho + 1)) * (meas**(0.5)) * np.exp(1j* np.angle(X)) + (rho/(rho + 1)) * X #amplitude-based Gaussian loss L. This is sign dependent
            R_M_X = 2 * Y - X
            x = A_pinv(R_M_X)
            #'''
            ########## TV Regularizaton ####################
            y_real = space.element(x.real.reshape(mask.shape))
            y_imag = space.element(x.imag.reshape(mask.shape))
            xr = x0
            xi = x0
            TVregularize(y_real, TvAlpha, Mask, xr, space, niter = TvIter, supportPrior = 'yes') 
            if np.linalg.norm(y_imag) > 1e-3:
                TVregularize(y_imag, TvAlpha, Mask, xi, space, niter = TvIter, supportPrior = 'yes') 
            
            lincomb = odl.LinCombOperator(space, 1, 1.j)
            XX = odl.ProductSpace(space, space)
            xx = XX.element([xr, xi])
            x = space.element(x.reshape(mask.shape))
            lincomb(xx, out = x) 
            x = op(x)
            x = x.__array__()

            Z = (Af(x.flatten())).flatten()
            X = (X) + 2 * alpha * (Z - Y) #alternative
            x = A_pinv(X) #(ifftn(X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten()
                                                 #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k) 
    if Algo == 'Gaussian-DRS + M_TVS': #support in TV
         #rho = 0 boils down to AAR. Other values of rho converge slowly and ressemble error reduction, the more rho gets to 1
                               
                               #pseudoinverse of A discards any vector component which is orthogonal to the range of A, and by so doing annihilates the kernel of A. Though A here is unital
        (rho, alpha) = rho_Gau_Poi[0]
        for k in range(maxiter):
           
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            
            (Qx, Qy) = X.shape
            X = X.flatten()
            X_save = X
         
            ########################### New TV ###########################
            #'''
            ########## TV Regularizaton ####################
            y_real = space.element(x.real.reshape(mask.shape))
            y_imag = space.element(x.imag.reshape(mask.shape))
            xr = x0
            xi = x0
            TVregularize(y_real, TvAlpha, Mask, xr, space, niter = TvIter, supportPrior = 'yes') 
            if np.linalg.norm(y_imag) > 1e-3:
                TVregularize(y_imag, TvAlpha, Mask, xi, space, niter = TvIter, supportPrior = 'yes') 
            lincomb = odl.LinCombOperator(space, 1, 1.j)
            XX = odl.ProductSpace(space, space)
            xx = XX.element([xr, xi])
            x = space.element(x.reshape(mask.shape))
            lincomb(xx, out = x) 
            x = op(x)
            x = x.__array__()

            Y = (Af(x.flatten())).flatten()

            ########################### End New TV ###########################
          
            RX_Y = 2 * Y - X
            Z = (1) * (1/(rho + 1)) * (meas**(0.5)) * np.exp(1j* np.angle(RX_Y)) + (rho/(rho + 1)) * RX_Y #amplitude-based Gaussian loss L. This is sign dependent
            X = (X) + 2 * alpha * (Z - Y) # (1/(rho + 1)) * (X) + ((rho - 1)/(rho + 1)) * Y + (1/(rho + 1)) * (Z)   #bad if prho >0        # (X) + 2 * alpha * (Z - Y) # performs better IEEE
            #X = .5 * X  + ((rho - 1)/2 * (rho + 1)) * Y + (1/(rho + 1)) * (Z)
            #X = alpha * X + (1 - alpha) * ((1/(rho + 1)) * (meas**(0.5)) * np.exp(1j* np.angle(X_save)) + (rho/(rho + 1)) * X_save) RAAR
            x = A_pinv(X) #Af.pinv(X) # #(ifftn(X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten()
                                                 #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)                 
    if Algo == 'Gaussian-DRS + TV1': 
        #rho = 0 boils down to AAR. Other values of rho converge slowly and ressemble error reduction, the more rho gets to 1
                               
                               #pseudoinverse of A discards any vector component which is orthogonal to the range of A, and by so doing annihilates the kernel of A. Though A here is unital
        (rho, alpha) = rho_Gau_Poi[0]
        for k in range(maxiter):
          
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            
            (Qx, Qy) = X.shape
            X = X.flatten()
            X_save = X
   
            ############ For Previous TV : comment off line below  ##########################
            Y = Af(P_S(x)).flatten() 
            
            RX_Y = 2 * Y - X
            Z = (1) * (1/(rho + 1)) * (meas**(0.5)) * np.exp(1j* np.angle(RX_Y)) + (rho/(rho + 1)) * RX_Y #amplitude-based Gaussian loss L. This is sign dependent
            X = (X) + 2 * alpha * (Z - Y) # (1/(rho + 1)) * (X) + ((rho - 1)/(rho + 1)) * Y + (1/(rho + 1)) * (Z)   #bad if prho >0        # (X) + 2 * alpha * (Z - Y) # performs better IEEE
            #X = .5 * X  + ((rho - 1)/2 * (rho + 1)) * Y + (1/(rho + 1)) * (Z)
            #X = alpha * X + (1 - alpha) * ((1/(rho + 1)) * (meas**(0.5)) * np.exp(1j* np.angle(X_save)) + (rho/(rho + 1)) * X_save) RAAR
            x = A_pinv(X) #Af.pinv(X) # #(ifftn(X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten()
         
            ########## Previous TV Regularizaton ####################
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
                                      
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)  
  
    if Algo == 'Peaceman-Rachford RSRM':

        for k in range(maxiter):
          
            x = R_S( R_M(x) )

            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'Peaceman-Rachford RMRS':

        for k in range(maxiter):
          
            x = R_M( R_S(x) )

            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
   
    if Algo == 'Peaceman-Rachford + TV':

        for k in range(maxiter):
                         

            #x = R_S( R_M(x.real) )  + R_S( R_M(x.imag) ) * 1j
            x = R_S( R_M(x) )
            #x = .5 * x + .5 * R_S( R_M(x) ) #AAR

            ########## TV Regularizaton #################### Please, don't try this at home
            #'''
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
                             
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
  
    if Algo == 'Averaged Alternating Reflections RSRM': 
                                                   # classical Douglas-Rachford algorithm #Problem with phase retrieval : Fourier Magnitude equality constraint defines a non convex set. 
                                                   # Propo 2.1 It would converge to a fix point if constraints sets were all convex closed.
        for k in range(maxiter):
           
           ######  x = .5 * x + .5 * R_S( R_M(x) ) ###### also works ########
            #'''
            #X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            #(Qx, Qy) = X.shape
            #X = X.flatten() # (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            #P_M_X = (meas**(0.5)) * np.exp(1j* np.angle(X))  #Fourier amplitude fitting
            #R_M_X = 2 * P_M_X - X     
            #R_rangeA_R_M_X = R_M_X #A is invertible with inverse A_pinv. So no need for A @ ((A_pinv) @ R_M_X) #proximal point relative to the range of A.    
            #x_new = A_pinv(R_M_X)  # (ifftn(R_M_X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten() #back into object domain
            x_new = 2 * P_M(x) - x
            #break
            #iterates.append(x_new)
            r_s_R_M_x = 2 * P_S(x_new) - x_new # reflexion for support projection
            x = AAR_lambda * x + (1 - AAR_lambda) *  r_s_R_M_x
            
            #x = A_pinv((meas**(0.5)) * np.exp(1j* np.angle((Af(x)).flatten()))) #(Af(x)) should satisfies both constraints according to the convex theory. 
                                                                                         # Simulations show convexity is necessary
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'Averaged Alternating Reflections RSRM + TVS_M': 
                                                   
        for k in range(maxiter):

            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten() # (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()

            Y = (meas**(0.5)) * np.exp(1j* np.angle(X))  #Fourier amplitude fitting
            R_M_X = 2 * Y - X     
            #R_rangeA_R_M_X = R_M_X #A is invertible with inverse A_pinv. So no need for A @ ((A_pinv) @ R_M_X) #proximal point relative to the range of A.    
            x_bef = A_pinv(R_M_X)  # (ifftn(R_M_X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten() #back into object domain
            x = x_bef
            #'''
            ########## TV Regularizaton ####################
            y_real = space.element(x.real.reshape(mask.shape))
            y_imag = space.element(x.imag.reshape(mask.shape))
            xr = x0
            xi = x0
            TVregularize(y_real, TvAlpha, Mask, xr, space, niter = TvIter, supportPrior = 'yes') 
            if np.linalg.norm(y_imag) > 1e-3:
                TVregularize(y_imag, TvAlpha, Mask, xi, space, niter = TvIter, supportPrior = 'yes') 
            
            lincomb = odl.LinCombOperator(space, 1, 1.j)
            XX = odl.ProductSpace(space, space)
            xx = XX.element([xr, xi])
            x = space.element(x.reshape(mask.shape))
            lincomb(xx, out = x) 
            x = op(x)
            x = x.__array__()
            Z = (Af(x.flatten())).flatten()
            X = X + Z - Y
            x = A_pinv(X)

            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'Averaged Alternating Reflections RSRM + M_TVS': #same as 'Averaged Alternating Reflections RMRS + M_TVS'
                                                   
        for k in range(maxiter):
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            X = X.flatten() # (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()

            #'''
            ########## TV Regularizaton ####################
            y_real = space.element(x.real.reshape(mask.shape))
            y_imag = space.element(x.imag.reshape(mask.shape))
            xr = x0
            xi = x0
            TVregularize(y_real, TvAlpha, Mask, xr, space, niter = TvIter, supportPrior = 'yes') 
            if np.linalg.norm(y_imag) > 1e-3:
                TVregularize(y_imag, TvAlpha, Mask, xi, space, niter = TvIter, supportPrior = 'yes') 
            
            lincomb = odl.LinCombOperator(space, 1, 1.j)
            XX = odl.ProductSpace(space, space)
            xx = XX.element([xr, xi])
            x = space.element(x.reshape(mask.shape))
            lincomb(xx, out = x) 
            x = op(x)
            y = x.__array__()

            Y = Af(y)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            Y = Y.flatten() # (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            R_S_X = 2 * Y - X 
            Z = (meas**(0.5)) * np.exp(1j* np.angle(R_S_X))  #Fourier amplitude fitting
            X = X + Z - Y  
            x = A_pinv(X) 
 
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k) 
    if Algo == 'Averaged Alternating Reflections RSRM + TV1': 
                                                   
        for k in range(maxiter):
           
           ######  x = .5 * x + .5 * R_S( R_M(x) ) ###### also works ########
            #'''
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten() # (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            #P_M_X = (meas**(0.5)) * np.exp(1j* np.angle(X))  #Fourier amplitude fitting
            x_new = 2 * P_M(x) - x     
            #R_rangeA_R_M_X = R_M_X #A is invertible with inverse A_pinv. So no need for A @ ((A_pinv) @ R_M_X) #proximal point relative to the range of A.    
            #x_new = A_pinv(R_M_X)  # (ifftn(R_M_X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten() #back into object domain
            
            #r_s_R_M_x = 2 * x_new * mask.reshape((n,)) - x_new # reflexion for support projection
            r_s_R_M_x = 2 * P_S(x_new) - x_new # reflexion for support projection
            x = .5 * x + .5 *  r_s_R_M_x
           
            ########## TV Regularizaton #################### Please, don't try this at home
            
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
  
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
   
    if Algo == 'Averaged Alternating Reflections RMRS': 
                                                   # classical Douglas-Rachford algorithm #Problem with phase retrieval : Fourier Magnitude equality constraint defines a non convex set. 
                                                   # Propo 2.1 It would converge to a fix point if constraints sets were all convex closed.
        for k in range(maxiter):
           
          
            x = AAR_lambda * x + (1 - AAR_lambda) * R_M( R_S(x) ) # #AAR in object domain with support prior (not range(A) prior), performs worse.
                                            #One could maybe formulate an equivalent of range(A) prior in the object domain
     
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'Averaged Alternating Reflections RMRS + M_TVS': 
                                                   
        for k in range(maxiter):
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            X = X.flatten() # (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()

            #'''
            ########## TV Regularizaton ####################
            y_real = space.element(x.real.reshape(mask.shape))
            y_imag = space.element(x.imag.reshape(mask.shape))
            xr = x0
            xi = x0
            TVregularize(y_real, TvAlpha, Mask, xr, space, niter = TvIter, supportPrior = 'yes') 
            if np.linalg.norm(y_imag) > 1e-3:
                TVregularize(y_imag, TvAlpha, Mask, xi, space, niter = TvIter, supportPrior = 'yes') 
            
            lincomb = odl.LinCombOperator(space, 1, 1.j)
            XX = odl.ProductSpace(space, space)
            xx = XX.element([xr, xi])
            x = space.element(x.reshape(mask.shape))
            lincomb(xx, out = x) 
            x = op(x)
            y = x.__array__()

            Y = Af(y)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            Y = Y.flatten() # (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            R_S_X = 2 * Y - X 
            Z = (meas**(0.5)) * np.exp(1j* np.angle(R_S_X))  #Fourier amplitude fitting
            X = X + Z - Y  
            x = A_pinv(X) 
 
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)    
    if Algo == 'Averaged Alternating Reflections RMRS + TV1': 
                                                   
        for k in range(maxiter):
           
           ######  x = .5 * x + .5 * R_S( R_M(x) ) ###### also works ########
            #'''
            X = Af(x)#(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho'))#Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten() # (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()

            #r_s_R_M_x = 2 * x_new * mask.reshape((n,)) - x_new # reflexion for support projection
            r_s_x = 2 * P_S(x) - x # reflexion for support projection

            #P_M_X = (meas**(0.5)) * np.exp(1j* np.angle(X))  #Fourier amplitude fitting
            r_m_r_s_x = 2 * P_M(r_s_x) - r_s_x     
            #R_rangeA_R_M_X = R_M_X #A is invertible with inverse A_pinv. So no need for A @ ((A_pinv) @ R_M_X) #proximal point relative to the range of A.    
            #x_new = A_pinv(R_M_X)  # (ifftn(R_M_X.reshape((Qx, Qy)), s = mask.shape, norm = 'ortho')).flatten() #back into object domain

            x = .5 * x + .5 *  r_m_r_s_x
           
            ########## TV Regularizaton #################### Please, don't try this at home
            
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
  
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'RAAR': 

        for k in range(maxiter):
           
          
            x_AAR = .5 * x + .5 * R_M( R_S(x) ) 

            x = RAAR_beta * x_AAR + (1 - RAAR_beta) * P_M(x)
           
       
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)

    if Algo == 'new Hybrid Input-Output': ############### beta = 1 gives AAR #################

        for k in range(maxiter):

            x_bef = iterates[-1]
            PYx = P_M(x)
            Y_res = alph * P_S(PYx - x)
            X_res = bet * (P_S(PYx) - PYx)
            x = Y_res + X_res + gamm * x


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
            #X = Af(x).flatten() #(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            #X = (meas**(0.5)) * np.exp(1j* np.angle(X))               
            #X = A @ ((A_pinv) @ X)
            x = P_M(x) #(ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
         
            # indices = np.logical_or(np.logical_and(np.angle(x) < 0, mask.reshape((n,))),  
            #                         np.logical_and(np.angle(x) > (np.pi/2), mask.reshape((n,))),
            #                         np.logical_not(mask.reshape((n,))))
            # indices = np.logical_or(np.logical_and(x<0, mask.reshape((n,))), 
            #                     np.logical_not(mask.reshape((n,))))
            indices = np.logical_not(mask.reshape((n,)))
            x[indices] = (x_bef[indices] - HIO_beta * x[indices])
            
            #x = x * mask.reshape((n,)) + (x_bef - HIO_beta * x) * outsideMask.reshape((n,)) 
          
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)

    if Algo == 'Hybrid Input-Output real': ############### beta = 1 gives AAR #################

        for k in range(maxiter):
            
            #''' ########## HIO in object domain - support prior ##########
            iterates[-1] = np.real(x) # real image to initialize
            x_bef = iterates[-1]
            x = P_M(x)#(ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()

            x = np.real(x) + 0.j
            indices = np.logical_or(np.logical_and(x<0, mask.reshape((n,))), 
                                np.logical_not(mask.reshape((n,))))
            x[indices] = x_bef[indices]-HIO_beta*x[indices]
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
  
    if Algo == 'Hybrid Input-Output + TV': ############### beta = 1 gives AAR #################

        for k in range(maxiter):
            
            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
          
            x_bef = iterates[-1]
            X = Af(x).flatten() #(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))               
            #X = A @ ((A_pinv) @ X)
            x = A_pinv(X) #(ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
        
            #x = x * op(mask)
            indices = np.logical_not(mask.reshape((n,)))
            x[indices] = (x_bef[indices] - HIO_beta * x[indices])
    
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


            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
  
    if Algo == 'Output-Output':

        for k in range(maxiter):
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #'''
            x_bef = iterates[-1]
            # X = Af(x).flatten() #(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #A @ np.conjugate(x)
            # X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #/(np.abs(X))) * X
            # x = A_pinv(X)
            x = P_M(x)

            indices = np.logical_not(mask.reshape((n,)))
            x[indices] = (1 - OO_beta ) * x[indices]

            #x = x * mask.reshape((n,)) + (x - OO_beta * x) * outsideMask.reshape((n,))                           
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'basic Input-Output':

        for k in range(maxiter):
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #'''
            x_bef = iterates[-1]
            # X = Af(x).flatten() #(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #A @ np.conjugate(x)
            # X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #/(np.abs(X))) * X
            # x = A_pinv(X) #(ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #np.conjugate(A.T) @ (X)
            x = P_M(x)
            ########## TV Support Regularizaton ####################
            # y = space.element(x.reshape(mask.shape))
            # TVregularize(y, 0.15, Mask, x0, space, niter = 10)
            # x = op(x0)

            #x = x * op(mask)
            indices = np.logical_not(mask.reshape((n,)))
            x[indices] = x_bef[indices] - IO_beta  * x[indices]
            #x = x_bef * mask.reshape((n,)) + (x_bef - IO_beta * x) * outsideMask.reshape((n,))

            #x = soft_shrinkage(x.real, lamda = 0.) + soft_shrinkage(x.imag, lamda = 0.) * 1j # L-1 Regularisation
            #x = (np.sum(meas)**(0.5)/np.linalg.norm(x)) * x #Normalising with Parseval will make us reconstruct the noise
            

            #'''                                             #But for noiseless measurements, it does help getting the right scale
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'Error Reduction : support prior':

        for k in range(maxiter):
   
            # x = x.reshape((n,)) * mask.reshape((n,))
            # X = Af(x).flatten() #(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() # A @(x)

            # X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #attribute measured Fourier Intensities
           
            x = P_M(x) #(ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #np.conjugate(A.T) @ (X)

            x = P_S(x)

            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'Error Reduction + TV-support': #to revisit

        for k in range(maxiter):
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #'''
            x = x.reshape((n,)) * mask.reshape((n,))
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
            # X = Af(x).flatten() # (fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #A @ np.conjugate(x)
            # X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #/(np.abs(X))) * X
            x = P_M(x) # (ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten() #np.conjugate(A.T) @ (X)
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
            # X = Af(x).flatten() #(fftn(x.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            # X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #/(np.abs(X))) * X
            # #X = A @ ((A_pinv) @ X) #proximal point relative to the range of A. It finds the true support better than a direct support estimation prior appplication. Why?
            # x = A_pinv(X) #(ifftn(X.reshape(mask.shape), s = mask.shape, norm = 'ortho')).flatten()
            x = P_M(x)
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