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

    Mask = space.element(mask)
    op = odl.FlatteningOperator(space)

    iterates = []
    iterates.append(x) 
    
    
    beta = .7
    def P_S(x):
        x = x.copy()
        indices = np.logical_not(mask.reshape((n,)))
        x[indices] = 0
        return x 
    def P_M(x):
        X = Af(x).flatten() 
        X = (meas**(0.5)) * np.exp(1j* np.angle(X))
        return A_pinv(X) 
    gamma_M = 1 
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
            if l == k : 
                L /= xi
                gamma = (1+kappa)#/(L)#*mag 

                x = iterates[k]
                grad_psi_x = np.asarray(map.grad_psi(x))
                grad_f_x = np.asarray(map.grad_f(x))

                z = grad_psi_x - gamma * grad_f_x #mirror gradient descent step
                x_temp  = a * map.grad_psi_star(z) + b * ( map.grad_psi_star(z.real) +  map.grad_psi_star(z.imag) * 1j ) #mirror step

                
                if 1 < 0: 
                    gamma = (1-kappa)/L 
                    x = iterates[k]
                    grad_psi_x = np.asarray(map.grad_psi(x))
                    grad_f_x = np.asarray(map.grad_f(x))
                    z = grad_psi_x - gamma * grad_f_x
                    x = a * map.grad_psi_star(z) + b * ( map.grad_psi_star(z.real) +  map.grad_psi_star(z.imag) * 1j )  # map.grad_psi_star(z)

        
                    iterates.append(x)
                    L *= (xi**(int(j/maxiter)) + 1) 
                    break  
                x = x_temp
                iterates.append(x)

                #print("ho")
                j += 1 
                #if j % 10 == 0:
                    
            if k % 1 == 0:
            
               print('iterate', k+1)

    if Algo == 'Poisson-DRS': 
        (rho, alpha) = rho_Gau_Poi[1] 
        for k in range(maxiter):
        
            X = Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()
            Y = Af(P_S(x)).flatten() 
            RX_Y = 2 * Y - X
            Z = ( (-1) * (rho/(2*(rho + 2))) * np.abs(RX_Y) + (1/(2)) * ( ( ((rho**2/((rho + 2)**2)) * np.abs(RX_Y)**2 
                                                                    + ((8)/(rho + 2)) * meas) )**(0.5) ) ) * ( np.exp(1j* np.angle(RX_Y)) )
     
            X = (X) + 2 * alpha * (Z - Y) 

            x = A_pinv(X) 
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)  
    if Algo == 'Poisson-DRS in OS': 
        (rho, alpha) = rho_Gau_Poi[1] 
        for k in range(maxiter):
        
            X = Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()
            y = (P_S(x))
            r_S_x = 2 * y - x 
            RX_Y = Af(r_S_x).flatten()
            Z = ( (-1) * (rho/(2*(rho + 2))) * np.abs(RX_Y) + (1/(2)) * ( ( ((rho**2/((rho + 2)**2)) * np.abs(RX_Y)**2 
                                                                    + ((8)/(rho + 2)) * meas) )**(0.5) ) ) * ( np.exp(1j* np.angle(RX_Y)) )
            z = A_pinv(Z)
            
            x = (x) + 2 * alpha * (z - y) 
            
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)  
    if Algo == 'Poisson-DRS + TVS_M': 
        (rho, alpha) = rho_Gau_Poi[1]
        for k in range(maxiter):
      
            X = Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()

            Y = ( (-1) * (rho/(2*(rho + 2))) * np.abs(X) + (1/(2)) * ((((rho**2/((rho + 2)**2)) * np.abs(X)**2 
                                                                        + ((8)/(rho + 2)) * meas))**(0.5))) * ( np.exp(1j* np.angle(X)) )
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
            X = (X) + 2 * alpha * (Z - Y)
            x = A_pinv(X) 
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k) 
    if Algo == 'Poisson-DRS in OS + TVS_M': 
        (rho, alpha) = rho_Gau_Poi[1] 
        for k in range(maxiter):
      
            X = Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()

            Y = ( (-1) * (rho/(2*(rho + 2))) * np.abs(X) + (1/(2)) * ((((rho**2/((rho + 2)**2)) * np.abs(X)**2 
                                                                        + ((8)/(rho + 2)) * meas))**(0.5))) * ( np.exp(1j* np.angle(X)) )
            y = A_pinv(Y)
            r_M_x = 2 * y - x
            x_bef = x
            x = r_M_x

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

            z = x
            x = x_bef + 2 * alpha * (z - y)

            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k) 
    if Algo == 'Poisson-DRS + M_TVS': 
        (rho, alpha) = rho_Gau_Poi[1]
        for k in range(maxiter):
      
            X = Af(x)
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

            Y = (Af(x.flatten())).flatten()          
            RX_Y = 2 * Y - X
            Z = ( (-1) * (rho/(2*(rho + 2))) * np.abs(RX_Y) + (1/(2)) * ((((rho**2/((rho + 2)**2)) * np.abs(RX_Y)**2 
                                                                           + ((8)/(rho + 2)) * meas))**(0.5))) * ( np.exp(1j* np.angle(RX_Y)) )#amplitude-based Gaussian loss L
            
            
            X = (X) + 2 * alpha * (Z - Y) 
            x = A_pinv(X) 
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)  
    if Algo == 'Poisson-DRS in OS + M_TVS':
        (rho, alpha) = rho_Gau_Poi[1] 
        for k in range(maxiter):
            x_bef = x
            X = Af(x)
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
                   
            RX_Y = (Af(r_s_x.flatten())).flatten()
            Z = ( (-1) * (rho/(2*(rho + 2))) * np.abs(RX_Y) + (1/(2)) * ((((rho**2/((rho + 2)**2)) * np.abs(RX_Y)**2 
                                                                           + ((8)/(rho + 2)) * meas))**(0.5))) * ( np.exp(1j* np.angle(RX_Y)) )#amplitude-based Gaussian loss L
            z = A_pinv(Z)
            
            x = x_bef + 2 * alpha * (z - y)
            
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)  
    if Algo == 'Poisson-DRS + TV1': 
        (rho, alpha) = rho_Gau_Poi[1] 
        for k in range(maxiter):
      
            X = Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()

            Y = (Af(P_S(x))).flatten()
            
            RX_Y = 2 * Y - X


            Z = ( (-1) * (rho/(2*(rho + 2))) * np.abs(RX_Y) + (1/(2)) * ((((rho**2/((rho + 2)**2)) * np.abs(RX_Y)**2 + ((8)/(rho + 2)) * meas))**(0.5))) * ( np.exp(1j* np.angle(RX_Y)) )#amplitude-based Gaussian loss L
            
           
            X = (X) + 2 * alpha * (Z - Y) 
            x = A_pinv(X) 
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
    if Algo == 'Poisson-DRS in OS + TV1': 
        (rho, alpha) = rho_Gau_Poi[1] 
        for k in range(maxiter):
        
            X = Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()
            y = (P_S(x))
            r_S_x = 2 * y - x 
            RX_Y = Af(r_S_x).flatten()
            Z = ( (-1) * (rho/(2*(rho + 2))) * np.abs(RX_Y) + (1/(2)) * ( ( ((rho**2/((rho + 2)**2)) * np.abs(RX_Y)**2 
                                                                    + ((8)/(rho + 2)) * meas) )**(0.5) ) ) * ( np.exp(1j* np.angle(RX_Y)) )#amplitude-based Gaussian loss L
            z = A_pinv(Z) 
            x = (x) + 2 * alpha * (z - y) 
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
    if Algo == 'Gaussian-DRS': 
        (rho, alpha) = rho_Gau_Poi[0]
        for k in range(maxiter):
         
            X = Af(x)
            
            (Qx, Qy) = X.shape
            X = X.flatten()
            X_save = X
            
            Y = Af(P_S(x)).flatten() 
            RX_Y = 2 * Y - X
            Z = (1) * (1/(rho + 1)) * (meas**(0.5)) * np.exp(1j* np.angle(RX_Y)) + (rho/(rho + 1)) * RX_Y 
            X =  (X) + 2 * alpha * (Z - Y) 
            x = A_pinv(X) 
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k) 
    if Algo == 'Gaussian-DRS + TVS_M':
        (rho, alpha) = rho_Gau_Poi[1]
        for k in range(maxiter):
      
            X = Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()

            Y = (1) * (1/(rho + 1)) * (meas**(0.5)) * np.exp(1j* np.angle(X)) + (rho/(rho + 1)) * X  
            R_M_X = 2 * Y - X
            x = A_pinv(R_M_X)
            
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
            X = (X) + 2 * alpha * (Z - Y)
            x = A_pinv(X)
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k) 
    if Algo == 'Gaussian-DRS + M_TVS':

        (rho, alpha) = rho_Gau_Poi[0]
        for k in range(maxiter):
           
            X = Af(x)
            
            (Qx, Qy) = X.shape
            X = X.flatten()
            X_save = X

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
            Z = (1) * (1/(rho + 1)) * (meas**(0.5)) * np.exp(1j* np.angle(RX_Y)) + (rho/(rho + 1)) * RX_Y
            x = A_pinv(X) 
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)                 
    if Algo == 'Gaussian-DRS + TV1': 

        (rho, alpha) = rho_Gau_Poi[0]
        for k in range(maxiter):
          
            X = Af(x)
            
            (Qx, Qy) = X.shape
            X = X.flatten()
            X_save = X
   
            ############ For Previous TV : comment off line below  ##########################
            Y = Af(P_S(x)).flatten() 
            
            RX_Y = 2 * Y - X
            Z = (1) * (1/(rho + 1)) * (meas**(0.5)) * np.exp(1j* np.angle(RX_Y)) + (rho/(rho + 1)) * RX_Y 
            X = (X) + 2 * alpha * (Z - Y)
            x = A_pinv(X)
         
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
                         

            
            x = R_S( R_M(x) )
            
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
                                                   
        for k in range(maxiter):
           
           
            x_new = 2 * P_M(x) - x
            
            r_s_R_M_x = 2 * P_S(x_new) - x_new 
            x = AAR_lambda * x + (1 - AAR_lambda) *  r_s_R_M_x
            
            
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'Averaged Alternating Reflections RSRM + TVS_M': 
                                                   
        for k in range(maxiter):

            X = Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten() 

            Y = (meas**(0.5)) * np.exp(1j* np.angle(X)) 
            R_M_X = 2 * Y - X     
          
            x_bef = A_pinv(R_M_X)  
            x = x_bef
            
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
    if Algo == 'Averaged Alternating Reflections RSRM + M_TVS': 
                                                   
        for k in range(maxiter):
            X = Af(x)
            X = X.flatten() 


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

            Y = Af(y)
            Y = Y.flatten()
            R_S_X = 2 * Y - X 
            Z = (meas**(0.5)) * np.exp(1j* np.angle(R_S_X))  
            X = X + Z - Y  
            x = A_pinv(X) 
 
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k) 
    if Algo == 'Averaged Alternating Reflections RSRM + TV1': 
                                                   
        for k in range(maxiter):

            X = Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten()
            x_new = 2 * P_M(x) - x     
            
            r_s_R_M_x = 2 * P_S(x_new) - x_new 
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
                                                   
        for k in range(maxiter):
           
          
            x = AAR_lambda * x + (1 - AAR_lambda) * R_M( R_S(x) ) 
     
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'Averaged Alternating Reflections RMRS + M_TVS': 
                                                   
        for k in range(maxiter):
            X = Af(x)
            X = X.flatten() 


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

            Y = Af(y)
            Y = Y.flatten()
            R_S_X = 2 * Y - X 
            Z = (meas**(0.5)) * np.exp(1j* np.angle(R_S_X))
            X = X + Z - Y  
            x = A_pinv(X) 
 
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)    
    if Algo == 'Averaged Alternating Reflections RMRS + TV1': 
                                                   
        for k in range(maxiter):
           

            X = Af(x)
            (Qx, Qy) = X.shape
            X = X.flatten() 
            r_s_x = 2 * P_S(x) - x 
            r_m_r_s_x = 2 * P_M(r_s_x) - r_s_x     
            
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

    if Algo == 'new Hybrid Input-Output': 

        for k in range(maxiter):

            x_bef = iterates[-1]
            PYx = P_M(x)
            Y_res = alph * P_S(PYx - x)
            X_res = bet * (P_S(PYx) - PYx)
            x = Y_res + X_res + gamm * x


            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)  
    if Algo == 'Hybrid Input-Output':
        for k in range(maxiter):
          

            x_bef = iterates[-1]
            
            x = P_M(x) 
            indices = np.logical_not(mask.reshape((n,)))
            x[indices] = (x_bef[indices] - HIO_beta * x[indices])
            
            
          
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)

    if Algo == 'Hybrid Input-Output real': 

        for k in range(maxiter):
            
            #''' ########## HIO in object domain - support prior ##########
            iterates[-1] = np.real(x) 
            x_bef = iterates[-1]
            x = P_M(x)

            x = np.real(x) + 0.j
            indices = np.logical_or(np.logical_and(x<0, mask.reshape((n,))), 
                                np.logical_not(mask.reshape((n,))))
            x[indices] = x_bef[indices]-HIO_beta*x[indices]
            #'''
 
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
  
    if Algo == 'Hybrid Input-Output + TV':

        for k in range(maxiter):
    
            x_bef = iterates[-1]
            X = Af(x).flatten() 
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))               
        
            x = A_pinv(X)
        
     
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

            x_bef = iterates[-1]
  
            x = P_M(x)

            indices = np.logical_not(mask.reshape((n,)))
            x[indices] = (1 - OO_beta ) * x[indices]

                   
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'basic Input-Output':

        for k in range(maxiter):
 
            x_bef = iterates[-1]
 
            x = P_M(x)

            indices = np.logical_not(mask.reshape((n,)))
            x[indices] = x_bef[indices] - IO_beta  * x[indices]

            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'Error Reduction : support prior':

        for k in range(maxiter):
   
    
            x = P_M(x) 

            x = P_S(x)

            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'Error Reduction + TV-support': #to revisit

        for k in range(maxiter):

            x = x.reshape((n,)) * mask.reshape((n,))
            X = Af(x).flatten() 
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))          
            y = A_pinv(X) 
            z = 2 * y  - x 
            ########## TV Support Regularizaton ####################
            t = space.element(z.reshape(mask.shape)) 
        
            xr = x0.real
            xi = x0.real
            TVregularize(t, TvAlpha, Mask, x0, space, niter = TvIter, supportPrior = 'no') 
            #TVregularize(y.imag, 0.15, Mask, xi, space, niter = TvIter) 
            x = x + op(x0) - y
            x = x.__array__()

      
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k) 
    if Algo == 'Gerchberg-Saxton':

        for k in range(maxiter):
            x = P_M(x)
            x = np.abs(x_true_vect) * np.exp(1j* np.angle(x))

            iterates.append(x)
            if k % 100 == 0: 
                print('iteration k', k) 
    if Algo == 'Gerchberg-Saxton + TV-support':

        for k in range(maxiter):
         
            X = Af(x).flatten()
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))             
            x = A_pinv(X) 
            x = np.abs(x_true_vect) * np.exp(1j* np.angle(x))

            ########## TV Support Regularizaton ####################
            y = space.element(x.reshape(mask.shape)) 
            xr = x0.real
            xi = x0.imag
            TVregularize(y, TvAlpha, Mask, x0, space, niter = TvIter, supportPrior = 'yes') 
            #TVregularize(space.element(y.imag), 0.15, Mask, xi, space, niter = TvIter) 
            x = op(x0)

            
            iterates.append(x)
            if k % 100 == 0: 
                print('iteration k', k)            
    if Algo == 'No Support prior - fast':

        for k in range(maxiter):
        
            x = P_M(x)
        
           
            iterates.append(x)
            if k % 100 == 0: 
                  print('iteration k', k)
    if Algo == 'No Support prior - slow':

        for k in range(maxiter):
           
            X = A @ np.conjugate(x)
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))            
            x = np.conjugate(A.T) @ (X)

         
            iterates.append(x)
            if k % 100 == 0: 
                print('iteration k', k)
    if Algo == 'No Support prior + TV-support real_imag_separated':

        for k in range(maxiter):

            X = Af(x).flatten()
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))         
            x = A_pinv(X) 

            ########## TV Support Regularizaton ####################
           
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
          
            x = x.__array__()
            iterates.append(x)
            #x = x.flatten()
            if k % 100 == 0: 
                print('iteration k', k)
    if Algo == 'No Support prior + TV-support':

        for k in range(maxiter):
       
            X = Af(x).flatten()
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))           
            x = A_pinv(X) 

            ########## TV Support Regularizaton ####################
           
            y = space.element(x.reshape(mask.shape))
          
            TVregularize(y, TvAlpha, Mask, x0, space, niter = TvIter, supportPrior = 'yes') 
            #TVregularize(y_imag, 1., Mask, xi, space, niter = TvIter) 
            x = op(x0).__array__() #+ op(xi).__array__() * 1j

          
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