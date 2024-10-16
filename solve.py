import numpy as np
import odl
import scipy.misc
from Operators import soft_shrinkage
from TVRegularise import TVregularize

beta = .7
def P_S(x, mask, n):
    return x * mask.reshape((n,))
def P_M (x, meas, A):
    X = A @ np.conjugate(x)
    X = (meas**(0.5)) * np.exp(1j* np.angle(X))
    return np.conjugate(A.T) @ (X)
gamma_M = -1/beta
def R_M(x):
    return (1+gamma_M)*P_M(x) - gamma_M*x
gamma_S = 1/beta
def R_S(x):
    return (1+gamma_S)*P_S(x) - gamma_S*x



def phase_retrieval(L, kappa, xi, Algo, map, mask, n, A, meas, maxiter, x ):
    
    space = odl.uniform_discr(min_pt=[0, 0], max_pt=mask.shape, shape=mask.shape, dtype='complex64')
    x0 = space.zero()
    Mask = space.element(mask)
    op = odl.FlatteningOperator(space)

    iterates = []
    iterates.append(x)
    
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

                    x = x * mask.reshape((n,))
                    ############ TV Support Regularization ############
                    # y = space.element(x.reshape(mask.shape))
                    # TVregularize(y, 0.15, Mask, x0, space, niter = 10)
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
    if Algo == 'FIENUP':

        for k in range(maxiter):
            #x = x + beta * (P_S(R_M(x)) - P_M(R_S(x)))
            #'''
            X = A @ np.conjugate(x)
            X = (meas**(0.5)) * np.exp(1j* np.angle(X))               #/(np.abs(X))) * X
            x = np.conjugate(A.T) @ (X)

            ########## TV Support Regularizaton ####################
            # y = space.element(x.reshape(mask.shape))
            # TVregularize(y, 0.15, Mask, x0, space, niter = 10)
            # x = op(x0)

            #x = x * op(mask)
            x = x * mask.reshape((n,))
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