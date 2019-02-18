import time
import numpy as np
from tqdm import trange
import scipy.sparse as sp
from scipy.linalg import norm
from vezda.math_utils import humanReadable
from vezda.svd_utils import load_svd, svd_needs_recomputing, compute_svd
from vezda.LinearOperators import asConvolutionalOperator

#==============================================================================
# Super class (Parent class)
# A class for solving linear systems Ax = b
#
# Class data objects: 
#   linear operator: A
#   right-hand side vectors: B = [b1 b2 ... bn]
#
# Class methods:
#   solve by singular-value decomposition: solve_svd
#   solve by least-squares mininum residual: solve_lsmr
#   solve by conjugate gradient: solve_cg (A^HAx = A^Hb)
#==============================================================================
class LinearSystem(object):
    
    def __init__(self, LinearOperator, rhs_vectors):
        self.A = LinearOperator
        self.B = rhs_vectors
        
        
    def solve_svd(self, U, s, Vh, alpha=0.0):
        #======================================================================
        # Construct the pseudoinverse of A : A+ = V Sp Uh
        if np.issubdtype(U.dtype, np.complexfloating):
            # singular vectors are complex
            Uh = U.getH()
            V = Vh.getH()
        else:
            # singular vectors are real
            Uh = U.T
            V = Vh.T
        
        # Construct the diagonal matrix 'Sp' from 's'
        s = np.divide(s, alpha + s**2)
        Sp = sp.diags(s)
        #======================================================================
        # Apply SVD to obtain solution matrix X
        M, N = self.A.shape
        K = self.B.shape[2]
        
        # initialize solution matrix X
        X = np.zeros((N, K), dtype=self.A.dtype)
        
        # Opportunity to parallelize here...
        startTime = time.time()
        for i in trange(K):
            b = self.B[:, :, i].reshape(M)
            #print('b.shape =', b.shape)
            X[:, i] = V.dot(Sp.dot(Uh.dot(b)))
        endTime = time.time()
        print('Elapsed time:', humanReadable(endTime - startTime))
        
        return X
        
        
    def solve_lsmr(self, alpha=0.0, tol=1.0e-4):
        M, N = self.A.shape
        K = self.B.shape[2]
        
        # initialize solution matrix X
        X = np.zeros((N, K), dtype=self.A.dtype)
        
        # initialize array for convergence data
        # R[:, 0] = index for stop reason
        # R[:, 1] = number of iterations performed
        # R[:, 2] = l2 norm of residual vector
        #R = np.zeros((K, 3))
        
        # Opportunity to parallelize here...
        startTime = time.time()
        for i in trange(K):
            b = self.B[:, :, i].reshape(M)
            X[:, i] = sp.linalg.lsmr(self.A, b, damp=alpha, atol=tol)[0]
            #X[:, i], R[i, :] = results[0], results[-3:]
        endTime = time.time()
        print('Elapsed time:', humanReadable(endTime - startTime))
        
        return X
    
        
    def solve_cg(self, tol=1.0e-4):
        M, N = self.A.shape
        K = self.B.shape[2]
        
        # initialize solution matrix X
        X = np.zeros((N, K), dtype=self.A.dtype)
        
        # Construct SPD matrix AhA
        Ah = self.A.adjoint()
        AhA = Ah.dot(self.A)
        # Opportunity to parallelize here...
        startTime = time.time()
        for i in trange(K):
            b = self.B[:, :, i].reshape(M)
            X[:, i] = sp.linalg.cg(AhA, Ah.dot(b), tol=tol)[0]
        endTime = time.time()
        print('Elapsed time:', humanReadable(endTime - startTime))
        
        return X
    
    
#==============================================================================
# Subclass (Derived class)
# A class for solving linear sampling problems of the form Ax = b
#
# Class data objects: 
#   kernel: data or test functions
#   right-hand side vectors: B = [b1 b2 ... bn]
#
# Class methods:
#   solve system of equations using specified method: solve(method)
#   construst image from solutions: construct_image()
#==============================================================================
class LinearSamplingProblem(LinearSystem):
    
    def __init__(self, operatorName, kernel, rhs_vectors):
        super().__init__(asConvolutionalOperator(kernel), rhs_vectors)
        self.operatorName = operatorName
        self.kernel = kernel
        
        
    def solve(self, method, alpha=0.0, tol=1.0e-4, k=None):
        '''
        method : specified direct or iterative method for solving Ax = b
        alpha : regularization parameter
        tol : tolerance for convergence of iterative methods
        k : number of singular values/vectors to compute
        '''
        #======================================================================
        if method == 'svd':
            # Load or recompute the SVD of A as needed
        
            if self.operatorName == 'nfo':
                filename = 'NFO_SVD.npz'
            elif self.operatorName == 'lso':
                filename = 'LSO_SVD.npz'
        
            try:
                U, s, Vh = load_svd(filename)
                if svd_needs_recomputing(self.kernel, k, U, s, Vh):
                    U, s, Vh = compute_svd(self.kernel, k, self.operatorName)
            except IOError:
                if k is None:
                    k = input('Specify the number of singular values and vectors to compute: ')
                U, s, Vh = compute_svd(self.kernel, k, self.operatorName)
            
            print('Localizing the source function...')
            return super().solve_svd(U, s, Vh, alpha)
                
        elif method == 'lsmr':
            print('Localizing the source function...')
            return super().solve_lsmr(alpha, tol)
        
        elif method == 'cg':
            print('Localizing the source function...')
            return super().solve_cg(tol)
            
    
    def construct_image(self, solutions):
        print('Constructing the image...')
        # Get machine precision
        eps = np.finfo(float).eps     # about 2e-16 (used in division
                                      # so we never divide by zero)
        if self.operatorName == 'nfo':
            Image = 1.0 / (norm(solutions, axis=0) + eps)
            
            # Normalize Image to take on values between 0 and 1
            Imin = np.min(Image)
            Imax = np.max(Image)
            Image = (Image - Imin) / (Imax - Imin + eps)
            
        elif self.operatorName == 'lso':
            Nm, Nsp = self.kernel.shape[1], self.kernel.shape[2]
            K = solutions.shape[1]
            solutions = solutions.reshape((Nsp, Nm, K))
            
            # Initialize the Image
            Image = np.zeros(Nsp)
            for i in range(K):
                indicator = norm(solutions[:, :, i], axis=1)
                Imin = np.min(indicator)
                Imax = np.max(indicator)
                indicator = (indicator - Imin) / (Imax - Imin + eps)
                Image += indicator**2
        
            # Image is defined as the root-mean-square indicator
            Image = np.sqrt(Image / K)
            
        return Image