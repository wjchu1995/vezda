import sys
import time
import numpy as np
import scipy.sparse as sp
from vezda.math_utils import humanReadable
from vezda.LinearOperators import asConvolutionalOperator

def compute_svd(kernel, k, operatorName):
    A = asConvolutionalOperator(kernel)
    
    if k_is_valid(k, min(A.shape)):
        if operatorName == 'nfo':
            name = 'near-field operator'
        elif operatorName == 'lso':
            name = 'Lippmann-Schwinger operator'
        
        if k == 1:
            print('Computing SVD of the %s for 1 singular value/vector...' %(name))
        else:
            print('Computing SVD of the %s for %s singular values/vectors...' %(name, k))
        
        startTime = time.time()
        U, s, Vh = sp.linalg.svds(A, k, which='LM')
        endTime = time.time()
        print('Elapsed time:', humanReadable(endTime - startTime))
        
        # sort the singular values and corresponding vectors in descending order
        # (i.e., largest to smallest)
        index = s.argsort()[::-1]   
        s = s[index]
        U = U[:, index]
        Vh = Vh[index, :]
        
        if np.issubdtype(U.dtype, np.complexfloating):
            # Exploit sparseness of SVD in frequency domain for efficient storage
            Nr, Ns = kernel.shape[0], kernel.shape[2]
            U = make_sparse(U, Nr, 'csc')
            Vh = make_sparse(Vh, Ns, 'csr')
    
        save_svd(U, s, Vh, operatorName)
        
        return U, s, Vh
    
    else:
        sys.exit()


def save_svd(U, s, Vh, operatorName):
    
    if operatorName == 'nfo':
        filename = 'NFO_SVD.npz'
    elif operatorName == 'lso':
        filename = 'LSO_SVD.npz'
    
    if np.issubdtype(U.dtype, np.complexfloating): 
        # singular vectors are complex
        # store as sparse matrices
        domain = 'freq'
        np.savez(filename,
                 U_data=U.data, U_indices=U.indices, U_indptr=U.indptr,
                 U_shape=U.shape,
                 Vh_data=Vh.data, Vh_indices=Vh.indices, Vh_indptr=Vh.indptr,
                 Vh_shape=Vh.shape,
                 s=s, domain=domain)
    
    else:
        # singular vectors are real
        domain = 'time'
        np.savez(filename, U=U, s=s, Vh=Vh, domain=domain)        
    

def load_svd(filename):
    print('Attempting to load SVD...', end='')
    try:
        loader = np.load(filename)
    except:
        raise IOError('A singular-value decomposition does not exist...')
    print('Success')
    s = loader['s']
    domain = loader['domain']
    
    if domain == 'freq':
        U = sp.csc_matrix((loader['U_data'], loader['U_indices'], loader['U_indptr']),
                       shape=loader['U_shape'])
        Vh = sp.csr_matrix((loader['Vh_data'], loader['Vh_indices'], loader['Vh_indptr']),
                        shape=loader['Vh_shape'])
    
    elif domain == 'time':
        U = loader['U']
        Vh = loader['Vh']
    
    return U, s, Vh
    

def make_sparse(A, r, compressedFormat):
    '''
    Return a sparse representation of a matrix A based on the r largest nonzero
    row/column elements.
    
    A: a dense matrix (2D array) to make sparse
    r: a positive integer giving the number of nonzero row/column elements to extract from A
    compressedFormat: compressed storage format of resulting sparse matrix.
    
    'csc' format results in a sparse column matrix. Extracts the r largest nonzero
    elements along each column of A. Best for 'tall' matrices.
    
    'csr' format results in a sparse row matrix. Extracts the r largest nonzero
    elements along each row of A. Best for 'wide' matrices.
    '''    
    M, N = A.shape
    
    if compressedFormat == 'csc':
        indx = np.argpartition(-np.abs(A), r, axis=0)[:r, :]
        data = A[indx, np.arange(N)].reshape(r * N, order='F')
        
        rows = indx.reshape(-1, order='F')
        cols = (np.ones((N, r), np.int_) * np.arange(N)[:, None]).reshape(-1)
        
        return sp.csc_matrix((data, (rows, cols)), shape=(M, N))
        
    
    elif compressedFormat == 'csr':
        indx = np.argpartition(-np.abs(A), r, axis=1)[:, :r]
        data = A[np.arange(M)[:, None], indx].reshape(r * M)

        rows = (np.ones((M, r), np.int_) * np.arange(M)[:, None]).reshape(-1)
        cols = indx.reshape(-1)
        
        return sp.csr_matrix((data, (rows, cols)), shape=(M, N))
    

def k_is_valid(k, maxVals):
    if type(k) == int:            
        if k >= 1 and k < maxVals:
            return True
        else:
            print('''Number of singular values must be a positive integer
                  between {n1} and {n2}.'''.format(n1=1, n2=maxVals))
            return False
    else:
        print('''Number of singular values must be a positive integer
              between {n1} and {n2}.'''.format(n1=1, n2=maxVals))
        return False


def svd_needs_recomputing(kernel, k, U, s, Vh):
    if k is None:
        return False
    
    Nr, Nm, Ns = kernel.shape
    M, N = Nr * Nm, Ns * Nm
    if k_is_valid(k, min(M, N)):
        if ((M, k), (k, N)) == (U.shape, Vh.shape) and k == len(s):
            return False
        else:
            print('Inconsistent dimensions: SVD needs recomputing...')
            return True
