import numpy as np
import pandas as pd
import scipy as sp
from scipy import io
import scipy.linalg as la
import os
import sys
import pickle

from sklearn import preprocessing
import h5py
import scipy.sparse as sparse

module_path = ''
if module_path not in sys.path:
    sys.path.append(module_path)
    
new_data_file = ''
results_save_path = ''
if not os.path.exists(results_save_path):
    os.makedirs(results_save_path)

def wthresh(a, thresh):
    #Soft wavelet threshold
    res = np.abs(a) - thresh
    return np.sign(a) * ((res > 0) * res)

# Alex: ignore the below coment but keep 0.03 in mind.
#Default threshold of .03 is assumed to be for input in the range 0-1...
#original matlab had 8 out of 255, which is about .03 scaled to 0-1 range
def go_dec(X, thresh=None, rank=2, power=0, alg='orig', tol=1e-3,
           max_iter=100, random_seed=0, verbose=True):
    """
    alg: sting
        One of 'orig' and 'corrrect'. According to my derivations the correct
        version should be as under 'correct'.
    
    """
    import gc
    gc.collect()

    m, n = X.shape
    assert (m > n), "Long matrix is assumed. Transpose if necessary"
    
    if thresh is None:
        thresh = 4* np.sum( np.abs( X.ravel() )) / ( m*n ) # f-la from Candes RPCA.
    if verbose: print("SoftTheresh: ", thresh)
    
    L = X
    S = np.zeros(L.shape)
    itr = 0
    random_state = np.random.RandomState(random_seed)    
    while itr < max_iter:
        if alg == 'orig':
            Y2 = random_state.randn(n, rank)
        elif alg == 'correct':
            Y2 = np.dot( L.T, random_state.randn(m, rank) )
        else:
            raise ValueError("Incorrect vakue for parameter alg.")
            
        for i in range(power + 1): # at least one iteration.
            Y2 = np.dot(L, Y2)
            Y2 = np.dot(L.T, Y2); # it computes: Y2 = 
        Q, R = la.qr(Y2, mode='economic')
        del Y2, R
        P1 = np.dot(L, Q)
        L_new = np.dot(P1, Q.T)
        T = L - L_new + S
        #L = L_new # Alex
        S = wthresh(T, thresh)
        T -= S
        err = la.norm(T.ravel(), 2)
        if (err < tol):
            break 
        L = L_new + T
        itr += 1
        gc.collect()
        if verbose:
            print("GoDec Iter. number: ", itr, ",  MSE error: ", err, ",  S sparsity per sample %: ", np.nonzero(S)[0].size / (S.shape[0]) )
    if verbose:
        print("Finished at iteration %d" % (itr))
        
    return P1, Q.T, sparse.csr_matrix(S)

def go_dec_experiment(p_go_dec_rank, p_threshold_multiplier, p_max_iter, p_save_file_name):
    
    
    # load the data ->
    import pandas as pd
    from sklearn import preprocessing
    
    # Code removed
    
    print('Experiment done!')
    
if __name__ == '__main__':
    # Command line parameters: rank, iteration number, save_file_name
    
    p_rank = int(sys.argv[1])
    p_threshold_multiplier = float(sys.argv[2])
    p_iter_num = int(sys.argv[3])
    p_file_name = sys.argv[4]
    print(p_rank, p_threshold_multiplier, p_iter_num, p_file_name) 
    go_dec_experiment(p_rank, p_threshold_multiplier, p_iter_num, p_file_name)
    
    
    
    