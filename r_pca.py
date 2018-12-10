from __future__ import division, print_function

import numpy as np
import scipy as sp
import scipy.linalg as la

try:
    from pylab import plt
except ImportError:
    print('Unable to import pylab. R_pca.plot_fit() will not work.')

try:
    # Python 2: 'xrange' is the iterative version
    range = xrange
except NameError:
    # Python 3: 'range' is iterative - no need for 'xrange'
    pass


class R_pca(object):

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * self.norm_p(np.abs(self.D), 1))

        self.mu_inv = 1 / self.mu
        print('Mu:  ', self.mu, ', inv_mu:  ', self.mu_inv)
        
        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))
        print("Lambda:  ", self.lmbda)
        
    @staticmethod
    def norm_p(M, p):
        return np.sum(np.power(M, p))

    @staticmethod
    def shrink(M, tau):
        res = np.abs(M) - tau   
        return np.sign(M) * ((res > 0) * res)
    
    @staticmethod
    def svd_threshold(M, tau):
        U, S, VT = la.svd(M, full_matrices=False)
        new_diag = np.diag(R_pca.shrink(S, tau))
        #return np.dot(U, np.dot(np.diag(R_pca.shrink(S, tau)), V))
        return U, new_diag, VT
        
    def fit(self, tol=None, max_iter=100, iter_print=1):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.norm_p(np.abs(self.D), 2)
    
        print('Tolerance: ', _tol)
        
        while (err > _tol) and iter < max_iter:
            Uk, Lk_diag, VTk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            
            Lk = np.dot(Uk, np.dot( Lk_diag, VTk))
            
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = np.sqrt( self.norm_p(np.abs(self.D - Lk - Sk), 2) )
            iter += 1
            if (iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol:
                density = np.nonzero(Sk)[0].size / (Sk.shape[0] * Sk.shape[1])
                print('iteration: {0}, error: {1}, sparcity level: {2}'.format(iter, err, density))

        self.Uk = Uk
        self.Lk_diag = np.diag(Lk_diag)
        self.VTk = VTk
        self.S = Sk
        
        return Uk, np.diag(Lk_diag), VTk, Sk

    def plot_fit(self, size=None, tol=0.1, axis_on=True):

        n, d = self.D.shape

        if size:
            nrows, ncols = size
        else:
            sq = np.ceil(np.sqrt(n))
            nrows = int(sq)
            ncols = int(sq)

        ymin = np.nanmin(self.D)
        ymax = np.nanmax(self.D)
        print('ymin: {0}, ymax: {1}'.format(ymin, ymax))

        numplots = np.min([n, nrows * ncols])
        plt.figure()

        for n in range(numplots):
            plt.subplot(nrows, ncols, n + 1)
            plt.ylim((ymin - tol, ymax + tol))
            plt.plot(self.L[n, :] + self.S[n, :], 'r')
            plt.plot(self.L[n, :], 'b')
            if not axis_on:
                plt.axis('off')
