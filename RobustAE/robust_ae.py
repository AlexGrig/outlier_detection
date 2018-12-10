#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 17:15:58 2018
@author: alexgrig

The file is an adaptation of Robust Autoencoder 
of https://arxiv.org/pdf/1704.06743.pdf and 
https://github.com/zc8340311/RobustAutoencoder

"""

import numpy as np
import tensorflow as tf
import scipy.sparse as sparse
import scipy.linalg as la
import os
import sys

from basic_ae import Deep_Autoencoder
#from BasicAutoencoder import DeepAE as DAE
#from shrink import l21shrink as SHR 



def shrink(epsilon, inp):
    """
    Input:
    --------------
        epsilon: float
            The positive shrinkage parameter (either a scalar or a vector)
        x: float vector
            The vector to shrink on

    Output:
        The shrunk vector
    """
#    output = np.array(x*0.)
#
#    for i in xrange(len(x)):
#        if x[i] > epsilon:
#            output[i] = x[i] - epsilon
#        elif x[i] < -epsilon:
#            output[i] = x[i] + epsilon
#        else:
#            output[i] = 0
            
      
    p1 = np.where( inp > epsilon, inp - epsilon, 0.0 )
    p2 = np.where( inp < -epsilon, inp + epsilon, 0.0 )
        
    output = p1 + p2
    
    return output

def l21shrink(epsilon, inp):
    """
    Args:
        epsilon: the positive .shrinkage parameter
        x: matrix to shrink on
    Ref:
        wiki Regularization: {https://en.wikipedia.org/wiki/Regularization_(mathematics)}
    Returns:
            The shrunk matrix
    """
    
#    output = x.copy()
#    norm = np.linalg.norm(x, ord=2, axis=0)
#    for i in xrange(x.shape[1]):
#        if norm[i] > epsilon:
#            for j in xrange(x.shape[0]):
#                output[j,i] = x[j,i] - epsilon * x[j,i] / norm[i]
#        else:
#            output[:,i] = 0.
#            
    
    norm = np.linalg.norm(inp, ord=2, axis=0) # norm of each column
    norm = np.tile(norm, (inp.shape[0],1) ) # extend along the 0 axis
    
    output = np.where( norm > epsilon, inp - epsilon* inp / norm, 0.0 )
    
    return output




class RobustAE(object):
    """
    
    Doses:
        X = L + S
        L is a non-linearly low dimension matrix and S is a sparse matrix.
        argmin ||L - Decoder(Encoder(L))|| + ||S||_2,1
        Use Alternating projection to train model
        The idea of shrink the l21 norm comes from the wiki 'Regularization' link: {
            https://en.wikipedia.org/wiki/Regularization_(mathematics)
        }
    Improve:
        1. fix the 0-cost bugs

    """
    def __init__(self, p_session, p_layers, p_outliers_type ='independent', p_lambda=1.0, p_seed = None, p_log_folder = '', p_clean_summary_folder=True):
        """
        sess: a Tensorflow tf.Session object
        layers_sizes: a list that contain the deep ae layer sizes, including the input layer
        lambda_: tuning the weight of l1 penalty of S
        
        
        p_outliers_type: string
            One of 'independent', 'rows' or 'columns'.
        
        """
        # TODO: parameters of AE should should all be here.
        self._lambda = p_lambda
        self._layer_sizes = p_layers
        self._session = p_session
        self._outliers_type = p_outliers_type
        self._seed = p_seed
        self.log_folder = p_log_folder
        if self.log_folder[-1] != '/':
            self.log_folder += '/'
        #self.error = error
        #self.errors=[]
        
# def __init__(self, p_input_dim_list, p_activ_type='relu', 
#                 p_bias_max_val=0, p_default_dropout_rate = 0, p_is_constrained=False, 
#                 p_last_layer_activ_type=None, p_session = None, p_data_type = None,
#                 p_seed=None, p_cost_normalize_type='per_sample', p_summary_folder='./Log', 
#                 p_clean_summary_folder=False):
#        
#        
        
        
        self._activ_type = 'relu'
        self._is_constrained = False
        self._last_layer_activ_type = None
        self._ae_data_type = tf.float64
        self._ae_cost_normalize_type = 'per_sample_per_dim'
        
        self.ae = Deep_Autoencoder( p_input_dim_list=p_layers, p_activ_type=self._activ_type, 
                         p_bias_max_val=0, p_default_dropout_rate = 0.0, p_is_constrained=self._is_constrained, 
                         p_last_layer_activ_type=self._last_layer_activ_type, p_session = p_session, p_data_type= self._ae_data_type, p_seed=p_seed, 
                         p_cost_normalize_type=self._ae_cost_normalize_type, p_summary_folder=p_log_folder, p_clean_summary_folder=p_clean_summary_folder )
        
        self.print_model_info()
        
    def train(self, X, p_optim_algorithm='sgd_momentum', p_learning_rate=0.15, p_inner_epochs = 10,
            total_iterations=20, p_batch_size=128, p_dropout_rate = 0, 
            p_ae_epoch_eval_freq = 20,
            verbose=True):
        """
        X: ndarray
            Data matrix
        learning_rate: float
            Autoencoder learning rate
        
        inner_iteration: 
        """
        ## The first layer must be the input layer, so they should have same sizes.
        assert X.shape[1] == self._layer_sizes[0]
        ## initialize L, S
        
        if p_optim_algorithm == 'sgd_momentum':
            optim_params={'learning_rate': p_learning_rate, 'momentum': 0.9}
            
        elif p_optim_algorithm == 'adam':
            optim_params={'learning_rate': p_learning_rate}
        else:
            raise ValueError("Wrong optim algorithm")
        
    
        self.L = np.zeros(X.shape, dtype=np.float64)
        self.S = np.zeros(X.shape, dtype=np.float64)
        
        ##LS0 = self.L + self.S
        ## To estimate the size of input X
            
        glob_inner_step_no = 0
        for it in range(total_iterations):
            mse_err = la.norm(np.ravel( X-self.L-self.S) , 2)
            if verbose:
                print("     Outer iteration: " , it, " MSE:", mse_err, " S sparsity per sample: ",  
                      np.nonzero(self.S)[0].size / (self.S.shape[0] ) )
               
            #import pdb; pdb.set_trace()
            ## alternating project, first project to L
            self.L = X - self.S

#            glob_inner_step_no = self.ae.train(self.L, p_valid_data=self.L, 
#                 p_epoch_num=p_inner_epochs, p_batch_size = p_batch_size, 
#                 p_dropout_rate = p_dropout_rate, p_optim_algorithm='sgd_momentum', 
#                 p_optim_params={'learning_rate': p_learning_rate, 'momentum': 0.9},
#                 p_init_step_no=glob_inner_step_no, p_epoch_eval_freq=5, p_epoch_weights_stat_freq=10,
#                 p_reinit_weights= (True if glob_inner_step_no == 0 else False) )
            
            glob_inner_step_no = self.ae.train(self.L, p_valid_data=self.L, 
                 p_epoch_num=p_inner_epochs, p_batch_size = p_batch_size, 
                 p_dropout_rate = p_dropout_rate, p_optim_algorithm=p_optim_algorithm, 
                 p_optim_params=optim_params,
                 p_init_step_no=glob_inner_step_no, p_epoch_eval_freq=p_ae_epoch_eval_freq, 
                 p_epoch_weights_stat_freq=p_ae_epoch_eval_freq,
                 p_reinit_weights= True,
                 p_verbose_level=1)
            
            
            ## get optmized L
            self.L = self.getRecon(self.L, p_batch_size = 100*p_batch_size)
            
            ## alternating project, now project to S and shrink S
            if self._outliers_type == 'independent':
                self.S = shrink(self._lambda, (X - self.L) )
            elif self._outliers_type == 'columns':
                self.S = l21shrink(self._lambda, (X - self.L))
            elif self._outliers_type == 'rows':
                self.S = l21shrink(self._lambda, (X - self.L).T).T
            else:
                raise ValueError()
        
        
        
        #self.save_model(total_iterations - 1, glob_inner_step_no)
        
        return self.L , self.S
    
    @staticmethod
    def make_save_file_name(p_global_iter_no, p_global_ae_step_no):
            
            file_name_t = str(p_global_iter_no) + '_' + str(p_global_ae_step_no)
            return file_name_t
    
    def print_model_info(self,):
        
        print("Robust Autoencoder:")
        print("  lambda: ", self._lambda)
        print("  layer sizes: ", self._layer_sizes)
        print("  outliers type: ", self._outliers_type)
        print("  log folder: ", self.log_folder)
        print("--------AE---------:")
        print("  activation type: ", self._activ_type)
        print("  is constrained: ", self._is_constrained)
        print("  last layer activ. type: ", self._last_layer_activ_type)
        print("  ae data type: ", self._ae_data_type )
        print("  ae cost normalization type: ",self._ae_cost_normalize_type)

        
    def getRecon(self, x_data, p_batch_size=None ):
        #return self.AE.getRecon(self.L, sess = sess)
        
        with self._session.graph.as_default():
            
            ret = self.ae.model_eval(x_data, p_batch_size=p_batch_size, p_return_transform=True)    
        return ret[1] # return transformed data part
    
    def save_model(self, p_file_name):
        """
        
        """
        
        #import pdb; pdb.set_trace()
        
        file_name_t = p_file_name
        sparse_file_name = 'S_' + file_name_t + '.npz'
        
        sparse.save_npz(os.path.join( self.log_folder, sparse_file_name), sparse.csr_matrix(self.S))
    
        self.ae.save_model( os.path.join(self.log_folder, 'models'), 'ae_' + file_name_t )
    
    def load_model(self, p_log_folder, p_file_name ):
        """
        
        """
        
        sparse_file_name = 'S_' + p_file_name + '.npz'
        
        self.S = sparse.load_npz(os.path.join( p_log_folder, sparse_file_name)).todense()
        
        self.ae.load_model( os.path.join(p_log_folder, 'models') , 'ae_' + p_file_name ) 
        
def test_robustAE():
    """
    
    """
    
    # Generate data ->
    from scipy import stats
    N = 1000; D = 200; seed = 34
    
    
    X = stats.bernoulli.rvs(0.3, size=(N,D))
    # Generate data <-
    with tf.Graph().as_default():
        sess = tf.Session()
        rae = RobustAE( sess, [D, 40, 10], 'independent', p_lambda=1.0, p_seed=seed, p_log_folder = './can_be_removed_Log_test_robustAE')
    
        rae.train(X.astype(np.float32), p_learning_rate=0.1, p_inner_epochs = 10,
            total_iterations=20, p_batch_size=128, p_dropout_rate = 0, 
            verbose=True)
    
def test_compare_with_paper():
    """
    Compare this AE with the AE from the paper: 
        Anomaly Detection with Robusr Deep Autoencoders. Chong Zhou, Randy C. Paffenroth.s
    
    """
    
    # Take the data from the data forlder of RobustAutoencoder repo ->
    
    data_show_path = os.path.abspath(os.path.join('../RobustAutoencoder/data')) # path is relative to notebook path.
    if data_show_path not in sys.path:
        sys.path.append(data_show_path)    
    
    
    
    x_data_file = os.path.abspath(os.path.join('../RobustAutoencoder/data/data.txt'))
    y_data_file = os.path.abspath(os.path.join('../RobustAutoencoder/data/y.txt'))
    
    x_data = np.loadtxt(x_data_file,delimiter=",")
    y_data = np.loadtxt(y_data_file,delimiter=",")
    
        # Show the data:
    import matplotlib.pyplot as plt
    import ImShow as Ii
    plt.figure(1)
    Xpic = Ii.tile_raster_images(X = x_data, img_shape=(28,28), tile_shape=(10,10))
    plt.imshow(Xpic,cmap='gray')
    plt.show()
    # Take the data from the data forlder of RobustAutoencoder repo <-
    
    
    
    
    # set AE parameters ->
    inner_epochs = 50 # in one outer iteration
    outer_iterations = 20
    p_re_init = True # whether to reinit the AE on every outer iteration
    
    p_lambda = 0.00095
    p_layers = [784, 400, 200] 
    p_lr = 0.001
    p_minibatch_size = 133
    p_seed = None
    # set AE parameters <-2
    
    
    # function to evaluate RAE ->
    def eval_RAE(sparse_matrix, low_rank_reconstr, caption_text="L21 "):
        """
        low_rank_reconstr - only for visualization.
        """
        
        # Calculate statistics
        from sklearn.metrics import f1_score
        from sklearn.metrics import recall_score as recall
        from sklearn.metrics import precision_score as precision
        from collections import Counter
        
        l21S = sparse_matrix
        l21R = low_rank_reconstr
        
        def binary_error(value):
            if value != 0.0:
                return "o" # 'majority'
            else:
                return "m" #'outlier'
            
        def binary_y(value):
            if value == 4:
                return "m"
            else:
                return "o"
        
        bi_y = list( map(binary_y,y_data) )
        print(Counter(bi_y))
        
        #import pdb; pdb.set_trace()
        
        S =  l21S  
        predictions = list( map(binary_error,np.linalg.norm(S,axis = 1)) )
        p = precision(bi_y,predictions,labels=["o","m"],pos_label="o")
        r = recall(bi_y,predictions,labels=["o","m"],pos_label="o")
        f1 = f1_score(bi_y,predictions,labels=["o","m"],pos_label="o")
        print("lambda:", p_lambda)
        print("stat:", Counter(predictions))
        print("precision",p)
        print("recall",r)
        print("f1",f1)
        
        
            # Showing the images
        inputsize = (28,28)
        iS = Image.fromarray(Ii.tile_raster_images(X=l21S,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1)))
        iR = Image.fromarray(Ii.tile_raster_images(X=l21R,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1)))
        iD = Image.fromarray(Ii.tile_raster_images(X=x_data,img_shape=inputsize, tile_shape=(10, 10),tile_spacing=(1, 1)))

        fig,ax = plt.subplots(nrows=1,ncols=3, squeeze=False)
        fig.set_size_inches(12, 3)
        #import pdb; pdb.set_trace()
        
        ax[0][0].imshow(iR,cmap = "gray")
        ax[0][1].imshow(iS,cmap = "gray")
        ax[0][2].imshow(iD,cmap = "gray")
        
        
        ax[0][0].set_title(caption_text + "R")
        ax[0][1].set_title(caption_text + "S")
        ax[0][2].set_title("X")
        
        ax[0][0].get_xaxis().set_visible(False); ax[0][0].get_yaxis().set_visible(False)
        ax[0][1].get_xaxis().set_visible(False); ax[0][1].get_yaxis().set_visible(False)
        ax[0][2].get_xaxis().set_visible(False); ax[0][2].get_yaxis().set_visible(False)
            
        plt.show()
    # function to evaluate RAE <-
    
    
    
    # Train other RAE ->
    re_path = os.path.abspath(os.path.join('../RobustAutoencoder/')) # path is relative to notebook path.
    if re_path not in sys.path:
        sys.path.append(re_path)
        
    import PIL.Image as Image
    from model import l21RobustDeepAutoencoderOnST as l21RDA

    with tf.Graph().as_default():
        with tf.Session() as sess:
            rael21 = l21RDA.RobustL21Autoencoder(sess = sess, lambda_= p_lambda*x_data.shape[0], layers_sizes=p_layers)
            _, l21S = rael21.fit(X = x_data, sess = sess, inner_iteration = inner_epochs, iteration = outer_iterations, 
                                    batch_size = p_minibatch_size, learning_rate = p_lr,  re_init=p_re_init,verbose = True)
            # Attention! there is a bug in the basic autoencoder in the paper, so the learning_rate is not affecting at all.
            l21R = rael21.getRecon(X = x_data, sess = sess)
            
            eval_RAE(l21S, l21R)
    # Train other RAE <-
    
    
    
    
    # my RAE ->
        with tf.Graph().as_default():
            sess = tf.Session()
            my_rae = RobustAE( sess, p_layers, 'rows', p_lambda=p_lambda*x_data.shape[0], p_seed=p_seed, 
                       p_log_folder = './Log')
    
            _, my_S = my_rae.train(x_data, p_learning_rate=p_lr, p_inner_epochs = inner_epochs,
            total_iterations=outer_iterations, p_batch_size=p_minibatch_size, p_dropout_rate = 0, 
            p_ae_epoch_eval_freq=20, verbose=True)
    
            my_R = my_rae.getRecon(x_data, p_batch_size=None)
            
            eval_RAE(my_S, my_R)
    # my RAE <-

def test_shrinkage():
    
     # Take the data from the data forlder of RobustAutoencoder repo ->
    
    x_data = np.random.randn(1000,500) + 1
    # Take the data from the data forlder of RobustAutoencoder repo <-
    
    
    # import other shrinkage ->
    re_path = os.path.abspath(os.path.join('../RobustAutoencoder/model/')) # path is relative to notebook path.
    if re_path not in sys.path:
        sys.path.append(re_path)
    
    from shrink import l21shrink as other21
    from shrink import l1shrink as other1
    # import other shrinkage <-
    
    
    # compare shrinking ->
    eps = np.linspace(0,2,20)
    
    test21_fail = False
    for ee in eps:
        other_m = other21.l21shrink(ee, x_data)
        my_m = l21shrink(ee, x_data)
    
        print( "Test shrink21 fails:   ", np.any(other_m != my_m ) )
        
        test21_fail = np.any(other_m != my_m )
        if test21_fail:
            break
        
        
    test1_fail = False
    for ee in eps:
        other_m = other1.shrink(ee, x_data.ravel() )
        my_m = shrink(ee, x_data.ravel() )
    
        print( "Test shrink1 fails:   ", np.any(other_m != my_m ) )
        
        test1_fail = np.any(other_m != my_m )
        if test1_fail:
            break
    # compare shrinking <-
    
if __name__ == "__main__":
    
    # test_shrinkage()
    test_robustAE()
    #test_compare_with_paper()
    #test_pwc_data()
    