#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:09:22 2018
@author: alexgrig

Basic authoencoder. It is quite complete implementation which is easy to use.
"""



import os
import sys
import tensorflow as tf
import numpy as np
import logger
import sys

def choose_activation(p_act_type, inp):
    if (p_act_type is None) or (p_act_type == 'none') or (p_act_type == 'linear'):
        act = inp
    elif p_act_type == 'relu':
        act = tf.nn.relu(inp)
    elif p_act_type == 'tanh':
        act = tf.tanh(inp)
    elif p_act_type == 'sigmoid':
        act = tf.sigmoid(inp)
    elif p_act_type == 'selu':
        act = tf.nn.selu(inp)
    
    return act

#Good model class properties:    
#    1) Seed. If provided the results are exactly repeatable between runs.
#    2) Logging.
#    3) Fit and evaluate methods
#    4) Serialization
#    5) Good to have data type control e.g. float32 vs float64.
#    6) Print model. Prints all relevant internal variables. Params of last optim run. (missing)
#    7) Make a test for model evaluation, that the evaluation error is the same if we change the evaluation batch size.
    
class Deep_Autoencoder(object):
    
    def __init__(self, p_input_dim_list, p_activ_type='relu', 
                 p_bias_max_val=0, p_default_dropout_rate = 0, p_is_constrained=False, 
                 p_last_layer_activ_type=None, p_session = None, p_data_type = None,
                 p_seed=None, p_cost_normalize_type='per_sample', p_summary_folder='./Log', 
                 p_clean_summary_folder=False):
        """
        Inputs:
        ------------------
        
        sess: TF session
        
        input_dim_list: list of integers.
            First one is input dim, last one is hidden layer dim
        
        dropout_rate: float
            Droupout rate of the middle layer.
            
        p_cost_normalize_type: string
            How to normalize the cost function.
                'per_sample' - cost is divided by the number of samples in a batch
                'per_sample_per_dim' - cost is divided by the number of samples and dimensions
        Output:
        ------------------
        
        """
        
        assert len(p_input_dim_list) >= 2, "Too short p_input_dim_list "
        
        self.dim_list = p_input_dim_list
        
        self.is_constrained = p_is_constrained
        self.activ_type = p_activ_type
        self.last_layer_activ_type = p_last_layer_activ_type if p_last_layer_activ_type is not None else self.activ_type
        self.bias_max_val = p_bias_max_val
        self.default_dropout_rate = p_default_dropout_rate
        self.seed = p_seed
        self.summary_folder = p_summary_folder
        
        self.input_dim = self.dim_list[0]
        if p_data_type is None:
            self._data_type = tf.float32
        else:
            self._data_type = p_data_type
            
        self._input_x = None # This is an iterator next step. Used as a beginning of computational graph.
        self._input_iterator = None # generic iterator which has been used to make the graph. Created once, then can be set different data:
        self._input_data_placeholder = tf.placeholder(dtype=self._data_type, shape=tf.TensorShape( dims= [None, self.input_dim]) )
        
        self._encoder_part_built = False; self._encode_part = None # Graph for encoder part
        self._decoder_part_built = False; self._decode_part = None # Graph for decoder part
        self._cost_part_built = False; self._cost_part = None # Graph for cost part
        self._input_feed_dict = {} # Contains scalars of comp. graph. Updated either in encode or in model_eval methods.
        self._summaries_built = False; self._summaries_part = None # Graph for cost part
        
        self._session = tf.Session() if (p_session is None) else p_session
        
        
        self.logger = logger.Logger(self.summary_folder) # , self._session.graph)
        self.logger.clean_log_folder(p_clean_summary_folder)
        
        # Init ENCODER weights and biases:
        weight_list, bias_list = self.init_enc_or_dec_weights(p_input_dim_list, p_activ_type,
                              p_bias_max_val=self.bias_max_val, p_is_constrained=False, p_variables_scope='encoder', p_data_type=self._data_type, p_seed=self.seed)
        
        self.enc_W_list = weight_list
        self.enc_b_list = bias_list
        
        # Init DECODER weights and biases:
        weight_list, bias_list = self.init_enc_or_dec_weights(list(reversed(p_input_dim_list)), p_activ_type,
                              p_bias_max_val=self.bias_max_val, p_is_constrained=p_is_constrained, p_variables_scope='decoder', p_data_type=self._data_type, 
                              p_seed=self.seed)
        
        self.dec_b_list = bias_list
        if self.is_constrained:
            self.dec_W_list = None
        else:
            self.dec_W_list = weight_list
        
        save_list = self.enc_W_list + self.enc_b_list + self.dec_b_list
        if not self.is_constrained:
            save_list += self.dec_W_list
        
        #import p
        self._saver = tf.train.Saver( save_list )
        
        self._build_graph(p_cost_normalize_type=p_cost_normalize_type) # build the computational graph
        
        
    def model_eval(self, p_numpy_eval_data, p_eval_data=None ,p_batch_size=None, p_return_transform=False):
        """
        Performs mode evaluation.
        
        Input:
        ------------
             p_numpy_eval_data: numpy array
                Always need this to feed into existing data iterator, or build new iterator.
                
            p_eval_data: None or tf.Dataset.
                If it is tf.Dataset, then the function is called internally.
                The dataset defines repetitions of data and other stuff.
                The actual data which is fed is in p_numpy_eval_data then.
        
                if it None then the new dataset is build.
                 
            p_batch_size: int or None
                if None the batching is not done at all. This can consume much memory.
            
            p_eval_data: next element of an iterator
            
            
            p_return_transform: bool
                Whether to return transformed data
        """
        
        # process inputs ->
        eval_feed_dict = {}
        inp_shape = self.dim_list[0]
        
        if p_eval_data is None: # ndarray
            assert p_numpy_eval_data.shape[1] == inp_shape, "Shape of input and input layer must coincide! model_eval_1 "
            eval_data_len = p_numpy_eval_data.shape[0] if (p_batch_size is None) else p_batch_size
            
            #import pdb; pdb.set_trace()
            eval_data = tf.data.Dataset.from_tensor_slices( self._input_data_placeholder ).batch(eval_data_len) # .shuffle(buffer_size=(data_len + 1), seed=self.seed,reshuffle_each_iteration=True)
           
        else: # already datasets
            assert isinstance(p_numpy_eval_data, np.ndarray ), "p_numpy_eval_data must be an array"
            #TODO: assert p_eval_data is Dataset
            eval_data = p_eval_data
        
        #if p_batch_size is not None:
       #     eval_data = eval_data.batch(p_batch_size)
            
        self._activate_data_iterator(eval_data, p_numpy_eval_data)
        # process inputs <-
    
        # Switch out the training mode ->
        eval_feed_dict = self._eval_feeddict()
        # Switch out the training mode <-
    
        #import pdb; pdb.set_trace()
        total_sample_count = 0
        total_cost = 0
        decode_data = None
        while True:
            try:
                _, decode_output, batch_mean_cost = self._session.run([self._input_x, self._decode_part, self._cost_part,], eval_feed_dict )
                # Idea: compute mean cost per data sample. Hence multiply first, later divide by total
                total_sample_count += decode_output.shape[0]
                total_cost += batch_mean_cost*decode_output.shape[0]
                
                if p_return_transform:
                    if decode_data is None:
                        decode_data = decode_output
                    else:
                        decode_data = np.vstack( (decode_data, decode_output))
                
            except tf.errors.OutOfRangeError:
                #print("End evaluation.")
                break
        
        total_cost = total_cost / total_sample_count
        
        return total_cost, decode_data, total_sample_count
            
            
            
            
    def train(self, p_train_data, p_valid_data=None, p_epoch_num=10, p_batch_size = 1, p_valid_batch_size=None,
                 p_dropout_rate = None, p_optim_algorithm='sgd_momentum', p_optim_params={},
                 p_init_step_no=0, p_epoch_eval_freq=3, p_epoch_weights_stat_freq=4,
                 p_reinit_weights=True, p_verbose_level=3):
        """
        Inputs:
        --------------------
        
        p_train_data: np array
            Samples are rows
        
        p_init_step_no: int
            Initial global step no. This is needed if the fitting is called more then one time and
            logging is continued from the previous step.
            
        p_reinit_weights: bool
            If true - the default, then all the variables are reinitialized. If we want to run
            fit second time e.g. with different optimizer, the we need to set it to false.
            Then, weights are not initialized, only optimizer is.
            
        p_valid_batch_size: int
            Batch size for validation data. If none then all dat is used.
        """

        # initialize datasets ->
        data_len = p_train_data.shape[0]
        batches_num = int(np.ceil(data_len / p_batch_size))
        
        train_data = tf.data.Dataset.from_tensor_slices( self._input_data_placeholder ).shuffle(buffer_size=(data_len + 1),
                        seed=self.seed,reshuffle_each_iteration=True).batch(p_batch_size).repeat()
        
        #import pdb;pdb.set_trace()
        
        self._activate_data_iterator(train_data, p_train_data )
    
        if p_valid_data is None: # must be numpy array
             p_valid_data = p_train_data
             
        val_data_len = p_valid_data.shape[0] if (p_valid_batch_size is None) else p_valid_batch_size
        eval_data = tf.data.Dataset.from_tensor_slices( self._input_data_placeholder ).batch(val_data_len)
            
        # initialize datasets <-
        #import pdb; pdb.set_trace()
        
        # separate functions for evalation ->
        def do_evaluation(epoch_no, epoch_eval_freq):
            """
            """
            if p_valid_data is not None:
                
                if epoch_no % epoch_eval_freq == 0:
                    eval_cost, _, samples_count = self.model_eval(p_valid_data, p_eval_data=eval_data, p_return_transform=False)
                    self._activate_data_iterator(train_data, p_train_data ) # return iterator to the training data
                    
                    if p_verbose_level > 0:
                        print("Epoch {};   evaluation loss: {:.4f} on {} samples".format(epoch_no, eval_cost, samples_count))
                    #tf.summary.scalar("evaluation_cost", eval_cost)
                    self.logger.scalar_summary("evaluation_cost", eval_cost, step_no)
                    self.logger.writer.flush()
        # separate functions for evalation <-
        
        # optimizer ->
        
        if p_optim_algorithm == 'sgd_momentum':
            #optim_step = tf.train.MomentumOptimizer(**p_optim_params).minimize(train_mb_cost)
            optimizer = tf.train.MomentumOptimizer(**p_optim_params)
        
        if p_optim_algorithm == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(**p_optim_params)
            
        if p_optim_algorithm == 'adam':
            optimizer = tf.train.AdamOptimizer(**p_optim_params)
        
        grads_and_vars = optimizer.compute_gradients(self._cost_part)
        optim_step = optimizer.apply_gradients(grads_and_vars)
        
        if p_verbose_level > 0:
            print('Optimizer: {}'.format(p_optim_algorithm), "   parameters: ", p_optim_params)
        # optimizer <-
        
        
        # summaries ->
        merged_summaries = self.log_var_and_grad_summaries(grads_and_vars, log_histograms=True)
        
        def compute_summaries(epoch_no, epoch_stats_freq, input_feed_dict):
            """
            The separate function is done to be able to do this more flexible e.g. in the very beginning of iterations
            """
            
            if epoch_no % epoch_stats_freq == 0:
                summary = self._session.run(merged_summaries, input_feed_dict)
                #import pdb; pdb.set_trace()
                self.logger.writer.add_summary(summary, step_no)
                self.logger.writer.flush()
            
        # summaries <-
        
        # init ->
        
        if p_reinit_weights:
            self._session.run(tf.global_variables_initializer())
        else:
            
            self._session.run(tf.variables_initializer(optimizer.variables()))
        # init <-
        
        # Running dictionary ->
        #train_input_feed_dict = { kk: vv for (kk, vv) in self._input_feed_dict.items() if (not isinstance(vv,str) ) }
        
        train_feed_dict = self._training_feeddict(p_dropout_rate)
        #print(self._input_feed_dict)
        print(train_feed_dict)
        
        # Running dictionary <-
        
        #res = self._session.run([train_mb_cost,], train_input_feed_dict )
        
        #print(res)
        #return 
        
        step_no = p_init_step_no
        epoch_loss = 0
        denom_epoch_loss = 0
        #import pdb; pdb.set_trace()
        for ep in range(p_epoch_num):
            epoch_loss = 0
            denom_epoch_loss = 0
            
            compute_summaries(ep, p_epoch_weights_stat_freq, train_feed_dict)
            if ep == 0: # TODO:
                do_evaluation(ep, p_epoch_eval_freq)
            
            for i in range(batches_num):
                
                train_batch_data, _cost, _ = self._session.run([self._input_x, self._cost_part, optim_step], train_feed_dict )
                
                #gv = self._session.run( grads_and_vars, train_input_feed_dict )
                #import pdb; pdb.set_trace()
                
                epoch_loss += _cost*train_batch_data.shape[0]
                denom_epoch_loss += train_batch_data.shape[0]
                
                step_no += 1
                #print(train_batch_data)
                if p_verbose_level > 2:
                    print("        Epoch: {},  Iter: {}, Loss: {:.4f}".format(ep, i, _cost))
                self.logger.scalar_summary("running_cost", _cost, step_no)
                #tf.summary.scalar("running_cost", _cost)
                
            epoch_loss /=denom_epoch_loss
            if p_verbose_level > 1:
                print("Epoch {} end. Mean epoch loss:  {:.4f}".format(ep, epoch_loss))
            self.logger.scalar_summary("epoch_mean_cost", epoch_loss, step_no)
                      
            do_evaluation(ep, p_epoch_eval_freq)
            compute_summaries(ep, p_epoch_weights_stat_freq, train_feed_dict)
                
        return step_no
    
    
    def _activate_data_iterator(self, p_inp_data=None, p_numpy_data=None):
        """
        This function associates the internal data iterator with the given data.
        If p_iterator is not provided, it is created. If it is provided it is activated
        by 
        
        
        Input:
        -------------
        
        p_inp_data: Tensorflow data - result of tf.data.Dataset.from_tensor_slices. Various methods e.g batch are applied to it.
        
        """
        
        #_data = tf.data.Dataset.from_tensor_slices(p_inp) # .shuffle(buffer_size=(data_len + 1), seed=self.seed,reshuffle_each_iteration=True)
        
        #import pdb; pdb.set_trace()
        
        if self._input_x is None: # here just build iterator. Is run only once.
#            if p_inp_data is not None: # probably this branch is not needed anymore.
#                _iter = tf.data.Iterator.from_structure(p_inp_data.output_types, p_inp_data.output_shapes)
#                _iter_init = _iter.make_initializer(p_inp_data)
#                self._session.run(_iter_init)
#            else:
            _iter = tf.data.Iterator.from_structure(self._data_type, tf.TensorShape( dims= [None, self.input_dim]))
                
            _inp = _iter.get_next()
            
            self._input_x = _inp # iterator next step
            self._input_iterator = _iter
        else:
            assert p_numpy_data is not None and isinstance(p_numpy_data, np.ndarray), "Input data must be Numpy array"
            
            _iter_init = self._input_iterator.make_initializer(p_inp_data) # init the iteratro with the new data
            self._session.run(_iter_init, feed_dict={self._input_data_placeholder: p_numpy_data })
        
        #self._input_x = _inp # iterator next step
        #self._input_iter_init = _iter_init
        
    def _build_graph(self, p_cost_normalize_type='per_sample'):
        """
        The function builds the computations graph for autoencoder, and puts
        the feed_dict variables (placeholders) into self._input_feed_dict:
            
        """
        
        # build computational graph ->
        self._activate_data_iterator(p_inp_data=None) # Activate without any data. It only creates the data iterator.
        
        # Either builds the graph, or returns the cost part of the existing graph:
        if not self._cost_part_built: # Graph for cost part
            self._cost_part = self.cost( self.decode(self.encode( self._input_x)), self._input_x, p_cost_normalize_type
                                        =p_cost_normalize_type )
        else:
            pass
        
        # build computational graph <-
    
    
    def _find_feed_dict_keys(self, which=None):
        """
        
        """
        
        if which == 'is_training':
            is_training_key = [ kk for kk in self._input_feed_dict.keys() if kk.startswith('is_training')]
            assert len(is_training_key) == 1, "Can not find is_training key."
            key = is_training_key[0]
            #feed_dict[is_training_key] = False
        elif which == 'dropout_rate':
            # Nullify the dropout rate
            drop_rate_key = [ kk for kk in self._input_feed_dict.keys() if kk.startswith('dropout_rate')]
            assert len(drop_rate_key) == 1, "Can not find dropout_rate key."
            key = drop_rate_key[0]
        else:
            raise ValueError("Wrong key name")
        
        return key
        
    def _training_feeddict(self, p_drop_out=None): 
        """
        
        """
        feed_dict = {}
        
        feed_dict[ self._find_feed_dict_keys('is_training') ] = True
        feed_dict[ self._find_feed_dict_keys('dropout_rate') ] = p_drop_out if p_drop_out is not None else self.default_dropout_rate
    
        return feed_dict
    
        
    def _eval_feeddict(self,):
        """
        
        """
        
        feed_dict = {}
        
        feed_dict[ self._find_feed_dict_keys('is_training') ] = False
        feed_dict[ self._find_feed_dict_keys('dropout_rate') ] = 0
    
        return feed_dict
        
        
    def encode(self, p_inp, p_is_training=True, p_dropout_rate=None, p_batch_size=None, p_compute=False):
        """
        Transforms the input with the encoding part of autoencoder.
        if p_inp is not nd.array then
        before the call to this function the call to _activate_data_iterator must be done.
        That call fills the compulsory self._input_x and self._input_iterator
        
        
        
        Input:
        -------------
        p_inp: either nd.array or iterator next_step
            Samples are rows!
        
        p_compute: bool
            Do actual computations e.g. run the graph, otherwise just modify the graph
        
        p_batch_size: int
            Used only when p_inp is array     
        
        p_dropout_rate: float
            If undefined then the default dropout rate will be used.
        """
        
        # process inputs ->
        if isinstance(p_inp, np.ndarray): # ndarray
            assert p_inp.shape[1] == self.input_dim, "Shape of input and input layer must coincide! 1 "
            
            batch_size = p_batch_size if (p_batch_size is not None) else p_inp.shape[0]
            
            _data = tf.data.Dataset.from_tensor_slices( self._input_data_placeholder ).batch(batch_size)
            # Do the same as in train method
            #_data = tf.data.Dataset.from_tensor_slices(p_inp) # .shuffle(buffer_size=(data_len + 1), seed=self.seed,reshuffle_each_iteration=True)
            
            self._activate_data_iterator(_data, p_inp )
        else: # placeholder
            #import pdb; pdb.set_trace()
            
            assert p_inp.get_shape()[1] == self.input_dim, "Shape of input and input layer must coincide! 2"
            #assert p_iterator is not None, "Input iterator can not be None. Generic iterator is expected."
            inp = self._input_x
        # process inputs <-
            
        # build or retrive encoder subgraph ->
        inp = tf.transpose(inp) # Now samples are rows. It is just easier to write formulas in this way.
        # Can be changed in the future is formulas are substituted by layers
        
        feed_dict = {}
        if not self._encoder_part_built:
            
            is_training = tf.placeholder(tf.bool, name='is_training')
            drop_rate = tf.placeholder(self._data_type, name='dropout_rate')
            
            # Internal dictionary which has placeholders names. Values are not important
            self._input_feed_dict[is_training.name] = True
            self._input_feed_dict[drop_rate.name] = self.default_dropout_rate
            feed_dict = self._input_feed_dict
            
            for i in range(len(self.dim_list)-1): # -1 because not iterate on last value
                    out = choose_activation( self.activ_type, tf.add( tf.matmul( self.enc_W_list[i], inp ), self.enc_b_list[i] ) )
                    inp = out
            
            drop_fn = lambda: tf.nn.dropout(out, keep_prob=(1-drop_rate))
            no_drop_fn = lambda: tf.identity( out )
            
            out = tf.cond( is_training, true_fn=drop_fn, false_fn=no_drop_fn)
 #               tf.layers.dropout( )
            self._encoder_part_built = True
            self._encode_part = out
        else:
            # Switch out the training mode ->
            if p_is_training:
                feed_dict = self._training_feeddict(p_dropout_rate)
            else:
                feed_dict = self._eval_feeddict()
            # Switch out the training mode <-
        
            out = self._encode_part
        # build or retrive encoder subgraph <-
            
        #self._input_feed_dict.update( feed_dict ) # update 
        
        if p_compute:
            ret = self._session.run([self._input_x, self._encode_part,], feed_dict )
        else:
            ret = self._encode_part
        return ret 
        
    def decode(self, p_inp_part, p_compute=False):
        """
        
        Input:
        -------------
        p_inp_part: tf.placeholder (middle_dim, None)
            Samples are columns!
            
        p_compute: bool
            Whether to compute the output by running the session.
        """
            
        hidden_shape = self.dim_list[-1]
        assert p_inp_part.get_shape()[0] == hidden_shape, "Shape of hidden input and middle layer shape must coincide!"
        inp = p_inp_part
        
        # build or retrive encoder subgraph ->
        if not self._decoder_part_built:
            
            #import pdb; pdb.set_trace()
            iter_len = len(list(reversed(self.enc_W_list)))
            if self.is_constrained:
                weights_list = list(reversed(self.enc_W_list))
            else:
                weights_list = self.dec_W_list
                
            for i in range( iter_len ):
                if self.is_constrained:
                    if i == iter_len: # last layer
                        out = choose_activation( self.last_layer_activ_type, tf.add( tf.matmul( weights_list[i], inp , transpose_a=True), self.dec_b_list[i] ) )
                    else:
                        out = choose_activation( self.activ_type, tf.add( tf.matmul( weights_list[i], inp, transpose_a=True ), self.dec_b_list[i] ) )
                else:
                    if i == iter_len-1: # last layer
                        out = choose_activation( self.last_layer_activ_type, tf.add( tf.matmul( weights_list[i], inp ), self.dec_b_list[i] ) )
                    else:
                        out = choose_activation( self.activ_type, tf.add( tf.matmul( weights_list[i], inp ), self.dec_b_list[i] ) )
                inp = out
                    
            self._decode_part = tf.transpose(out)
            self._decoder_part_built = True
        else:
            out = self._decode_part
         # build or retrive encoder subgraph <-
         
        if p_compute:
            ret = self.session.run([self._input_x, self._decode_part,], self._input_feed_dict )
        else:
            ret = self._decode_part
            
        return ret
    
    def cost(self, p_recon, p_true_out, p_cost_type = 'mse', p_cost_normalize_type='per_sample',p_compute=False):
        """
        Input:
        -------------
        p_recon: tf.placeholder (None, data_dim)
            Samples are rows!
        
        p_cost_normalize_type: string
            How to normalize the cost function.
                per_sample - cost is divided by the number of samples in a batch
                per_sample_per_dim - cost is divided by the number of samples and dimensions
            
        Output:
        -------------
        Either real cost if p_compute is true, or graph for computing the cost.
        
        The cost is mean over the samples, but NOT averaged across dimensions.
        
        """
        
        cost = None
        
        #import pdb; pdb.set_trace()
        # build cost part ->
        if not self._cost_part_built:
            if p_cost_type == 'mse':
                
                if p_cost_normalize_type=='per_sample': # cost is an average per sample
                    cost = tf.reduce_mean(tf.square(p_recon - p_true_out)) * tf.cast(tf.shape(p_recon)[1], self._data_type ) #self.dim_list[0] # multiply by input dimensionality to have loss per sample
                elif p_cost_normalize_type=='per_sample_per_dim':
                    cost = tf.reduce_mean(tf.square(p_recon - p_true_out))
                    
            self._cost_part = cost
            self._cost_part_built = True
        else:
            cost = self._cost_part
        # build cost part <-
        
        if p_compute:
            ret = self.session.run([cost,], self._input_feed_dict )
        else:
            ret = cost
                
        return ret
    
    def log_var_and_grad_summaries(self, grad_vars, log_histograms=False):
        
        """
        Makes tensorflow summaries. This can be run only after optimizer and hence
        grad_vars are available.
        
        
        Inputs:
        -------------
        
        grad_vars: tuple
            The result of the compute_gradient call of the optimizer.
        
        """
        
        
        
        #import pdb; pdb.set_trace()
        def log_vars(gv, histograms=False):
            
            var_name = gv[1].name[ 0:gv[1].name.find(':') ] # second element - variable, first - gradient
            tf.summary.scalar( var_name + '/fro' , tf.norm( gv[1], ord='euclidean') ) # this is actually frobenious for matrices
            
            if histograms:
                #import pdb; pdb.set_trace()
                tf.summary.histogram(var_name + '/h', gv[1])
        
            tf.summary.scalar( var_name + '/grad_fro' , tf.norm( gv[0], ord='euclidean') ) # this is actually frobenious for matrices
        
            if histograms:
                tf.summary.histogram(var_name + '/grad', gv[0])
                
        if not self._summaries_built:
            
            for gv_tmp in grad_vars:
                log_vars(gv_tmp, log_histograms)
            
            merged_summaries = tf.summary.merge_all()
            
            self._summaries_built = True
            self._summaries_part = merged_summaries
            
        else:
            merged_summaries = self._summaries_part
          
        return merged_summaries
    
    
    def save_model(self, p_path, p_label):
        """
        Save the weights variables
        """
        
        self._saver.save(self._session, os.path.join(p_path, str(p_label)) )
        
        
    
    def load_model(self, p_path, p_label):
        """
        Loads the model.
        """
        self._saver.restore(self._session, os.path.join(p_path, str(p_label)) )
        
        
    def log_graph(self, p_step):
        """
        Log computational graph
        """
        
        self.logger.graph_summary(self._session.graph, p_step)
        
    @staticmethod    
    def init_enc_or_dec_weights(p_layer_dims, p_activ_type, p_bias_max_val=0, p_variables_scope='', p_is_constrained = False, 
                                p_data_type=tf.float32, p_tf_graph=None, p_seed=None):
        """
        
        Inputs:
        ------------------
            p_variables_scope: string
                Top scope in the variables names e.g. encoder
            
            
        """
        
        tf_graph = p_tf_graph if (p_tf_graph is not None) else tf.get_default_graph()
        with tf_graph.as_default():
            weights_list = []
            biases_list = []
            
            #with tf.variable_scope(p_variables_scope):
            for i in range(len(p_layer_dims)-1): # -1 because not iterate on last value
                input_dim = p_layer_dims[i]
                output_dim = p_layer_dims[i+1]
                
                if not p_is_constrained:
                    #with tf.variable_scope("weights"):
                    W_in = Deep_Autoencoder.init_weight( (output_dim, input_dim),  p_activ_type=p_activ_type, p_tf_graph = p_tf_graph, 
                                                        p_data_type=p_data_type, p_seed=p_seed)
                    weights_list.append( tf.Variable(W_in, name='weights/' + p_variables_scope + '/' + str(i)) ) # Names are relevant for summaries
                        
                #with tf.variable_scope("biases"):
                b_in = Deep_Autoencoder.init_bias((output_dim,1 ), p_bias_max_val, p_tf_graph = p_tf_graph, p_data_type=p_data_type, p_seed=p_seed)
                biases_list.append( tf.Variable(b_in, name='biases/' + p_variables_scope + '/' + str(i) ) )
        
        return weights_list, biases_list
        
    
    @staticmethod
    def init_weight(p_shape, p_init_type='xavier', p_activ_type='', p_tf_graph =None, p_data_type=tf.float32, p_seed=None):
        """
        Initialize one weight matrix
        
        Input:
        ---------------
        p_shape: list
            [0] - output, [1] - input
        
        Output:
        ---------------
            
        """
        tf_graph = p_tf_graph if (p_tf_graph is not None) else tf.get_default_graph()
        with tf_graph.as_default():
            
            if p_init_type == 'xavier':
                max_value = np.sqrt( 6.0 / (p_shape[0] + p_shape[1]))
                gain = 1 # see pytorch Glorot init
                
                if p_activ_type == 'relu': gain = np.sqrt(2)
                if p_activ_type == 'tanh': gain = 5/3
                if p_activ_type == 'lrelu': raise NotImplemented() # see pytorch doc
                
                max_value *= gain
                
                ret = tf.random_uniform(p_shape, minval=-max_value, maxval=max_value, dtype=p_data_type, seed= p_seed)
            
            if p_init_type == 'snn_paper':
                
                std = np.sqrt( 1 / p_shape[1] ) # 
                ret = tf.random_normal(p_shape, std= std, dtype=p_data_type, seed=p_seed)
                
        return ret

    @staticmethod    
    def init_bias(p_shape, p_magn, p_random_type='uniform', p_tf_graph =None, p_data_type=tf.float32, p_seed=None):
        """
        Input:
        ----------------
        
        p_magn: int
            Either border value of uniform distribution or, std of Normal.
        """
        
        tf_graph = p_tf_graph if (p_tf_graph is not None) else tf.get_default_graph()
        with tf_graph.as_default():
        
            if p_magn == 0:
                ret = tf.zeros(p_shape, dtype=p_data_type)
            
            else:
                if p_random_type == 'uniform':
                    ret = tf.random_normal(p_shape, std= p_magn, dtype=p_data_type, seed=p_seed)            
                else:
                    ret = tf.random_uniform(p_shape, minval=-p_magn, maxval=p_magn, dtype=p_data_type, seed= p_seed)
            
        return ret


def test_training():
    """
    
    """
    seed= 3
    data_dim = 4
    reinit_weights = True
    
    # generate data ->
    np.random.seed(seed)
    data = np.random.randn(10,data_dim)#.astype(np.float32)
    data2 = np.random.randn(3,data_dim)#.astype(np.float32)
    
    
    def get_ae(p_constrained=True):
        ae = Deep_Autoencoder( p_input_dim_list=[data_dim,3,2], p_activ_type='tanh', 
                     p_bias_max_val=0, p_default_dropout_rate = 0.0, p_is_constrained=p_constrained, 
                     p_last_layer_activ_type=None, p_session = None, p_seed=seed, 
                     p_summary_folder='./Log', p_clean_summary_folder=True )
        return ae
    # build the AE object
    def p1():
        with tf.Graph().as_default():
            ae = get_ae()
            
            # filt for the first time and return the last step_no
            prev_step_no = ae.train(data, p_valid_data=data, p_epoch_num=7, p_batch_size = 4, p_optim_algorithm='sgd_momentum', 
                   p_optim_params={'learning_rate': 0.001, 'momentum': 0.9},
                   p_init_step_no=0, p_epoch_eval_freq=3, p_epoch_weights_stat_freq=4,
                   p_reinit_weights=True)
            
            ae.log_graph(1)
            # Continue fitting
            prev_step_no = ae.train(data, p_valid_data=data, p_epoch_num=7, p_batch_size = 4, p_optim_algorithm='sgd_momentum', 
                   p_optim_params={'learning_rate': 0.001, 'momentum': 0.9},
                   p_init_step_no=prev_step_no, p_epoch_eval_freq=3, p_epoch_weights_stat_freq=4,
                   p_reinit_weights=reinit_weights)
            ae.log_graph(2)
            
            last_step_no = ae.train(data, p_valid_data=data, p_epoch_num=7, p_batch_size = 4, p_optim_algorithm='sgd_momentum', 
                   p_optim_params={'learning_rate': 0.001, 'momentum': 0.9},
                   p_init_step_no=prev_step_no, p_epoch_eval_freq=3, p_epoch_weights_stat_freq=4,
                   p_reinit_weights=reinit_weights)
            ae.log_graph(3)
            
            
            ev1 = ae.model_eval(data2, p_batch_size=None, p_return_transform=True)
            print(ev1)
    
            ae.save_model('./tmp','model_1')
    
    def p2():
        with tf.Graph().as_default():
            ae2 = get_ae()
            
            ae2.load_model('./tmp','model_1')
        
            ev2 = ae2.model_eval(data2, p_batch_size=None, p_return_transform=True)
            print(ev2)
    
    p1()
    p2()
    
def test_compare_with_paper(train_other=True, train_my=True):
    """
    Compare this AE with the AE from the paper: 
        Anomaly Detection with Robusr Deep Autoencoders. Chong Zhou, Randy C. Paffenroth.s
    
    """

    #import pdb; pdb.set_trace()
    # Take the data from the data forlder of RobustAutoencoder repo ->
    
    data_show_path = os.path.abspath(os.path.join('../notused_RobustAutoencoder_keepit/data')) # path is relative to notebook path.
    if data_show_path not in sys.path:
        sys.path.append(data_show_path)    
    
    
    
    x_data_file = os.path.abspath(os.path.join('../notused_RobustAutoencoder_keepit/data/data.txt'))
    y_data_file = os.path.abspath(os.path.join('../notused_RobustAutoencoder_keepit/data/y.txt'))
    
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
    p_layers = [784, 400, 200] ## S trans
    p_lr = 0.04
    p_epochs = 3 #201
    p_epoch_eval_freq = 2#20 # used only in my
    p_data_type = tf.float64 # used only in my
    p_minibatch_size = 128
    p_seed = None
    # set AE parameters <-
    
    
    # train other ae ->
    if train_other:
            # Include the repo in the path for model importing:
        re_path = os.path.abspath(os.path.join('../notused_RobustAutoencoder_keepit/experiments/Outlier Detection')) # path is relative to notebook path.
        if re_path not in sys.path:
            sys.path.append(re_path)
            
        from DAE_tensorflow import Deep_Autoencoder as other_de
        
        #x_data = x_data - np.mean(x_data, axis= 0) - 0.2345340980979827398569
        #import pdb; pdb.set_trace()
        
        with tf.Graph().as_default():
            sess1 = tf.Session()
            other_de_obj = other_de( sess1, input_dim_list=p_layers)
            other_de_obj.fit(x_data, sess1, learning_rate=p_lr,
                    iteration=p_epochs, batch_size=p_minibatch_size, verbose=True)
    
    # train other ae <-
    
    
    
    # train my ae ->
    if train_my:
        g1 = tf.Graph()
        with g1.as_default():
            my_ae = Deep_Autoencoder( p_input_dim_list=p_layers, p_activ_type='sigmoid', 
                             p_bias_max_val=0, p_default_dropout_rate = 0.0, p_data_type = p_data_type, p_is_constrained=True, 
                             p_last_layer_activ_type='sigmoid', p_session = None, p_seed=p_seed, 
                             p_summary_folder='./Log', p_clean_summary_folder=True,
                             p_cost_normalize_type='per_sample_per_dim')
        
            prev_step_no = my_ae.train(x_data, p_valid_data=x_data, p_epoch_num=p_epochs, p_batch_size = p_minibatch_size, 
                                       p_valid_batch_size=None, p_optim_algorithm='sgd', 
                       p_optim_params={'learning_rate': p_lr},
                       p_init_step_no=0, p_epoch_eval_freq=p_epoch_eval_freq, p_epoch_weights_stat_freq=20,
                       p_verbose_level=1)
    # train my ae <-
    with g1.as_default():
        eval_batch_size = None
        eval_cost, _, samp_num = my_ae.model_eval(x_data, p_eval_data=None, p_batch_size=eval_batch_size, p_return_transform=False)
        print("Eval cost: ", eval_cost, "  eval_batch: ", eval_batch_size, "  eval samples: ", samp_num )
        
        eval_batch_size =  1
        eval_cost, _, samp_num = my_ae.model_eval(x_data, p_eval_data=None, p_batch_size=eval_batch_size, p_return_transform=False)
        print("Eval cost: ", eval_cost, "  eval_batch: ", eval_batch_size, "  eval samples: ", samp_num )
    
    


if __name__ == "__main__":
    
    
#    ph = tf.placeholder(tf.float32, shape=[5, None], name='ph')
#    
#    ph_shape  = tf.shape(ph)
#    bb = tf.constant([[1,],[2,],[3.0,]] )
#
#
#    xx = tf.matmul( tf.random_normal( (3,5)  ), ph  ) + bb    
    
    #init = tf.global_variables_initializer()
    #sess = tf.Session( )
    #sess.run(init)
    
    ##ph_shape_val, xx_val = sess.run( [ph_shape,xx], {ph: np.random.normal(size=(5,10) )} )
    
    #xx_val = xx.eval({ph: np.random.normal(size=(5,10) )}, session=sess)
    
    
#    ## Test dropout 
#    ph = tf.placeholder(tf.float32, shape=[None, 5], name='ph')
#    
#    out = tf.nn.dropout(ph, keep_prob=0.2, noise_shape=[1,5])
#    
#    init = tf.global_variables_initializer()
#    sess = tf.Session( )
#    sess.run(init)
#    
#    out_val = sess.run( [out], {ph: np.random.normal(size=(10,5) )} )
#    
#    
#    
    #data_load_test()
    # ret = test_training()
    
    test_compare_with_paper()
    sys.exit()
    
    
    #run_no = int(sys.argv[1])
   # 
   # if run_no == 1:
   #     run1()
   # elif (run_no == 2):
   #     run2()
   # elif (run_no == 3):
   #     run3()
   # elif (run_no == 4):
   #     run4()
   # elif (run_no == 5):
  #      run5()
  #  elif (run_no == 6):
   #     run6()
   # elif (run_no == 7):
   #     run7()
   # else:
   #     raise ValueError("run_no is wrong!")
        
    
    
    
    
    
    
    
    
    