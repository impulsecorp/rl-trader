import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#tf.logging.set_verbosity(tf.logging.ERROR)

class Model():
      def __init__(self,config):
        """Hyperparameters"""
        num_layers = config['num_layers']
        hidden_size = config['hidden_size']
        max_grad_norm = config['max_grad_norm']
        batch_size = config['batch_size']
        sl = config['sl']
        crd = config['crd']
        num_l = config['num_l']
        learning_rate = config['learning_rate']
        self.sl = sl
        self.batch_size = batch_size


        # Nodes for the input variables
        self.x = tf.placeholder("float", shape=[batch_size, sl], name = 'Input_data')
        self.x_exp = tf.expand_dims(self.x,1)
        self.keep_prob = tf.placeholder("float")

        with tf.variable_scope("Encoder") as scope:
            with tf.variable_scope(tf.get_variable_scope(),reuse=False):
                  #Th encoder cell, multi-layered with dropout
                  cell_enc = tf.contrib.rnn.LSTMCell(hidden_size)
                  stacked_rnn = []
                  for iiLyr in range(num_layers):
                     stacked_rnn.append(tf.contrib.rnn.LSTMCell(hidden_size))
                  cell_enc = tf.contrib.rnn.MultiRNNCell(cells=stacked_rnn)
                  cell_enc = tf.contrib.rnn.DropoutWrapper(cell_enc,output_keep_prob=self.keep_prob)

                  #Initial state
                  initial_state_enc = cell_enc.zero_state(batch_size, tf.float32)

                  outputs_enc,_ = tf.contrib.legacy_seq2seq.rnn_decoder(tf.unstack(self.x_exp,axis=2),initial_state_enc,cell_enc)
                  cell_output = outputs_enc[-1]  #Only use the final hidden state #tensor in [batch_size,hidden_size]
        with tf.name_scope("Enc_2_lat") as scope:
             with tf.variable_scope(tf.get_variable_scope(),reuse=False):
                  #layer for mean of z
                  W_mu = tf.get_variable('W_mu', [hidden_size,num_l])
                  b_mu = tf.get_variable('b_mu',[num_l])
                  self.z_mu = tf.nn.xw_plus_b(cell_output,W_mu,b_mu,name='z_mu')  #mu, mean, of latent space

                  #Train the point in latent space to have zero-mean and unit-variance on batch basis
                  lat_mean,lat_var = tf.nn.moments(self.z_mu,axes=[1])
                  self.loss_lat_batch = tf.reduce_mean(tf.square(lat_mean)+lat_var - tf.log(lat_var)-1)

        with tf.name_scope("Lat_2_dec") as scope:
            with tf.variable_scope(tf.get_variable_scope(),reuse=False):
                  #layer to generate initial state
                  W_state = tf.get_variable('W_state', [num_l,hidden_size])
                  b_state = tf.get_variable('b_state',[hidden_size])
                  z_state = tf.nn.xw_plus_b(self.z_mu,W_state,b_state,name='z_state')  #mu, mean, of latent space

        with tf.variable_scope("Decoder") as scope:
            with tf.variable_scope(tf.get_variable_scope(),reuse=False):
            
                  # The decoder, also multi-layered
                  cell_dec = tf.contrib.rnn.LSTMCell(hidden_size)
                  stacked_drnn = []
                  for iiLyr in range(num_layers):
                     stacked_drnn.append(tf.contrib.rnn.LSTMCell(hidden_size))
                  cell_dec = tf.contrib.rnn.MultiRNNCell(cells=stacked_drnn)

                  #Initial state
                  initial_state_dec = tuple([(z_state,z_state)]*num_layers)
                  dec_inputs = [tf.zeros([batch_size,1])]*sl
                  outputs_dec,_ = tf.contrib.legacy_seq2seq.rnn_decoder(dec_inputs,initial_state_dec,cell_dec)
        with tf.name_scope("Out_layer") as scope:
            with tf.variable_scope(tf.get_variable_scope(),reuse=False):
                  params_o = 2*crd   #Number of coordinates + variances
                  W_o = tf.get_variable('W_o',[hidden_size,params_o])
                  b_o = tf.get_variable('b_o',[params_o])
                  outputs = tf.concat(outputs_dec,0)                    #tensor in [sl*batch_size,hidden_size]
                  h_out = tf.nn.xw_plus_b(outputs,W_o,b_o)
                  h_mu,h_sigma_log = tf.unstack(tf.reshape(h_out,[sl,batch_size,params_o]),axis=2)
                  h_sigma = tf.exp(h_sigma_log)
                  dist = tf.contrib.distributions.Normal(h_mu,h_sigma)
                  px = dist.cdf(tf.transpose(self.x))
                  loss_seq = -tf.log(tf.maximum(px, 1e-20))             #add epsilon to prevent log(0)
                  self.loss_seq = tf.reduce_mean(loss_seq)

        with tf.name_scope("train") as scope:
            with tf.variable_scope(tf.get_variable_scope(),reuse=False):
                  #Use learning rte decay
                  global_step = tf.Variable(0,trainable=False)
                  lr = tf.train.exponential_decay(learning_rate,global_step,1000,0.1,staircase=False)


                  self.loss = self.loss_seq + self.loss_lat_batch

                  #Route the gradients so that we can plot them on Tensorboard
                  tvars = tf.trainable_variables()
                  #We clip the gradients to prevent explosion
                  grads = tf.gradients(self.loss, tvars)
                  grads, _ = tf.clip_by_global_norm(grads,max_grad_norm)
                  self.numel = tf.constant([[0]])

                  #And apply the gradients
                  optimizer = tf.train.AdamOptimizer(lr)
                  gradients = zip(grads, tvars)
                  self.train_step = optimizer.apply_gradients(gradients,global_step=global_step)
            #      for gradient, variable in gradients:  #plot the gradient of each trainable variable
            #        if isinstance(gradient, ops.IndexedSlices):
            #          grad_values = gradient.values
            #        else:
            #          grad_values = gradient
            #
            #        self.numel +=tf.reduce_sum(tf.size(variable))
            #        tf.summary.histogram(variable.name, variable)
            #        tf.summary.histogram(variable.name + "/gradients", grad_values)
            #        tf.summary.histogram(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))

                  self.numel = tf.constant([[0]])
        tf.summary.tensor_summary('lat_state',self.z_mu)
        #Define one op to call all summaries
        self.merged = tf.summary.merge_all()
        #and one op to initialize the variables
        self.init_op = tf.global_variables_initializer()

        
class AutoEncoder(object):
    def __init__(self):
        pass
    
    def fit(self,X_train,y_train):
        tf.reset_default_graph()
        num_classes = len(np.unique(y_train))
        config = {}                             #Put all configuration information into the dict
        config['num_layers'] = 2                #number of layers of stacked RNN's
        config['hidden_size'] = 90              #memory cells in a layer
        config['max_grad_norm'] = 5             #maximum gradient norm during training
        config['batch_size'] = batch_size = 10 
        config['learning_rate'] = .005
        config['crd'] = 1                       #Hyperparameter for future generalization
        config['num_l'] = num_classes        #number of units in the latent space

        plot_every = 100                        #after _plot_every_ GD steps, there's console output
        max_iterations = 1000                   #maximum number of iterations
        dropout = 0.8 
        N = X_train.shape[0]
        D = X_train.shape[1]
        config['sl'] = sl = D          #sequence length
        self.N=N
        self.batch_size=batch_size
        print('We have %s observations with %s dimensions'%(N,D))
        # Organize the classes
        
        epochs = np.floor(batch_size*max_iterations / N)
        print('Train with approximately %d epochs' %(epochs))
        """Training time!"""
        self.model = Model(config)
        self.sess = tf.Session()
    
        perf_collect = np.zeros((2,int(np.floor(max_iterations/plot_every))))

        if True:
          self.sess.run(self.model.init_op)


          step = 0      # Step is a counter for filling the numpy array perf_collect
          for i in range(max_iterations):
            batch_ind = np.random.choice(N,batch_size,replace=False)
            result = self.sess.run([self.model.loss, self.model.loss_seq,self.model.loss_lat_batch,self.model.train_step],feed_dict={self.model.x:X_train[batch_ind],self.model.keep_prob:dropout})

            if i%plot_every == 0:
              #Save train performances
              perf_collect[0,step] = loss_train = result[0]
              loss_train_seq, lost_train_lat = result[1], result[2]

              #Calculate and save validation performance


              result = self.sess.run([self.model.loss, self.model.loss_seq,self.model.loss_lat_batch,self.model.merged], feed_dict={ self.model.x: X_train[batch_ind],self.model.keep_prob:1.0})
              perf_collect[1,step] = loss_val = result[0]
              loss_val_seq, lost_val_lat = result[1], result[2]
              #and save to Tensorboard
              summary_str = result[3]


              print("At %6s / %6s train (%5.3f, %5.3f, %5.3f), val (%5.3f, %5.3f,%5.3f) in order (total, seq, lat)" %(i,max_iterations,loss_train,loss_train_seq, lost_train_lat,loss_val, loss_val_seq, lost_val_lat))
              step +=1
    def predict(self,X_Test): 
          ##Extract the latent space coordinates of the validation set
        start = 0
        z_run = []

        while start < len(X_Test):
            run_ind = range(start,start+self.batch_size)
            z_mu_fetch = self.sess.run(self.model.z_mu, feed_dict = {self.model.x:X_Test[run_ind],self.model.keep_prob:1.0})
            z_run.append(np.argmax(z_mu_fetch[:],1))
            start += self.batch_size

        z_run = np.concatenate(z_run,axis=0)
        self.sess.close()
        return z_run
