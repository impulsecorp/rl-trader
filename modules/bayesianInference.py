import numpy as np
from edward.models import Categorical
from edward.models import Normal
import tensorflow as tf
import edward as ed
ed.set_seed(314159)
class BayesianInference(object):
    def __init__(self):
        pass
    def fit(self,X,Y):
        N = 100   # number of images in a minibatch.
        D = X.shape[1]  # number of features.
        K = len(np.unique(Y))    # number of classes.
        # Create a placeholder to hold the data (in minibatches) in a TensorFlow graph.
        x = tf.placeholder(tf.float32, [None, D])
        # Normal(0,1) priors for the variables. Note that the syntax assumes TensorFlow 1.1.
        w = Normal(loc=tf.zeros([D, K]), scale=tf.ones([D, K]))
        b = Normal(loc=tf.zeros(K), scale=tf.ones(K))
        # Categorical likelihood for classication.
        y = Categorical(tf.matmul(x,w)+b)
        # Contruct the q(w) and q(b). in this case we assume Normal distributions.
        self.qw = Normal(loc=tf.Variable(tf.random_normal([D, K])),
                      scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, K]))))
        self.qb = Normal(loc=tf.Variable(tf.random_normal([K])),
                      scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))
        # We use a placeholder for the labels in anticipation of the traning data.
        y_ph = tf.placeholder(tf.int32, [N])
        # Define the VI inference technique, ie. minimise the KL divergence between q and p.
        inference = ed.KLqp({w: self.qw, b: self.qb}, data={y:y_ph})
        # Initialse the infernce variables
        inference.initialize(n_iter=5000, n_print=100, scale={y: float(1) / N})
        self.sess = tf.InteractiveSession()
        # Initialise all the vairables in the session.
        tf.global_variables_initializer().run()
        #Let the training begin. We load the data in minibatches and update the VI infernce using each new batch.
        for _ in range(inference.n_iter):
            X_batch, Y_batch = self.data_iterator(N,X,Y).__next__()
            # TensorFlow method gives the label data in a one hot vetor format. We convert that into a single label.
            #Y_batch = np.argmax(Y_batch,axis=1)
            info_dict = inference.update(feed_dict={x: X_batch, y_ph: Y_batch})
            inference.print_progress(info_dict)
    def predict(self,X_Test):
        w_samp = self.qw.sample()
        b_samp = self.qb.sample()
        # Also compue the probabiliy of each class for each (w,b) sample.
        prob = tf.nn.softmax(tf.matmul( X_Test,tf.cast(w_samp,tf.float64) ) +tf.cast( b_samp,tf.float64))
        preds = np.argmax(prob.eval(),axis=1).astype(np.int16)
        return  preds
    def predict_proba(self,X_Test):
        w_samp = self.qw.sample()
        b_samp = self.qb.sample()
        # Also compue the probabiliy of each class for each (w,b) sample.
        prob = tf.nn.softmax(tf.matmul( X_Test,tf.cast(w_samp,tf.float64) ) +tf.cast( b_samp,tf.float64))
        
        return  prob.eval()
    def data_iterator(self,batch_size,features,labels):
        """ A simple data iterator """
        batch_idx = 0
        while True:
            # shuffle labels and features
            idxs = np.arange(0, len(features))
            np.random.shuffle(idxs)
            shuf_features = features[idxs]
            shuf_labels = labels[idxs]
            batch_size = batch_size
            for batch_idx in range(0, len(features), batch_size):
                features_batch = shuf_features[batch_idx:batch_idx+batch_size]
                labels_batch = shuf_labels[batch_idx:batch_idx+batch_size]
                yield features_batch, labels_batch
