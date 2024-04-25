import numpy as np
import tensorflow as tf


## Class that defines a single Attention Head (inherits from tf.keras.layers.Layer)
## Input: embedding dimension, neurons of the attention head (dimensions of keys tensor)
##        if dim_v = None: same dimensions as the Keys tensor, unless specified with an integer
class AttentionHead(tf.keras.layers.Layer):
    def __init__(self, num_toks, emb_dim, dim_k, dim_v = None):
        ## initialize the parent class
        super().__init__()

        ## initialize the value dimension based on the input
        self.dim_k = dim_k
        self.dim_v = dim_k
        if(dim_v is not None): self.dim_v = dim_v

        ## with the dense layer
        self.input_layer = tf.keras.layers.InputLayer(input_shape = (num_toks, emb_dim))

        ## initialize the 3 tensors with no bias
        self.Wq = tf.keras.layers.Dense(self.dim_k, use_bias = False)
        self.Wk = tf.keras.layers.Dense(self.dim_k, use_bias = False)
        self.Wv = tf.keras.layers.Dense(self.dim_v, use_bias = False)

        ## initialize the softmax layer
        self.soft_max = tf.keras.layers.Softmax()


    ## define the call function
    ## Input: - shape of the mask: batch_size x #tokens x #tokens (values that should be masked need values equal to 1, unmasked values need values equal to zero)
    def call(self, inputs, mask = None):
        ## Input layer
        inputs = self.input_layer(inputs)
        
        ## calculate the Query, Key and Values vector
        q = self.Wq(inputs)
        k = self.Wk(inputs)
        v = self.Wv(inputs)

        ## dot product of the queries and keys
        dot_p = tf.linalg.matmul(q, k, transpose_b = True)/np.sqrt(self.dim_k)

        ## apply the mask
        if(mask is not None):
            dot_p -= 1.0e15*mask

        ## calculate the softmax on the dot product of the queries and keys
        softm = self.soft_max(dot_p)

        ## return the dot product with the values matrix
        return tf.linalg.matmul(softm, v)


