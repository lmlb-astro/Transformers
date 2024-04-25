import numpy as np
import tensorflow as tf

# CONSIDER MULTIPLE INPUTS

## Class that implements a multihead attention layer, inherits from the tf.keras.layers.Layer class
## Input: the total Key and Value dimensions (heads x Key dimension & heads x Values dimension),
##        the number of attention heads and the envisioned output dimension of the full multihead attention layer (dim_layer)
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, dim_k_tot, dim_v_tot, num_heads, dim_layer):
        ## initialize the parent class
        super().__init__()

        ## initialize the value dimension based on the input
        self.num_heads = num_heads
        self.dim_k = dim_k_tot
        self.dim_v = dim_v_tot
        self.dim_layer = dim_layer

        ## initialize the 3 tensors/dense layers with no bias
        self.Wq = tf.keras.layers.Dense(self.dim_k, use_bias = False)
        self.Wk = tf.keras.layers.Dense(self.dim_k, use_bias = False)
        self.Wv = tf.keras.layers.Dense(self.dim_v, use_bias = False)

        ## initialize the softmax layer
        self.soft_max = tf.keras.layers.Softmax()

        ## intialize the final W_{0} tensor 
        self.W0 = tf.keras.layers.Dense(self.dim_layer, use_bias = False)

    
    ## The call function
    ## Shape of the input (batch_size, num_tokens, token_dimension)
    ## Input: - shape of the mask: batch_size x #heads x #tokens x #tokens (values that should be masked need values equal to 1, unmasked values need values equal to zero)
    def call(self, inputs, mask = None):
        ## calculate the Query, Key and Values vector on the input
        q = self.Wq(inputs)
        k = self.Wk(inputs)
        v = self.Wv(inputs)

        ## reshape q, k & v for the dot product
        q = self.to_4D(q)
        k = self.to_4D(k)
        v = self.to_4D(v)

        ## perform the dot product for all attention heads
        dot_p = tf.linalg.matmul(q, k, transpose_b = True)/np.sqrt(k.shape[3]) ## softmax(Q*K.transpose/sqrt(dim_k))

        ## apply the mask
        if(mask is not None):
            dot_p -= 1.0e15*mask

        ##  perform the softmax and dot product with the values tensor
        softm = self.soft_max(dot_p)
        att_heads = tf.linalg.matmul(softm, v)

        ## concatenate the output of the attention heads and multiple with W_{o} for the output of the layer
        att_conc = self.concatenate_heads(att_heads)
        return self.W0(att_conc)
        

    ## reshapes the 3D tensor to a 4D tensor and organizes it for the dot product
    def to_4D(self, tensor):
        ## reshape the 3D tensor to a 4D tensor
        tensor = tf.reshape(tensor, (tensor.shape[0], tensor.shape[1], self.num_heads, -1))

        ## return a permutation of the 4D tensor organized for the dot product
        return tf.transpose(tensor, perm = [0, 2, 1, 3])


    ## Reshapes the 4D tensor to concatenate the output of all attention heads
    def concatenate_heads(self, tensor):
        ## perform permutation to concatenate the heads for each token/word
        tensor = tf.transpose(tensor, perm = [0, 2, 1, 3])

        ## reshape the 4D tensor to a 3D tensor and return
        return tf.reshape(tensor, (tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3]))








