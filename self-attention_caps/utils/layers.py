# Copyright 2021 Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf


class SquashHinton(tf.keras.layers.Layer):
    """
    Squash activation function presented in 'Dynamic routinig between capsules'.

    ...
    
    Attributes
    ----------
    eps: int
        fuzz factor used in numeric expression
 
    Methods
    -------
    call(s)
        compute the activation from input capsules

    """

    def __init__(self, eps=10e-21, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, s):
        n = tf.norm(s,axis=-1,keepdims=True)
        return tf.multiply(n**2/(1+n**2)/(n+self.eps), s)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'eps': self.eps}

    def compute_output_shape(self, input_shape):
        return input_shape



class Squash(tf.keras.layers.Layer):
    """
    Squash activation used in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'.
    
    ...
    
    Attributes
    ----------
    eps: int
        fuzz factor used in numeric expression
    
    Methods
    -------
    call(s)
        compute the activation from input capsules
    """

    def __init__(self, eps=10e-21, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, s):
        n = tf.norm(s,axis=-1,keepdims=True)
        return (1 - 1/(tf.math.exp(n)+self.eps))*(s/(n+self.eps))

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'eps': self.eps}

    def compute_output_shape(self, input_shape):
        return input_shape




class PrimaryCaps(tf.keras.layers.Layer):
    """
    Create a primary capsule layer with the methodology described in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing'. 
    Properties of each capsule s_n are exatracted using a 2D depthwise convolution.
    
    ...
    
    Attributes
    ----------
    F: int
        depthwise conv number of features
    K: int
        depthwise conv kernel dimension
    N: int
        number of primary capsules
    D: int
        primary capsules dimension (number of properties)
    s: int
        depthwise conv strides
    Methods
    -------
    call(inputs)
        compute the primary capsule layer
    """
    def __init__(self, F, K, N, D, s=1, **kwargs):
        super(PrimaryCaps, self).__init__(**kwargs)
        self.F = F
        self.K = K
        self.N = N
        self.D = D
        self.s = s
        
    def build(self, input_shape):    
        self.DW_Conv2D = tf.keras.layers.Conv2D(self.F, self.K, self.s,
                                             activation='linear', groups=self.F, padding='valid')

        self.built = True
    
    def call(self, inputs):      
        x = self.DW_Conv2D(inputs)      
        x = tf.keras.layers.Reshape((self.N, self.D))(x)
        x = Squash()(x)
        
        return x
    
    def get_config(self):
        config = {
            'F': self.F,
            'K': self.K,
            'N': self.N, 
            'D': self.D, 
            's': self.s
        }
        base_config = super().get_config()
        return {**base_config, **config}
    


class FCCaps(tf.keras.layers.Layer):
    """
    Fully-connected caps layer. It exploites the routing mechanism, explained in 'Efficient-CapsNet: Capsule Network with Self-Attention Routing', 
    to create a parent layer of capsules. 
    
    ...
    
    Attributes
    ----------
    N: int
        number of digit capsules
    D: int
        digit capsules dimension (number of properties)
    kernel_initilizer: str
        matrix W initialization strategy
 
    Methods
    -------
    call(inputs)
        compute the primary capsule layer
    """
    def __init__(self, N, D, kernel_initializer='he_normal', **kwargs):
        super(FCCaps, self).__init__(**kwargs)
        self.N = N
        self.D = D
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        
    def build(self, input_shape):
        input_N = input_shape[-2]
        input_D = input_shape[-1]

        self.W = self.add_weight(shape=[self.N, input_N, input_D, self.D],initializer=self.kernel_initializer,name='W')
        self.b = self.add_weight(shape=[self.N, input_N,1], initializer=tf.zeros_initializer(), name='b')
        self.built = True
    
    def call(self, inputs, training=None):
        
        u = tf.einsum('...ji,kjiz->...kjz',inputs,self.W)    # u shape=(None,N,H*W*input_N,D)

        c = tf.einsum('...ij,...kj->...i', u, u)[...,None]        # b shape=(None,N,H*W*input_N,1) -> (None,j,i,1)
        c = c/tf.sqrt(tf.cast(self.D, tf.float32))
        c = tf.nn.softmax(c, axis=1)                             # c shape=(None,N,H*W*input_N,1) -> (None,j,i,1)
        c = c + self.b
        s = tf.reduce_sum(tf.multiply(u, c),axis=-2)             # s shape=(None,N,D)
        v = Squash()(s)       # v shape=(None,N,D)
        
        return v, c

    def compute_output_shape(self, input_shape):
        return (None, self.N, self.D) # slf.C, self.L

    def get_config(self):
        config = {
            'N': self.N,
            'D': self.D,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))



class Length(tf.keras.layers.Layer):
    """
    Compute the length of each capsule n of a layer l.
    ...
    
    Methods
    -------
    call(inputs)
        compute the length of each capsule
    """

    def call(self, inputs, **kwargs):
        """
        Compute the length of each capsule
        
        Parameters
        ----------
        inputs: tensor
           tensor with shape [None, num_capsules (N), dim_capsules (D)]
        """
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), - 1) + tf.keras.backend.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super().get_config()
        return config


class MyAttention_Zero_Head(tf.keras.layers.Layer):
    """
    Fully-connected caps layer with single head attention routing (no projection). 
    
    ...
    
    Attributes
    ----------
    kernel_initilizer: str
        matrix W initialization strategy
    with_capsules: bool
        If set to true, the Q,K,V are expected to have shape tensor rank == 4. ([batch size, n_parent_capsules (== n^{L+1}), n_child_capsules (== n^L), d^{L+1}])
    A_with_relu: bool
        If True, Attention matrix contains only positive values. Negative values are set to zero. Also , embeddings are computed using the non-negative attention maps.
 
    Methods
    -------
    call(inputs)
        Expects to recieve three inputs as a tuple in the following order: (query, key, value).
        The last three dimensions of the inputs should be [n^{L+1}, n^L, d^{L+1}].
    """
    def __init__(self, kernel_initializer='he_normal', with_capsules=True, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.with_capsules = with_capsules
        self.relu = tf.keras.layers.get()
        self.activation = tf.keras.activations.get(activation)
        if activation!= None:
            self.non_linearity = tf.keras.layers.Activation(self.activation)
        
    
    def call(self, inputs, training=None):
        '''
        Expect inputs in the following order: query, key and value 
        '''
        # Q, K, V shape, in the case of capsules, may be: [Batch_size, n^{L+1}, n^L, d^{L+1}]
        Q,K,V = inputs
           
        Attention_matrix = tf.matmul(Q,K, transpose_b=True) # shape of attention matrix [Batch_size, n^{L+1}, n^L,n^L]
        if self.activation!= None:
            Attention_matrix = self.non_linearity(Attention_matrix)
        new_embeddings = tf.matmul(Attention_matrix, V) # shape now is [Batch_size, n^{L+1}, n^L, d_v] (d_v usually == d^{L+1})
        
        return new_embeddings, Attention_matrix
    
    def compute_output_shape(self, input_shape):
        if self.with_capsules:
            return (input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][3]), (input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][2])
        else:
            return (input_shape[0][0], input_shape[0][1], input_shape[0][2]), (input_shape[0][0], input_shape[0][1], input_shape[0][2])

    def get_config(self):
        config = {
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'with_capsules': self.with_capsules,
            'activation': tf.keras.activations.serialize(self.activation)
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
   
class MyAttention_Multihead_projection(tf.keras.layers.Layer):
    """
    Fully-connected caps layer with multihead attention routing (even if heads = 1, it performs projection). 
    
    ...
    
    Attributes
    ----------
    D_model: int
        input digit capsules dimension (number of properties) or length of each embedding (last dimension)
        When in a capsule environment, D_model is the depth of the votes V^L (which is d^{L+1}).
    D_out: int
        output dimension of each parent capsule (normally, same as D_model)
        In the thesis we denote this quontity as d^{L+1}
        If set, the attention output is projected using Wo to a new dimension (D_out).
    kernel_initilizer: str
        matrix W initialization strategy
    A: int
        number of attention heads. If attention heads is set to 0 then no projection is used.
    D_v: int
        value dimension of each attention head (usually, A * D_v = D_out)
    D_k: int
        query & key dimension of each attention head
        usually, so as to not increase the parameters
        keep A * D_k = D_out
    use_biases: bool
        whether to use biases between transformations.
    activation: str
        Name of the non linearity that will be applied on each element of the attention matrix individually. In the algorithms, the ReLU is used (so just pass 'relu').
        Generally, we want a function that supports >0 values and minimizes the effect of values < 0. This is not part of the official attention.
        If None, no non-linearity is applied in a element-wise manner.
    return_meaned_heads: bool
        If set to True, the returned attention heads are summed along the head axis (and devided by the number of heads). This could be done externally but for a polymorphic use (with no projection class) is done internally.
    with_capsules: bool
        If set to true, the Q,K,V are expected to have shape length == 4. ([batch size, n^{L+1}, n^L, d^{L+1}])
    softmax_along_attention_rows: bool
        If set to True, then we get the original implementation where we softmax along the rows of the attention maps.
        In our case, we don't want that behavior so default is False.
 
    Methods
    -------
    call(inputs)
        Expects to recieve three inputs as a tuple in the following order: (query, key, value).
        The last two dimensions of the inputs should be [N_L,D_model].
    """
    def __init__(self, D_model=16, D_out=16, kernel_initializer='he_normal', A=2, D_k=8, D_v=8, use_biases=False, activation='relu', return_meaned_heads=True, with_capsules=True, softmax_along_attention_rows=False, **kwargs):
        super().__init__(**kwargs)
        self.D_model = D_model # d^{L+1} of V^L (d_in)
        self.D_out = D_out # d^{L+1} of C^{L+1} (d_out)
        self.A = A
        self.D_k = D_k
        self.D_v = D_v
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.biases = use_biases
        self.with_capsules = with_capsules
        self.return_meaned_heads = return_meaned_heads
        self.softmax_along_attention_rows = softmax_along_attention_rows
        self.activation = tf.keras.activations.get(activation)
        if activation!= None:
            self.non_linearity = tf.keras.layers.Activation(self.activation)
        if softmax_along_attention_rows:
            self.softmaxing =tf.keras.layers.Softmax(axis=-2)
        
        tf.debugging.Assert(self.A >= 1, 'A attribute when initializing MyAttention class can not be less than 1.')
        
        # initialize weights. These weights are shared across parent capsules.
        self.Wv = self.add_weight(shape=[self.D_model, self.A, self.D_v], initializer=self.kernel_initializer, name='W_v', trainable=True, dtype="float32") # [d^{L+1}, A, d_v^L] 
        self.Wq = self.add_weight(shape=[self.D_model, self.A, self.D_k], initializer=self.kernel_initializer, name='W_q', trainable=True, dtype="float32") # [d^{L+1}, A, d_k^L] 
        self.Wk = self.add_weight(shape=[self.D_model, self.A, self.D_k], initializer=self.kernel_initializer, name='W_k', trainable=True, dtype="float32") # [d^{L+1}, A, d_k^L] 
        self.Wo = self.add_weight(shape=[self.D_v, self.A, self.D_out], initializer=self.kernel_initializer, name='W_o', trainable=True, dtype="float32") # [d_v^L, A, d^{L+1}]
        
        if self.biases:
            # initialize biases.
            self.bv = self.add_weight(shape=[self.A, self.D_v], initializer='zeros', name="b_v", trainable=True, dtype='float32') # [A, d_v^L]
            self.bq = self.add_weight(shape=[self.A, self.D_k], initializer='zeros', name="b_q", trainable=True, dtype='float32') # [A, d_k^L]
            self.bk = self.add_weight(shape=[self.A, self.D_k], initializer='zeros', name="b_k", trainable=True, dtype='float32') # [A, d_k^L]
            self.bo = self.add_weight(shape=[self.D_out], initializer='zeros', name="b_o", trainable=True, dtype='float32') # [d^{L+1}]

    
    def call(self, inputs, training=None):
        '''
        Expect inputs in the following order: query, key and value 
        '''
        # Q, K, V shape, in the case of capsules, may be: [batch_size, n^{L+1}, n^L, D^{L+1}] (It's actually the votes, three times.)
        Q,K,V = inputs
        # New implementation (for no particular reason, the one in MyAttention worked just fine). Check dokimi.py for a comparison between new and old.
        
        Q = tf.einsum('...ki,izj->...kjz', Q, self.Wq) # [batch_size, n^{L+1}, n^L, d_k^L, A]
        K = tf.einsum('...ki,izj->...kjz', K, self.Wk) # [batch_size, n^{L+1}, n^L, d_k^L, A]
        V = tf.einsum('...ki,izj->...kjz', V, self.Wv) # [batch_size, n^{L+1}, n^L, d_v^L, A]
        
        # Add biases
        if self.biases:
            Q = Q + tf.transpose(self.bq)
            K = K + tf.transpose(self.bk)
            V = V + tf.transpose(self.bv)
        
        # Compute attention matrix.  
        Attention_maps = tf.einsum('...kij,...zij->...kzj',Q,K) # [batch_size, n^{L+1}, n^L, n^L, A]
        # Optional element-wise non-linearity
        if self.activation!= None:
            Attention_maps = self.non_linearity(Attention_maps)
        # Optional Softmax along n^L rows (this is the case in original implementation).
        if self.softmax_along_attention_rows:
            Attention_maps = self.softmaxing(Attention_maps)
        
        # Finally, compute embeddings.
        Embeddings = tf.einsum('...ijk, ...jmk ->...imk', Attention_maps, V) # [batch_size, n^{L+1}, n^L, d_v^L, A]
        
        Projected_embeddings = tf.einsum('...imk, mkz->...iz', Embeddings, self.Wo)  # [batch_size, n^{L+1}, n^L, d^{L+1}] 
        
        if self.biases:
            Projected_embeddings = Projected_embeddings + self.bo
            
        if self.return_meaned_heads:
            Attention_maps = tf.reduce_mean(Attention_maps, axis=-1) # Or reduce_sum ?
        
        return Projected_embeddings, Attention_maps
    
    def compute_output_shape(self, input_shape):
        if self.return_sumed_heads:
            if self.with_capsules:
                return (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.D_out), (input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][2])
            else:
                return (input_shape[0][0], input_shape[0][1], self.D_out), (input_shape[0][0], input_shape[0][1], input_shape[0][1])
        else:
            if self.with_capsules:
                return (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.D_out), (input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][2], self.A)
            else:
                return (input_shape[0][0], input_shape[0][1], self.D_out), (input_shape[0][0], input_shape[0][1], input_shape[0][1], self.A)

    def get_config(self):
        config = {
            'D_model': self.D_model,
            'D_out': self.D_out,
            'A': self.A,
            'D_k': self.D_k,
            'D_v': self.D_v,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'use_biases': self.biases,
            'activation': tf.keras.activations.serialize(self.activation),
            'return_meaned_heads': self.return_meaned_heads,
            'with_capsules': self.with_capsules,
            'softmax_along_attention_rows': self.softmax_along_attention_rows
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
   
# Old implementation of myattention. It is ok just the compute_output_shape() needs some fixing but it is not worth it as it is not used anymore.
# class MyAttention(tf.keras.layers.Layer):
#     """
#     Fully-connected caps layer with multihead attention routing. 
    
#     ...
    
#     Attributes
#     ----------
#     N: int
#         number of digit capsules or just number of "words"
#     D_model: int
#         input digit capsules dimension (number of properties) or length of each embedding (last dimension)
#     D_out: int
#         output dimension of each parent capsule (usually, same as D_model)
#         Otherwise, the attention output is projected using Wo to a new dimension.
#         If A is equal to 0, no projection is used so D_out equals D_v.
#     kernel_initilizer: str
#         matrix W initialization strategy
#     A: int
#         number of attention heads. If attention heads is set to 0 then no projection is used.
#     D_v: int
#         value dimension of each attention head (usually, A * D_v = D_out)
#     D_k: int
#         query & key dimension of each attention head
#         usually, so as to not increase the parameters
#         keep A * D_k = D_out
#     use_biases: bool
#         whether to use biases between transformations.
#     with_capsules: bool
#         If set to true, the Q,K,V are expected to have shape length == 4. ([batch size, N_parent_capsules, N_child_capsules, D])
 
#     Methods
#     -------
#     call(inputs)
#         Expects to recieve three inputs as a tuple in the following order: (query, key, value).
#         The last two dimensions of the inputs should be [N_L,D_model].
#     """
#     def __init__(self, N, D_model=16, D_out=16, kernel_initializer='he_normal', A=2, D_k=8, D_v=8, use_biases=False, with_capsules=True, **kwargs):
#         super(MyAttention, self).__init__(**kwargs)
#         self.N = N # Number of "words" or capsules
#         self.D_model = D_model
#         self.D_out = D_out
#         self.A = A
#         self.D_k = D_k
#         self.D_v = D_v
#         self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
#         self.biases = use_biases
#         self.with_capsules = with_capsules
        
#         tf.debugging.Assert(self.A >= 0, 'A attribute when initializing MyAttention class can not be less than 0.')
        
#         if self.A > 0:
#             # initialize weights. These weights are shared across parent capsules.
#             self.W_v = self.add_weight(shape=[self.D_model, self.A, self.D_v], initializer=self.kernel_initializer, name='W_v', trainable=True, dtype="float32")
#             self.W_q = self.add_weight(shape=[self.D_model, self.A, self.D_k], initializer=self.kernel_initializer, name='W_q', trainable=True, dtype="float32")
#             self.W_k = self.add_weight(shape=[self.D_model, self.A, self.D_k], initializer=self.kernel_initializer, name='W_k', trainable=True, dtype="float32")
#             self.W_o = self.add_weight(shape=[self.D_v * self.A, self.D_out], initializer=self.kernel_initializer, name='W_o', trainable=True, dtype="float32")
            
#             if self.biases:
#                 # initialize biases.
#                 self.b_v = self.add_weight(shape=[self.A, self.D_v], initializer='zeros', name="b_v", trainable=True, dtype='float32')
#                 self.b_q = self.add_weight(shape=[self.A, self.D_k], initializer='zeros', name="b_q", trainable=True, dtype='float32')
#                 self.b_k = self.add_weight(shape=[self.A, self.D_k], initializer='zeros', name="b_k", trainable=True, dtype='float32')
#                 self.b_o = self.add_weight(shape=[self.D_out], initializer='zeros', name="b_o", trainable=True, dtype='float32')
    
#     def build(self, batch_input_shape):
#         if self.A > 0:
#             if self.with_capsules:
#                 self.reshaper = tf.keras.layers.Reshape(target_shape=[-1, batch_input_shape[0][1], batch_input_shape[0][2], self.D_v* self.A])#tf.reshape(new_embeddings, shape=[new_embeddings.shape[0], new_embeddings.shape[1], new_embeddings.shape[2], new_embeddings.shape[-2]* new_embeddings.shape[-1]])
                
#             else:
#                 self.reshaper = tf.keras.layers.Reshape(target_shape=[-1, batch_input_shape[0][1], self.D_v* self.A])# reshape(new_embeddings, shape=[new_embeddings.shape[0], new_embeddings.shape[1], new_embeddings.shape[-2]* new_embeddings.shape[-1]])
        
#         super().build(batch_input_shape)

    
#     def call(self, inputs, training=None):
#         '''
#         Expect inputs in the following order: query, key and value 
#         '''
#         # Q, K, V shape, in the case of capsules, may be: [Batch_size, N_L+1, N, Dmodel]
#         Q,K,V = inputs
        
#         if self.A > 0:
#             Q = tf.einsum('...ki,izj->...zkj', Q, self.W_q) # result shape: [batch_size, N_L+1, N, A, D_k]
#             K = tf.einsum('...ki,izj->...zkj', K, self.W_k)
#             V = tf.einsum('...ki,izj->...zkj', V, self.W_v)
#             if self.biases:
#                 Q = Q + self.b_q
#                 K = K + self.b_k
#                 V = V + self.b_v
            
#         Attention_matrix = tf.matmul(Q,K, transpose_b=True) # shape of attention matrix [Batch_size, N_L+1, (A,) N,N]
#         new_embeddings = tf.matmul(Attention_matrix, V) # shape now is [Batch_size, N_L+1, (A,) N, D_v]
        
#         if self.A > 0:
#             # Swap head axis with N of words/capsules axis.
#             # new_embeddings = tf.experimental.numpy.swapaxes(new_embeddings, -2, -3) # Shape = [Batch_size, N_L+1, N, A, D_v]
#             if self.with_capsules:
#                 new_embeddings = tf.transpose(new_embeddings, perm=[0,1,3,2,4]) # Shape = [Batch_size, N_L+1, N, A, D_v]
#             else:
#                 new_embeddings = tf.transpose(new_embeddings, perm=[0,2,1,3]) # Shape = [Batch_size, N, A, D_v]
            
            
#             # Concatenate along head axis.  Shape = [Batch_size, N_L+1, N, D_v * A]
#             new_embeddings = self.reshaper(new_embeddings)
#             new_embeddings = tf.squeeze(new_embeddings, axis=1)
#             # if self.with_capsules:
#             #     new_embeddings = tf.keras.layers.Reshape(target_shape=[new_embeddings.shape[0], new_embeddings.shape[1], new_embeddings.shape[2], new_embeddings.shape[-2]* new_embeddings.shape[-1]])(new_embeddings) #tf.reshape(new_embeddings, shape=[new_embeddings.shape[0], new_embeddings.shape[1], new_embeddings.shape[2], new_embeddings.shape[-2]* new_embeddings.shape[-1]])
                
#             # else:
#             #     new_embeddings = tf.keras.layers.Reshape(target_shape=[new_embeddings.shape[0], new_embeddings.shape[1], new_embeddings.shape[-2]* new_embeddings.shape[-1]])(new_embeddings)# reshape(new_embeddings, shape=[new_embeddings.shape[0], new_embeddings.shape[1], new_embeddings.shape[-2]* new_embeddings.shape[-1]])
            
#             # Project to D_out
#             new_embeddings = tf.einsum('...i,iz->...z', new_embeddings, self.W_o)
#             if self.biases:
#                 new_embeddings = new_embeddings + self.b_o

        
#         return new_embeddings, Attention_matrix
    
#     def compute_output_shape(self, input_shape):
#         if self.A > 0:
#             return (input_shape[0][0], input_shape[0][1], self.N, self.D_out), (input_shape[0][0], input_shape[0][1], self.A, self.N, self.D_out)
#         else:
#             return (input_shape[0][0], input_shape[0][1], self.N, self.D_v), (input_shape[0][0], input_shape[0][1], self.N, self.D_v)

#     def get_config(self):
#         config = {
#             'N': self.N,
#             'D_model': self.D_model,
#             'D_out': self.D_out,
#             'A': self.A,
#             'A': self.A,
#             'D_k': self.D_k,
#             'D_v': self.D_v,
#             'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
#             'use_biases': self.biases,
#             'with_capsules': self.with_capsules
#         }
#         base_config = super(MyAttention, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
   

class FCCapsMultihead(tf.keras.layers.Layer):
    """
    Fully-connected caps layer with multihead attention routing. 
    
    ...
    
    Attributes
    ----------
    N: int
        number of digit capsules (n^{L+1})
    D: int
        digit capsules dimension (number of properties, d^{L+1})
    Alg1: bool
        If True, Algorithm 1 is chosen (called RooMAV). If False, Algorithm 2 is used (RoWSS).
    kernel_initilizer: str
        matrix W initialization strategy
    A: int
        number of attention heads
    QKD: int
        query & key dimension of each attention head
        usually, so as to not increase the parameters
        keep A * VD = D
    D_v: int
        value dimension of each attention head
        usually, so as to not increase the parameters
        keep A * D_v = D
    scaled_emb: bool
        If True, scale the resulting embeddings (DigitCaps) according to their Agreement Score.
    agreement_scores: bool
        There is one more variant if Alg1==False : Instead of picking rows by finding the maximum agreement score in each row, we pick rows by keeping the vector with the maximum length.
    Methods
    -------
    call(inputs)
        compute the primary capsule layer
    """
    def __init__(self, N, D, Alg1=True, kernel_initializer='he_normal', A=1, QKD=16, D_v=16, scaled_emb=True, agreement_scores=True, **kwargs):
        super(FCCapsMultihead, self).__init__(**kwargs)
        self.Alg1 = Alg1
        self.N = N
        self.D = D
        self.A = A
        self.QKD = QKD
        self.D_v = D_v
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.scaled_emb = scaled_emb
        self.agreement_scores = agreement_scores
        
    def build(self, input_shape):
        self.input_N = input_shape[-2]
        input_D = input_shape[-1]
        # flattener??
        self.W = self.add_weight(shape=[self.N, self.input_N, input_D, self.D],initializer=self.kernel_initializer,name='W')
        # self.b = self.add_weight(shape=[self.N, input_N,1], initializer=tf.zeros_initializer(), name='b')
        
        # The line below was used in version 1.0
        # self.multihead = tf.keras.layers.MultiHeadAttention(self.A, self.QKD, value_dim=16, dropout=0.0, use_bias=True, attention_axes=[2])
        # Version 2.0:
        # self.my_multihead = MyAttention(input_N, self.D)
        # Version 3.0:
        if self.A > 0:
            self.my_multihead = MyAttention_Multihead_projection(A = self.A, kernel_initializer=self.kernel_initializer, D_k=self.QKD, D_v=self.D_v, D_out=self.D, D_model=self.D)
        else:
            self.my_multihead = MyAttention_Zero_Head()
        self.built = True
    
    def call(self, inputs, training=None):
        # Compute the votes.
        u = tf.einsum('...ji,kjiz->...kjz',inputs,self.W)    # u shape=(None,N,(H*W*)input_N,D) OR in algorithmic terms: [batch_size, n^{L+1}, n^L, d^{L+1}]
        # Self-Attention
        # The line below was used in version 1.0
        # s, c = self.multihead(u,u,u, return_attention_scores=True)
        # Version 2.0 & 3.0:
        emb, AttentionMaps = self.my_multihead((u,u,u))
        # AttentionMaps shape=[batch_size, n^{L+1}, n^L, n^L (,A)] (e.g. (64, 10, 16, 16, 4))-> for each output capsule and for each attention head we have one attention map.
        # emb shape=[batch_size, n^{L+1}, n^L, d^{L+1}]
        # Attention map relates the votes with each other.
        # print("\n\nShape of attention result:", emb.shape,"\nShape of attention matrices:", AttentionMaps.shape, "\n")
        if self.Alg1:
            # Sum along the capsule votes of the prev layer (for each capsule in the last layer).
            s = tf.reduce_sum(emb,axis=-2)
            v = Squash()(s)       # v shape=(None,n^{L+1},d^{L+1})
        else: # Algorithm 2 is used
            Attention_scores = tf.reduce_sum(AttentionMaps, axis=-1) # [batch_size, n^{L+1} ,n^L]
            # Softmax attention scores capsules j in Ω_{L+1}.
            smx = tf.keras.layers.Softmax(axis=-2)
            SoftSC = smx(Attention_scores) # [batch_size, n^{L+1} ,n^L]
            if self.agreement_scores:
                Selection_criterion = SoftSC
            else:
                Selection_criterion = tf.sqrt(tf.reduce_sum(tf.square(emb), -1) + tf.keras.backend.epsilon()) # [batch_size, n^{L+1} ,n^L]
            Winner_indices = tf.argmax(Selection_criterion, axis=-1) # [batch_size, n^{L+1}]
            
            # Compute Mask
            Mask = tf.expand_dims(tf.keras.backend.one_hot(indices=Winner_indices, num_classes=self.input_N), axis=-1) # [batch_size, n^{L+1}, n^L, 1]
            s = (tf.reduce_sum(emb * Mask, axis=-2))
            if self.scaled_emb:
                winners_SC = tf.reduce_max(SoftSC, axis=-1, keepdims=True) # [batch_size, n^{L+1}, 1]
                s = winners_SC * s
            v = Squash()(s)
        return v, AttentionMaps

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.N, self.D) # [batch_size, n^{L+1}, n^L]

    def get_config(self):
        config = {
            'N': self.N,
            'D': self.D,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'A': self.A,
            'QKD': self.QKD,
            'D_v': self.D_v,
            'scaled_emb': self.scaled_emb,
            'agreement_scores': self.agreement_scores
        }
        base_config = super(FCCapsMultihead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class Length(tf.keras.layers.Layer):
    """
    Compute the length of each capsule n of a layer l.
    ...
    
    Methods
    -------
    call(inputs)
        compute the length of each capsule
    """

    def call(self, inputs, **kwargs):
        """
        Compute the length of each capsule
        
        Parameters
        ----------
        inputs: tensor
           tensor with shape [None, num_capsules (N), dim_capsules (D)]
        """
        return tf.sqrt(tf.reduce_sum(tf.square(inputs), - 1) + tf.keras.backend.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config


class Mask(tf.keras.layers.Layer):
    """
    Mask operation described in 'Dynamic routinig between capsules'.
    
    ...
    
    Methods
    -------
    call(inputs, double_mask)
        mask a capsule layer
        set double_mask for multimnist dataset
    """
    def call(self, inputs, double_mask=None, **kwargs):
        if type(inputs) is list:
            if double_mask:
                inputs, mask1, mask2 = inputs
            else:
                inputs, mask = inputs
        else:  
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            if double_mask:
                mask1 = tf.keras.backend.one_hot(tf.argsort(x,direction='DESCENDING',axis=-1)[...,0],num_classes=x.get_shape().as_list()[1])
                mask2 = tf.keras.backend.one_hot(tf.argsort(x,direction='DESCENDING',axis=-1)[...,1],num_classes=x.get_shape().as_list()[1])
            else:
                print(x)
                mask = tf.keras.backend.one_hot(indices=tf.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        if double_mask:
            masked1 = tf.keras.backend.batch_flatten(inputs * tf.expand_dims(mask1, -1))
            masked2 = tf.keras.backend.batch_flatten(inputs * tf.expand_dims(mask2, -1))
            return masked1, masked2
        else:
            masked = tf.keras.backend.batch_flatten(inputs * tf.expand_dims(mask, -1))
            return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # generation step
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config
    
