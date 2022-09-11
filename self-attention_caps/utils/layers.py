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
    
class MyAttention(tf.keras.layers.Layer):
    """
    Fully-connected caps layer with multihead attention routing. 
    
    ...
    
    Attributes
    ----------
    N: int
        number of digit capsules or just number of "words"
    D_model: int
        input digit capsules dimension (number of properties) or length of each embedding (last dimension)
    D_out: int
        output dimension of each parent capsule (usually, same as D_model)
        Otherwise, the attention output is projected using Wo to a new dimension.
        If A is equal to 0, no projection is used so D_out equals D_v.
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
    with_capsules: bool
        If set to true, the Q,K,V are expected to have shape length == 4. ([batch size, N_parent_capsules, N_child_capsules, D])
 
    Methods
    -------
    call(inputs)
        Expects to recieve three inputs as a tuple in the following order: (query, key, value).
        The last two dimensions of the inputs should be [N_L,D_model].
    """
    def __init__(self, N, D_model=16, D_out=16, kernel_initializer='he_normal', A=2, D_k=8, D_v=8, use_biases=False, with_capsules=True, **kwargs):
        super(MyAttention, self).__init__(**kwargs)
        self.N = N # Number of "words" or capsules
        self.D_model = D_model
        self.D_out = D_out
        self.A = A
        self.D_k = D_k
        self.D_v = D_v
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.biases = use_biases
        self.with_capsules = with_capsules
        
        tf.debugging.Assert(self.A >= 0, 'A attribute when initializing MyAttention class can not be less than 0.')
        
        if self.A > 0:
            # initialize weights. These weights are shared across parent capsules.
            self.W_v = self.add_weight(shape=[self.D_model, self.A, self.D_v], initializer=self.kernel_initializer, name='W_v', trainable=True, dtype="float32")
            self.W_q = self.add_weight(shape=[self.D_model, self.A, self.D_k], initializer=self.kernel_initializer, name='W_q', trainable=True, dtype="float32")
            self.W_k = self.add_weight(shape=[self.D_model, self.A, self.D_k], initializer=self.kernel_initializer, name='W_k', trainable=True, dtype="float32")
            self.W_o = self.add_weight(shape=[self.D_v * self.A, self.D_out], initializer=self.kernel_initializer, name='W_o', trainable=True, dtype="float32")
            
            if self.biases:
                # initialize biases.
                self.b_v = self.add_weight(shape=[self.A, self.D_v], initializer='zeros', name="b_v", trainable=True, dtype='float32')
                self.b_q = self.add_weight(shape=[self.A, self.D_k], initializer='zeros', name="b_q", trainable=True, dtype='float32')
                self.b_k = self.add_weight(shape=[self.A, self.D_k], initializer='zeros', name="b_k", trainable=True, dtype='float32')
                self.b_o = self.add_weight(shape=[self.D_out], initializer='zeros', name="b_o", trainable=True, dtype='float32')
    
    def build(self, batch_input_shape):
        if self.A > 0:
            if self.with_capsules:
                self.reshaper = tf.keras.layers.Reshape(target_shape=[-1, batch_input_shape[0][1], batch_input_shape[0][2], self.D_v* self.A])#tf.reshape(new_embeddings, shape=[new_embeddings.shape[0], new_embeddings.shape[1], new_embeddings.shape[2], new_embeddings.shape[-2]* new_embeddings.shape[-1]])
                
            else:
                self.reshaper = tf.keras.layers.Reshape(target_shape=[-1, batch_input_shape[0][1], self.D_v* self.A])# reshape(new_embeddings, shape=[new_embeddings.shape[0], new_embeddings.shape[1], new_embeddings.shape[-2]* new_embeddings.shape[-1]])
        
        super().build(batch_input_shape)

    
    def call(self, inputs, training=None):
        '''
        Expect inputs in the following order: query, key and value 
        '''
        # Q, K, V shape, in the case of capsules, may be: [Batch_size, N_L+1, N, Dmodel]
        Q,K,V = inputs
        
        if self.A > 0:
            Q = tf.einsum('...ki,izj->...zkj', Q, self.W_q) # result shape: [batch_size, N_L+1, N, A, D_k]
            K = tf.einsum('...ki,izj->...zkj', K, self.W_k)
            V = tf.einsum('...ki,izj->...zkj', V, self.W_v)
            if self.biases:
                Q = Q + self.b_q
                K = K + self.b_k
                V = V + self.b_v
            
        Attention_matrix = tf.matmul(Q,K, transpose_b=True) # shape of attention matrix [Batch_size, N_L+1, (A,) N,N]
        new_embeddings = tf.matmul(Attention_matrix, V) # shape now is [Batch_size, N_L+1, (A,) N, D_v]
        
        if self.A > 0:
            # Swap head axis with N of words/capsules axis.
            # new_embeddings = tf.experimental.numpy.swapaxes(new_embeddings, -2, -3) # Shape = [Batch_size, N_L+1, N, A, D_v]
            if self.with_capsules:
                new_embeddings = tf.transpose(new_embeddings, perm=[0,1,3,2,4]) # Shape = [Batch_size, N_L+1, N, A, D_v]
            else:
                new_embeddings = tf.transpose(new_embeddings, perm=[0,2,1,3]) # Shape = [Batch_size, N, A, D_v]
            
            
            # Concatenate along head axis.  Shape = [Batch_size, N_L+1, N, D_v * A]
            new_embeddings = self.reshaper(new_embeddings)
            new_embeddings = tf.squeeze(new_embeddings, axis=1)
            # if self.with_capsules:
            #     new_embeddings = tf.keras.layers.Reshape(target_shape=[new_embeddings.shape[0], new_embeddings.shape[1], new_embeddings.shape[2], new_embeddings.shape[-2]* new_embeddings.shape[-1]])(new_embeddings) #tf.reshape(new_embeddings, shape=[new_embeddings.shape[0], new_embeddings.shape[1], new_embeddings.shape[2], new_embeddings.shape[-2]* new_embeddings.shape[-1]])
                
            # else:
            #     new_embeddings = tf.keras.layers.Reshape(target_shape=[new_embeddings.shape[0], new_embeddings.shape[1], new_embeddings.shape[-2]* new_embeddings.shape[-1]])(new_embeddings)# reshape(new_embeddings, shape=[new_embeddings.shape[0], new_embeddings.shape[1], new_embeddings.shape[-2]* new_embeddings.shape[-1]])
            
            # Project to D_out
            new_embeddings = tf.einsum('...i,iz->...z', new_embeddings, self.W_o)
            if self.biases:
                new_embeddings = new_embeddings + self.b_o

        
        return new_embeddings, Attention_matrix
    
    def compute_output_shape(self, input_shape):
        if self.A > 0:
            return (input_shape[0][0], input_shape[0][1], self.N, self.D_out), (input_shape[0][0], input_shape[0][1], self.A, self.N, self.D_out)
        else:
            return (input_shape[0][0], input_shape[0][1], self.N, self.D_v), (input_shape[0][0], input_shape[0][1], self.N, self.D_v)

    def get_config(self):
        config = {
            'N': self.N,
            'D_model': self.D_model,
            'D_out': self.D_out,
            'A': self.A,
            'A': self.A,
            'D_k': self.D_k,
            'D_v': self.D_v,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'use_biases': self.biases,
            'with_capsules': self.with_capsules
        }
        base_config = super(MyAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
   

class FCCapsMultihead(tf.keras.layers.Layer):
    """
    Fully-connected caps layer with multihead attention routing. 
    
    ...
    
    Attributes
    ----------
    N: int
        number of digit capsules
    D: int
        digit capsules dimension (number of properties)
    kernel_initilizer: str
        matrix W initialization strategy
    A: int
        number of attention heads
    QKD: int
        query & key dimension of each attention head
        usually, so as to not increase the parameters
        keep A * VD = D
 
    Methods
    -------
    call(inputs)
        compute the primary capsule layer
    """
    def __init__(self, N, D, kernel_initializer='he_normal', A=1, QKD=16, **kwargs):
        super(FCCapsMultihead, self).__init__(**kwargs)
        self.N = N
        self.D = D
        self.A = A
        self.QKD = QKD
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        
    def build(self, input_shape):
        input_N = input_shape[-2]
        input_D = input_shape[-1]
        # flattener??
        self.W = self.add_weight(shape=[self.N, input_N, input_D, self.D],initializer=self.kernel_initializer,name='W')
        # self.b = self.add_weight(shape=[self.N, input_N,1], initializer=tf.zeros_initializer(), name='b')
        
        # The line below was used in version 1.0
        # self.multihead = tf.keras.layers.MultiHeadAttention(self.A, self.QKD, value_dim=16, dropout=0.0, use_bias=True, attention_axes=[2])
        # Version 2.0:
        self.my_multihead = MyAttention(input_N, self.D)
        self.built = True
    
    def call(self, inputs, training=None):
        # Compute the votes.
        u = tf.einsum('...ji,kjiz->...kjz',inputs,self.W)    # u shape=(None,N,(H*W*)input_N,D)
        # Self-Attention
        # The line below was used in version 1.0
        # s, c = self.multihead(u,u,u, return_attention_scores=True)
        # Version 2.0:
        s, c = self.my_multihead((u,u,u))
        # c shape=(None, N, A, N_input, N_input) (e.g. (None, 10, 4, 16, 16))-> for each output capsule and for each attention head we have one attention map.
        # Attention map that relates the votes with each other.
        # print("\n\nShape of attention result:", s.shape,"\nShape of attention matrices:", c.shape, "\n")
        # Sum along the capsule votes of the prev layer (for each capsule in the last layer).
        s = tf.reduce_sum(s,axis=-2)
        v = Squash()(s)       # v shape=(None,N,D)
        
        return v, c

    def compute_output_shape(self, input_shape):
        return (None, self.C, self.L)

    def get_config(self):
        config = {
            'N': self.N,
            'D': self.D,
            'A': self.A,
            'QKD': self.QKD,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer)
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
    
