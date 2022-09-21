'''
Define Layers
'''

import tensorflow as tf
from utils.tools import squash, tanh_vector

class PrimaryCaps(tf.keras.layers.Layer):
    '''
    Primary Capsule layer, as described in Dynamic Routing Between Capsules.
    '''
    def __init__(self, kernel=9, stride=2, vector_depth=8, capsule_filters=32, **kwargs):
        super().__init__(**kwargs)
        self.k = kernel
        self.d = vector_depth # Primary caps dimension. Dimension of capsule vector.
        self.s = stride
        self.f = capsule_filters # Primary Caps Layers

    def build(self, input_shape):
        # Create the kernel weights for convolution and the biases. kernel weights of shape (as always): kernel x kernel x input_features x output_features
        # Output features are computed as Primary_Caps_layers x depth_of_each_vector
        self.kernel = self.add_weight(shape=(self.k, self.k, input_shape[-1], self.f*self.d), initializer='glorot_uniform', name='kernel')
        self.biases = self.add_weight(shape=(self.f,self.d), initializer='zeros', name='biases')
        self.built = True
    
    def call(self, inputs):
        x = tf.nn.conv2d(inputs, self.kernel, self.s, 'VALID')
        h,w = x.shape[1:3]
        x = tf.keras.layers.Reshape((h, w, self.f, self.d))(x)
        # x /= self.f No reference of this operation found in the original paper.
        x += self.biases
        x = squash(x)
        return x
    
    def compute_output_shape(self, input_shape):
        h,w = input_shape.shape[1:3]
        return (None, (h - self.k)/self.s + 1, (w - self.k)/self.s + 1, self.f, self.d)

    def get_config(self):
        config = {
            'f': self.f,
            'd': self.d,
            'k': self.k,
            's': self.s
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class DigitCaps(tf.keras.layers.Layer):
    '''
    In this layer we implement the crazy SOM idea.
    '''
    def __init__(self, vector_depth=16, num_out_caps=10, transformation_matrices_for_each_capsule=10, l_thetas=[1, 0.2, 0.1], lr = 1.0,  **kwargs):
        super().__init__(**kwargs)
        self.c = num_out_caps
        self.d = vector_depth
        self.m = transformation_matrices_for_each_capsule
        self.l_thetas = tf.constant(l_thetas, dtype=tf.float32)
        self.lr = tf.constant(lr,dtype=tf.float32)

    def build(self, input_shape):
        assert len(input_shape) >= 5, "The input Tensor should have\
             shape=[None,height,width,input_capsule_layers,input_capsule_depth]"
        
        H = input_shape[-4]
        W = input_shape[-3]
        input_f = input_shape[-2]
        input_d = input_shape[-1]

        # Define the transformation matrices.
        
        self.W = self.add_weight(shape=[H*W*input_f, input_d, self.d*self.m], 
        initializer='glorot_uniform', name='Wij')
        # Create the output tensor. By setting the output tensor here, the outputs will be initialized once.
        # Then, in the next input, the outputs will be started from where they left of.
        # The alternative is to define a constant vector in the call method (then for shape we will add the batch size bc
        # we want to initialize for each input).
        self.digit_caps = self.add_weight(shape=[self.c, self.d], initializer=tf.keras.initializers.RandomUniform(minval=-1.,maxval=1.,seed=None), name='out_caps', trainable=False)
        
        self.built = True
    
    def call(self, inputs):
        H,W,input_f,input_d = inputs.shape[1:]          # input shape=(None,H,W,input_f,input_d)
        x = tf.reshape(inputs,(-1, H*W*input_f, input_d)) #     x shape=(None,H*W*input_f,input_d)
        u = tf.einsum('...ji,jik->...jk', x, self.W)      #     u shape=(None,H*W*input_f,c*self.d)
        u = tf.reshape(u,(-1, H*W*input_f * self.m, self.d))#     u shape=(None,H*W*input_f*tf_mappers,self.d)

        # SO-Routing algorithm
        # Winning counter for the capsules.
        #wins = self.add_weight(shape=[inputs.shape[0], self.c], initializer=tf.keras.initializers.zeros(), name='win_count', trainable=False)

        # Normalize vectors
        n = tf.norm(u, axis=-1,keepdims=True)
        # u = tf.multiply(n**2/(1+n**2)/(n + tf.keras.backend.epsilon()), u) # squash
        u = tf.multiply(tf.math.divide(tf.math.tanh(n),n), u) # tanh for vectors # OR tanh_vector(u)

        # 1) Broadcasting + tile + expand to make subtraction between self.output and input
        u = tf.expand_dims(u, -2)
        b = tf.constant([1,1,self.c,1], tf.int32)
        u = tf.tile(u,b)
        # Alternatively, self.W = self.add_weight(shape=[H*W*input_f, input_d, self.d*self.m] and u = tf.reshape(u,(-1, H*W*input_f, self.c, self.d))
        # This is different than SOM.

        # 2) Take the l2 norm (along last axis) between all i-j and for all instances in batch.
        #tf.norm(u-self.output, ord='euclidean', axis=-1, keepdims=None, name=None)
        # norms = tf.math.reduce_euclidean_norm(u-self.output, axis=-1, keepdims=False, name=None) # shape = (None, H*W*input_f*self.m, 10)

        # 3) Find the winners and their indexes (argmax). For the minimum, there should be one winner parent capsule per batch index per capsule index.
        # winners = tf.math.argmax(norms, axis=-1, output_type=tf.dtypes.int64, name=None)

        sparse_updates_all = tf.zeros_like(u) # u after tiling

        # Used if we enable weights acording to the number of wins.
        # win_count_all = tf.zeros(shape=[self.c], dtype=tf.int64)

        # Get the difference used in update.
        differences = u-self.digit_caps # Try using only the u

        # Create a masked digit caps in case we want multiple iterations to create a sence of neighborhood. Using masked_digit_caps we won't pick the same winners.
        masked_digit_caps = tf.ones_like(u)
        # Compute final similarities here or at the end, after updating digit_caps?
        # similarities_final = tf.reduce_sum(tf.math.multiply(u,digit_caps),axis=-1)

        # List of neighboorhood
        
        

        for theta in self.l_thetas:
            #### IN LOOP ####
            # Get similarities
            similarities = tf.reduce_sum(tf.math.multiply(u*masked_digit_caps,self.digit_caps),axis=-1)

            # Get for each vote, which digit_caps is the winner.
            winners = tf.math.argmax(similarities, axis=-1, output_type=tf.dtypes.int64, name=None)

            # If you want, you can compute the winner ratio
            # winners_f = tf.reshape(winners,shape=[-1,*winners.shape[2:]])
            # y, idx, count = tf.unique_with_counts(winners_f, out_idx=tf.dtypes.int32, name=None)
            # win_count = tf.scatter_nd(tf.expand_dims(y,axis=-1), count, shape=[self.c], name=None)
            # win_count_all += win_count

            # Alternatively, we could softmax the similarities instead of having hard winners. If you do so, choose a bigger learning rate. Probably, then you should not use iterations (just one).

            # Convert winners to one hot vectors. Then multiply them with similarities etc. to keep win similarities. Try to multiply it with difference vectors to form the update vector.
            winners_mask = tf.keras.backend.one_hot(indices=winners, num_classes=self.c)

            
            # If you want to take into account the similarity of the pairs in the update, uncomment the two lines below and comment the third one.
            #sparse_similarities = similarities * winners_mask
            #sparse_updates = differences * tf.expand_dims(sparse_similarities,axis=-1) * self.lr
            sparse_updates = differences * tf.expand_dims(winners_mask,axis=-1) * self.lr * theta

            # Update Mask so as to not choose the same winners as the next neighboors.
            masked_digit_caps = masked_digit_caps - tf.expand_dims(winners_mask, axis=-1)

            # Add the neighboors updates (if any)
            sparse_updates_all = sparse_updates + sparse_updates_all

        #### OUT OF LOOP ####
        # Now we average/sum the updates along the vote and batch axis.
        # updates = tf.reduce_sum(sparse_updates_all, axis=[0,1])
        updates = tf.reduce_mean(sparse_updates_all, axis=[0,1])

        # If you want, you can compute the winner ratio
        # win_weights = tf.cast(win_count/idx.shape[0], dtype=tf.float32)

        # Then, take the tanh
        # assignment_attributes = tf.expand_dims(tf.math.tanh(win_weights), axis=-1)
        # And use it to update the digits
        # digit_caps = digit_caps +  tf.expand_dims(win_weights, axis=-1) * updates
        # Or if you choose to use u instead of u - digit_caps in the difference, you could try this:
        # digit_caps = (1-assignment_attributes)*digit_caps + assignment_attributes * updates

        
        self.digit_caps = self.digit_caps + updates

        # Take the norm of digit_caps (and repeat).
        d_norm = tf.norm(self.digit_caps, axis=-1, keepdims=True)
        self.digit_caps = tf.divide(self.digit_caps, d_norm)

        # Re-compute similarities with the updated vector.
        similarities_final = tf.reduce_sum(tf.math.multiply(u,self.digit_caps),axis=-1)
        # Compute the output (if we have multiple iterations, this will be out of the loop)
        output = tf.math.softmax(tf.reduce_mean(similarities_final, axis=-2), axis=-1)


        # 4) Update at once the out_caps weights summing along the batch axis. Only the winning out_caps will have an no zero value in their update vector.
        #    The winning child capsules will have the eucledian distance (which can be multiplied by learning rate). [Also the inverse eucledian
        #    distance can be used as a win metric instead of the win counter.]
        return output

    def compute_output_shape(self, input_shape):
        return (None, self.c, self.d)

    def get_config(self):
        config = {
            'c': self.c,
            'd': self.d
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Length(tf.keras.layers.Layer):
    """
    Compute the length of each capsule n of a layer l.
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

    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  
            inputs, mask = inputs
        else:  
            x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            mask = tf.keras.backend.one_hot(indices=tf.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

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
    
