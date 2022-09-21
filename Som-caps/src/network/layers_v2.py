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
    def __init__(self, vector_depth=16, num_out_caps=10, transf_mat_for_each_caps=10, reduced_votes=False, iter=1, softmax=False, tanh_like=False, l_thetas=[1.0],
     lr_som = 1.0, radical=False, normalize_digit_caps=False, normalize_votes=False, norm_type=0, take_into_account_similarity=False, take_into_account_winner_ratios=False, normalize_d_in_loop=False,  **kwargs):
        super().__init__(**kwargs)
        self.c = num_out_caps
        self.d = vector_depth
        self.m = transf_mat_for_each_caps
        self.l_thetas = tf.constant(l_thetas, dtype=tf.float32)
        self.lr = tf.constant(lr_som, dtype=tf.float32)
        self.reduced_votes = reduced_votes
        self.iterations = iter
        self.softmax = softmax
        self.radical = radical
        self.tanh_like = tanh_like
        self.normalize_d = normalize_digit_caps
        self.normalize_u = normalize_votes
        self.norm_type = norm_type
        self.take_into_account_similarity = take_into_account_similarity
        self.take_into_account_winner_ratios = take_into_account_winner_ratios
        self.normalize_d_in_loop = normalize_d_in_loop
    
    def build(self, input_shape):
        assert len(input_shape) >= 5, "The input Tensor should have\
             shape=[None,height,width,input_capsule_layers,input_capsule_depth]"
        
        H = input_shape[-4]
        W = input_shape[-3]
        input_f = input_shape[-2]
        input_d = input_shape[-1]

        # Define the transformation matrices.
        if self.reduced_votes:
            self.W = self.add_weight(shape=[H*W*input_f, input_d, self.d*self.c], name='W', initializer=tf.keras.initializers.he_normal(), trainable=True)
        else:
            self.W = self.add_weight(shape=[H*W*input_f, input_d, self.d*self.m], name='W', initializer=tf.keras.initializers.he_normal(), trainable=True)

        self.digit_caps = self.add_weight(shape=[self.c, self.d], initializer=tf.keras.initializers.RandomUniform(minval=-1.,maxval=1.,seed=None), name='out_caps', trainable=False)
        self.built = True
    
    def call(self, inputs):
        H,W,input_f,input_d = inputs.shape[1:]          # input shape=(None,H,W,input_f,input_d)
        x = tf.reshape(inputs,(-1, H*W*input_f, input_d)) #     x shape=(None,H*W*input_f,input_d)
        u = tf.einsum('...ji,jik->...jk', x, self.W)      #     u shape=(None,H*W*input_f,c*self.d)
        
        # Note that if reduced_votes is True, transf_mat_for_each_caps dose nothing (and should be equal to output classes).
        # If reduced_votes is True, we use the transformation matrices to map each primary capsule to as many votes as the number of classes. 
        # Each vote which belongs to a certain class is compared (using a similarity measure) to the respective digit_caps vector. So, when
        # reduced_votes is True, the competing votes stem from different transformation matrices. On the other hand, if reduced_votes is False,
        # we once again map the primary capsules to as many votes as the number 'transf_mat_for_each_caps' which can now be independent from the
        # number of classes. That is because the votes pertaining different classes share the same transformation matrices. Consequently, the metrics
        # of the som algorithm are not affected by the 'learning' of the matrices.
        if not self.reduced_votes:
            u = tf.reshape(u,(-1, H*W*input_f*self.m, self.d))
            # Broadcasting + tile + expand to make subtraction between self.output and input
            # In simple terms, we copy the votes as many times as the classes (so as to take the similarity measure).
            u = tf.expand_dims(u, -2)
            b = tf.constant([1,1,self.c,1], tf.int32)
            u = tf.tile(u,b)
        else:
            u = tf.reshape(u,(-1, H*W*input_f, self.c, self.d))
            

        
        # SO-Routing algorithm
        # Winning counter for the capsules.
        #wins = self.add_weight(shape=[inputs.shape[0], self.c], initializer=tf.keras.initializers.zeros(), name='win_count', trainable=False)

        # Normalize vectors
        if self.normalize_u:
            n = tf.norm(u, axis=-1,keepdims=True)
            if self.norm_type==0:
                u = tf.multiply(n**2/(1+n**2)/(n + tf.keras.backend.epsilon()), u)
            else:
                u = tf.multiply(tf.math.divide(tf.math.tanh(n),n), u) # tanh for vectors # OR tanh_vector(u)


        # Used if we enable weights acording to the number of wins.
        if self.take_into_account_winner_ratios:
            win_count_all = tf.zeros(shape=[self.c], dtype=tf.int64)
        
        for i in range(self.iterations):

            sparse_updates_all = tf.zeros_like(u) # u after tiling

            # Get the difference used in update.
            if not self.radical:
                # Original.
                differences = u-self.digit_caps
            else:
                differences = u

            # Create a masked digit caps in case we want multiple iterations to create a sence of neighborhood. Using masked_digit_caps we won't pick the same winners.
            masked_digit_caps = tf.ones_like(u)
            # Compute final similarities here or at the end, after updating digit_caps?
            #similarities_final = tf.reduce_sum(tf.math.multiply(u,digit_caps),axis=-1)

            for theta in self.l_thetas:
        
                #### IN LOOP ####
                # Get similarities
                similarities = tf.reduce_sum(tf.math.multiply(u*masked_digit_caps,self.digit_caps),axis=-1)

                # Get for each vote, which digit_caps is the winner.
                if self.softmax:
                    # Alternatively, we could softmax the similarities instead of having hard winners. If you do so, choose a bigger learning rate. Probably, then you should not use iterations (just one).
                    # Instead of hard winners, we have soft winners == every digit is updated according to their similarity. What if instead of softmax we used tanh?? then dissimilar digitvectors would
                    # go furhter away.
                    winners_mask = tf.math.softmax(similarities, axis=-1, name=None) 
                elif self.tanh_like:
                    winners_mask = (tf.math.softmax(similarities, axis=-1, name=None) - 0.5) * 2.0
                else:
                    # Original case.
                    winners = tf.math.argmax(similarities, axis=-1, output_type=tf.dtypes.int64, name=None)
                    # Convert winners to one hot vectors. Then multiply them with similarities etc. to keep win similarities. Try to multiply it with difference vectors to form the update vector.
                    winners_mask = tf.keras.backend.one_hot(indices=winners, num_classes=self.c)
                
               
                # If you want, you can compute the winner ratio
                if self.take_into_account_winner_ratios:
                    winners_f = tf.reshape(winners,shape=[-1,*winners.shape[2:]])
                    y, idx, count = tf.unique_with_counts(winners_f, out_idx=tf.dtypes.int32, name=None)
                    win_count = tf.scatter_nd(tf.expand_dims(y,axis=-1), count, shape=[self.c], name=None)
                    win_count_all += win_count
                
                # If you want to take into account the similarity of the pairs in the update, uncomment the two lines below and comment the third one.
                if self.take_into_account_similarity:
                    sparse_similarities = similarities * winners_mask
                    sparse_updates = differences * tf.expand_dims(sparse_similarities,axis=-1) * self.lr
                else:
                    sparse_updates = differences * tf.expand_dims(winners_mask,axis=-1) * self.lr * theta

                # Add the neighboors updates (if any)
                sparse_updates_all = sparse_updates + sparse_updates_all

            #### OUT OF LOOP ####
            # Now we average/sum the updates along the vote and batch axis.
            updates = tf.reduce_mean(sparse_updates_all, axis=[0,1])

            
            if not self.take_into_account_winner_ratios:
                # If not take winner ratios into account, procceed normally.
                # Actually, the more the winners, the greater the mean value of update so no need to
                # consider neither win ratios nor similarities when updating digit caps.
                if self.radical:
                    # If radical changes are applied, we implement a moving average over
                    #d = ((self.iterations - 1.0)*self.digit_caps + updates)/self.iterations
                     self.digit_caps.assign(((self.iterations - 1.0)*self.digit_caps + updates)/self.iterations)

                else:
                    # Classical som update.
                    self.digit_caps.assign(self.digit_caps + updates)
                    #d = d + updates
            else:
                # If you want, you can compute the winner ratio
                win_weights = tf.cast(win_count_all/idx.shape[0], dtype=tf.float32)

                # Then, take the tanh
                assignment_attributes = tf.expand_dims(tf.math.tanh(win_weights), axis=-1)
                # And use it to update the digits
                if not self.radical:
                    self.digit_caps.assign(self.digit_caps +  assignment_attributes * updates)
                else:
                    # Or if you choose to use u instead of u - digit_caps in the difference, you could try this:
                    self.digit_caps.assign((1-assignment_attributes)*self.digit_caps + assignment_attributes * updates)
                    # We advise to use this with d normalization.
            
            # Take the norm of digit_caps in the loop.
            if self.normalize_d_in_loop:
                if self.norm_type==0:
                    self.digit_caps.assign(self.digit_caps)
                else:
                    d_norm = tf.norm(self.digit_caps, axis=-1, keepdims=True)
                    self.digit_caps.assign(tf.divide(self.digit_caps, d_norm))

        # Re-compute similarities with the updated vector.
        similarities_final = tf.reduce_sum(tf.math.multiply(u,self.digit_caps),axis=-1)
        # Compute the output (if we have multiple iterations, this will be out of the loop)
        output = tf.reduce_mean(similarities_final, axis=-2)

        # Take the norm of digit_caps.
        if self.normalize_d_in_loop:
                if self.norm_type==0:
                    self.digit_caps.assign(self.digit_caps)
                else:
                    d_norm = tf.norm(self.digit_caps, axis=-1, keepdims=True)
                    self.digit_caps.assign(tf.divide(self.digit_caps, d_norm))


        return output, self.digit_caps

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, x):
        if self.trainable:  
            #x = tf.sqrt(tf.reduce_sum(tf.square(inputs), -1))
            mask = tf.keras.backend.one_hot(indices=tf.argmax(x, 1), num_classes=x.get_shape().as_list()[1])
        else:
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
    
