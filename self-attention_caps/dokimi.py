from curses.ascii import SO
from re import S
import tensorflow as tf
import numpy as np


# Initialize
shapeW = [5,4,3,6] # (n^{L+1}, n^L ,d^L, D^{L+1})
W = tf.range(np.prod(shapeW), dtype=tf.int64)
W = tf.reshape(W,shapeW)

shapeCL = [1, 4, 3]# [batch_size, n^L, D^L]
CL = tf.range(np.prod(shapeCL), dtype=tf.int64)
CL = tf.reshape(CL, shapeCL)

# Compute votes
V = tf.einsum('...ji,kjiz->...kjz',CL,W) # [batch_size, n^{L+1}, n^L, D^{L+1}]

# Pass it to attention

# (Initialize)
shapeWk = shapeWq =  [6, 2, 3]# [d^{L+1}, A, d_k^L] 
Wq = tf.range(np.prod(shapeWk), dtype=tf.int64)
Wk = Wq = tf.reshape(Wq, shapeWk)

shapeWv = [6, 2 ,3]# [d^{L+1}, A, d_v^L] 
Wv = tf.reverse(tf.range(np.prod(shapeWv), dtype=tf.int64), axis=[0])
Wv = tf.reshape(Wv, shapeWv)

shapeWo_old = [3, 2, 6] # [A, d_v^L, d^{L+1}]
Wo_old = tf.range(-np.prod(shapeWo_old)/2, +np.prod(shapeWo_old)/2, dtype=tf.int64)
Wo_old = tf.reshape(Wo_old, shapeWo_old)

shapeWo = [3*2, 6] # [d_v^L * A, d^{L+1}]
Wo = tf.range(-np.prod(shapeWo)/2, +np.prod(shapeWo)/2, dtype=tf.int64)
Wo = tf.reshape(Wo, shapeWo)

shapeWo_new = [3, 2, 6] # [d_v^L, A, d^{L+1}]
Wo_new = tf.range(-np.prod(shapeWo_new)/2, +np.prod(shapeWo_new)/2, dtype=tf.int64)
Wo_new = tf.reshape(Wo_new, shapeWo_new)

shapebk = shapebq = [2, 3] # [A, d_k^L]
bk = bq = tf.range(np.prod(shapebq), dtype=tf.int64)
# bk = bq =tf.ones(shapebk, dtype=tf.int64)
bk=bq = tf.reshape(bk, shapebk)

shapebv = [2, 3] # [A, d_v^L]
bv = tf.ones(shapebv, dtype=tf.int64)

shapebk_old = shapebq_old = [2, 1, 3] # [A, d_k^L]
bk_old = bq_old = tf.range(np.prod(shapebq_old), dtype=tf.int64)
# bk = bq =tf.ones(shapebk, dtype=tf.int64)
bk_old=bq_old = tf.reshape(bk_old, shapebk_old)

shapebv_old = [2, 1, 3] # [A, d_v^L]
bv_old = tf.ones(shapebv_old, dtype=tf.int64)


shapebo = [6] # [d^{L+1}]
bo = tf.range(np.prod(shapebo))
bo = tf.reshape(bo, shapebo)

# Start projection of votes to KQV (only if A > 0)
# Original: 
Q_old = tf.einsum('...ki,izj->...zkj', V, Wq) # [batch_size, n^{L+1}, A, n^L, d_k^L]
Q_old_transformed = tf.transpose(Q_old, perm=[0,1,3,4,2])

# if you want output shape to be [batch_size, d^{L+1}, n^L, d_k^L, A] then do '...ki,izj->...kjz' . This is what we will try now.
Q = tf.einsum('...ki,izj->...kjz', V, Wq) # [batch_size, n^{L+1}, n^L, d_k^L, A]
# If you want to have shape [batch_size, n^{L+1}, A, n^L, d_k^L] then do: Q_new_transformed = tf.transpose(Q, perm=[0,1,4,2,3])
tf.math.reduce_all(tf.equal(Q,Q_old_transformed)) # True, so they are equal!

# Do the same for the other matrices (K,V)
K_old = tf.einsum('...ki,izj->...zkj', V, Wk) # [batch_size, n^{L+1}, A, n^L, d_k^L]
V_old = tf.einsum('...ki,izj->...zkj', V, Wv) # [batch_size, n^{L+1}, A, n^L, d_v^L]

K = tf.einsum('...ki,izj->...kjz', V, Wk) # [batch_size, n^{L+1}, n^L, d_k^L, A]
V = tf.einsum('...ki,izj->...kjz', V, Wv) # [batch_size, n^{L+1}, n^L, d_v^L, A]

# Add biases
Qb = Q + tf.transpose(bq)
Kb = K + tf.transpose(bk)
Qv = V + tf.transpose(bv)

Qb_old = Q_old + bq_old
Kb_old = K_old + bk_old
Qv_old = V_old + bv_old

Attention_matrix_old = tf.matmul(Q_old,K_old, transpose_b=True) # [batch_size, n^{L+1}, A, n^L, n^L]
Attention = tf.einsum('...kij,...zij->...jkz',Q,K) # [batch_size, n^{L+1}, A, n^L, n^L]
tf.math.reduce_all(tf.equal(Attention,Attention_matrix_old)) # True
# If we wanted attention heads in last dimension:
Attention_new = tf.einsum('...kij,...zij->...kzj',Q,K) # [batch_size, n^{L+1}, n^L, n^L, A]

Embeddings_old = tf.matmul(Attention_matrix_old, V_old) # [batch_size, n^{L+1}, (A,) n^L, d_v^L] # same as tf.matmul(Attention, V_old)

Projected_embeddings_old = tf.einsum('...kim, mkz->...iz', Embeddings_old, Wo_old) # [batch_size, n^{L+1}, n^L, d^{L+1}] # Wo_new.shape == Wo_old.shape: [d_v^L, A, d^{L+1}]

Embeddings_new = tf.einsum('...ijk, ...jmk ->...imk', Attention_new, V) # [batch_size, n^{L+1}, n^L, d_v^L, A]

Projected_embeddings_new = tf.einsum('...imk, mkz->...iz', Embeddings_new, Wo_new) # [batch_size, n^{L+1}, n^L, d^{L+1}]

tf.math.reduce_all(tf.equal(Projected_embeddings_old,Projected_embeddings_new)) # True

# να δω για μαξ, εφαρμογή relu... softmax, attention scores....
# Let's first see how to sum together (mean) attention heads (easy).
Attention_new_mean_heads = tf.reduce_mean(Attention_new, axis=-1)
# Now let's see how to compute attention scores.
Attention_scores = tf.reduce_sum(Attention_new_mean_heads, axis=-1) # [batch_size, n^{L+1} ,n^L]
smx = tf.keras.layers.Softmax(axis=-2)
SoftSC = smx(tf.cast(Attention_scores,dtype=tf.float64)) # [batch_size, n^{L+1} ,n^L]
winners = tf.reduce_max(SoftSC, axis=-1, keepdims=True)

# Find

# Test
shaper2 = [1,5,4] # [batch_size, n^{L+1} ,n^L]
SoftSC2= tf.math.random.uniform(
    shaper2,
    minval=-10,
    maxval=10,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)
SoftSC2 = smx(SoftSC2)

# Multiplied SoftSC2 to embeddings (same with scaling attention matrix's rows with similarity scores).
Projected_embeddings_new_scaled = (tf.expand_dims(SoftSC2,axis=-1) * tf.cast(Projected_embeddings_new,dtype=tf.float32))
# Below is old code, for the first edition of MyAttention class.

# Find the length of each projected embedding (another way to measure similarity).
Length_of_Projected_embeddings_new =  tf.sqrt(tf.reduce_sum(tf.square(tf.cast(Projected_embeddings_new,dtype=tf.float32)), -1)) # Can also be used with Projected_embeddings_new_scaled

# Now, SoftSC and Length_of_Projected_embeddings_new have the same shape [batch_size, n^{L+1}, n^L] so the procedure could be the same.

# Find winners and winner indices.
winners2SC = tf.reduce_max(SoftSC2, axis=-1, keepdims=True) # [batch_size, n^{L+1}, 1]
winner_indicesSC = tf.argmax(SoftSC2, axis=-1) # [batch_size, n^{L+1}]
winners2_Emb = tf.reduce_max(Length_of_Projected_embeddings_new, axis=-1, keepdims=True) # [batch_size, n^{L+1}, 1]
winner_indices_Emb = tf.argmax(Length_of_Projected_embeddings_new, axis=-1) # [batch_size, n^{L+1}]

# Build a mask and apply it.
mask_Emb = tf.expand_dims(tf.keras.backend.one_hot(indices=winner_indices_Emb, num_classes=4), axis=-1) # [batch_size, n^{L+1}, n^L, 1]
mask_SC2 = tf.expand_dims(tf.keras.backend.one_hot(indices=winner_indicesSC, num_classes=4), axis=-1) # [batch_size, n^{L+1}, n^L, 1]

# Aply mask and sum the zeros (sum n^L)
Digit_Caps_SC2_scaled = (tf.reduce_sum(Projected_embeddings_new_scaled * mask_SC2, axis=-2)) # [batch_size, n^{L+1}]
Digit_Caps_SC2 = (tf.reduce_sum(tf.cast(Projected_embeddings_new, dtype=tf.float32) * mask_SC2, axis=-2))
tf.math.reduce_all(tf.equal(winners2SC * Digit_Caps_SC2,Digit_Caps_SC2_scaled))  # True

# Do the same for the second criterion (Length)
Digit_Caps_Emb = (tf.reduce_sum(tf.cast(Projected_embeddings_new, dtype=tf.float32) * mask_Emb, axis=-2))
Digit_Caps_Emb_scaled = winners2SC * Digit_Caps_Emb

shape = [12,2,6] # (D, A ,D_new)
W = tf.math.random.uniform(
    shape,
    minval=0,
    maxval=None,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)

shape2 = [2, 10, 3,12] # (batch_size, N_L+1, N, D)
V = tf.math.random.uniform(
    shape2,
    minval=0,
    maxval=None,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)


u = tf.matmul(V,W)
u = tf.einsum('...ki,izj->...zkj',V,W)

shape3 = [2,6] # (batch_size, N_L+1, N, D)
b_v = tf.math.random.uniform(
    shape3,
    minval=50,
    maxval=100,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)

shape4 = [6*2,12] # (A * D_new, D_out)
W_o = tf.math.random.uniform(
    shape4,
    minval=-1,
    maxval=1,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)

att = tf.matmul(u,u,transpose_b=True)
new_emb = (att @ u)
new_new_emb = tf.transpose(new_emb, [0,1,3,2,4])
new_new_emb2 = tf.experimental.numpy.swapaxes(new_emb, -2, -3)

new_new_emb = tf.reshape(new_new_emb,list(new_new_emb.shape[:-2]) + [new_new_emb.shape[-2]* new_new_emb.shape[-1]])
res = tf.einsum('...i,iz->...z', new_new_emb, W_o)

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
 
    Methods
    -------
    call(inputs)
        Expects to recieve three inputs as a tuple in the following order: (query, key, value).
        The last two dimensions of the inputs should be [N_L,D_model].
    """
    def __init__(self, N, D_model=16, D_out=16, kernel_initializer='he_normal', A=2, D_k=8, D_v=8, use_biases=False, **kwargs):
        super(MyAttention, self).__init__(**kwargs)
        self.N = N # Number of "words" or capsules
        self.D_model = D_model
        self.D_out = D_out
        self.A = A
        self.D_k = D_k
        self.D_v = D_v
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.biases = use_biases
        
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
        
    # def build(self, inputs_shape):
    #     tf.debugging.assert_equal(inputs_shape[0][-1], inputs_shape[1][-1], message="Query and Key must have same number of features (last dimension).", summarize=None, name=None)
    #     tf.debugging.assert_equal(inputs_shape[1][-1], inputs_shape[2][-1], message="Key and Value must have same number of features (last dimension).", summarize=None, name=None)

    #     d_model = inputs_shape[0][-1]
    #     N_q = inputs_shape[0][-2]
    #     N_k = inputs_shape[1][-2]
    #     N_v = inputs_shape[2][-2]

    #     self.W_q = self.add_weight(shape=[self.A, d_model, self.QKD],initializer=self.kernel_initializer,name='W_q')
    #     self.W_k = self.add_weight(shape=[self.A, d_model, self.QKD],initializer=self.kernel_initializer,name='W_k')
    #     self.W_v = self.add_weight(shape=[self.A, d_model, self.D],initializer=self.kernel_initializer,name='W_q')
    #     # self.b = self.add_weight(shape=[self.N, input_N,1], initializer=tf.zeros_initializer(), name='b')
    #     self.multihead = tf.keras.layers.MultiHeadAttention(self.A, self.QKD, value_dim=16, dropout=0.0, use_bias=True, attention_axes=[2])
    #     self.built = True
    
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
            new_embeddings = tf.experimental.numpy.swapaxes(new_embeddings, -2, -3) # Shape = [Batch_size, N_L+1, N, A, D_v]
            # Concatenate along head axis.  Shape = [Batch_size, N_L+1, N, D_v * A]
            new_embeddings = tf.reshape(new_embeddings, list(new_new_emb.shape[:-2]) + [new_new_emb.shape[-2]* new_new_emb.shape[-1]])
            
            # Project to D_out
            new_embeddings = tf.einsum('...i,iz->...z', new_embeddings, self.N, self.N)
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
            'use_biases': self.biases
        }
        base_config = super(MyAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


inputs = tf.keras.Input(input_shape)

x = tf.keras.layers.Conv2D(256, 9, activation="relu", padding='valid')(inputs)

x = PrimaryCaps(256, 9, 32*6*6, 8, s=2)(x)
digit_caps, c = MyAttention(10,16, A=4, QKD=4)(x)
digit_caps_len = Length(name='length_capsnet_output')(digit_caps)
model = tf.keras.Model(inputs=inputs,outputs=[digit_caps_len], name='Efficient_CapsNet')

# Tensorflow dataset
import tensorflow_datasets as tfds
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(32)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(64)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

class StupidModel(tf.keras.Model):
    def __init__(self, units=128, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.flattener = tf.keras.layers.Flatten(input_shape=(28,28))
        self.hidden1 = tf.keras.layers.Dense(units, activation=activation)
        self.final = tf.keras.layers.Dense(10)
        
    def build(self, input_shape):
        
        print("\n In shpae== ", input_shape, "\n\n")
        self.built = True
        #super().build(input_shape)
    
    def call(self, inputs):
        input, _ = inputs
        res_temp = self.flattener(input)
        res_temp2 = self.hidden1(res_temp)
        result = self.final(res_temp2)
        
        return result

model = StupidModel()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)
import numpy as np
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


model.fit(
    (x_train, x_train),
    y_train,
    epochs=6,
    validation_data=((x_test, x_test), y_test),
)


# x = tf.constant([[1, 1, 1], [1, 1, 1]])

# layer = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=8, attention_axes=[2])
# ##u = tf.random.normal([10,16,8], 0, 1, tf.float32)
# u = tf.keras.Input(shape=[10, 16, 8])
# #c = tf.einsum('...ij,...kj->...i', u, u)[...,None]
# #dd =tf.multiply(u, c)
# #v = tf.linalg.matmul(u,u,transpose_b=True)
# #d = tf.math.reduce_sum(v,axis=2,keepdims=True)
# #print(v.shape)
# #source = tf.keras.Input(shape=[4, 16])
# output_tensor, weights = layer(u, u, return_attention_scores=True)
# print(output_tensor.shape)
# print(weights.shape)
# # Sum along the capsule votes of the prev layer (for each capsule in the last layer).
# res = tf.reduce_sum(output_tensor,axis=-2)
# print(res.shape)