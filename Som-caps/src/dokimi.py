# import tensorflow as tf

# W = tf.random.uniform(shape=[1152,8, 160])
# x = tf.random.uniform(shape=[32, 1152, 8])

# digit_caps = tf.random.uniform(shape=[10,16])

# u = tf.einsum('...ji,jik->...jk', x, W)

# u = tf.reshape(u,(-1, 6*6*32*10, 16)) # Or just u = tf.reshape(u,(-1, 6*6*32, 10, 16)) but this is different.
# b = tf.constant([1,1,10,1], tf.int32)

# # Normalize vectors
# n = tf.norm(u, axis=-1,keepdims=True)
# u = tf.multiply(n**2/(1+n**2)/(n + tf.keras.backend.epsilon()), u)
# u = tf.multiply(tf.math.divide(tf.math.tanh(n),n), u)

# u = tf.expand_dims(u, -2)
# u = tf.tile(u,b)
# #u = tf.ones(shape=[2,2,10,16])
# # Want to subtract u - digit_caps 
# # Need to follow general substruction rules/ broadcasting
# #digit_caps = tf.range(1,11,dtype=tf.float32)
# #digit_caps = tf.expand_dims(digit_caps,1)
# #b = tf.constant([1,16], tf.int32)
# #digit_caps = tf.tile(digit_caps,b)

# differences = u-digit_caps # Or test with u only
# norms = tf.norm(u-digit_caps, ord='euclidean', axis=-1, keepdims=False, name=None)
# similarities = tf.reduce_sum(tf.math.multiply(u,digit_caps),axis=-1)

# # If we want neighborhood, we repeat the two lines below and concatenate them (along axis=1). Remember to multiply new norms with the neghbour distance. Also remember to not pick the same winners.
# winners = tf.math.argmax(similarities, axis=-1, output_type=tf.dtypes.int64, name=None)
# win_norms = tf.gather(similarities, winners, validate_indices=None, axis=-1, batch_dims=2, name=None)

# win_vectors = tf.gather(u, winners, axis=2, batch_dims=2, name=None) # Same with u before expansion if different weights is not used.
# win_differences = tf.gather(differences, winners, axis=2, batch_dims=2, name=None) # Pick the differences between u and digit_caps vectors which have the largest similarity. Those differences will be used for updating digit_caps vectors.
# win_norms_with_vectors = tf.math.multiply(tf.expand_dims(win_norms,axis=-1) , win_differences, name=None) # Scale the differences according to their similarity. If similarity is large, the update will be proportionaly large. In original SOM, similarity is only used to pick the winner and not used anywhere else.
# lr = tf.constant(1.0,dtype=tf.float32)
# updates = win_norms_with_vectors * lr # Multiply with learning rate
# updates = tf.reshape(updates,shape=[-1,*updates.shape[2:]])

# winners = tf.reshape(winners,shape=[-1,*winners.shape[2:]])

# y, idx, count = tf.unique_with_counts(winners, out_idx=tf.dtypes.int32, name=None)
# win_count = tf.scatter_nd(tf.expand_dims(y,axis=-1), count, shape=[10], name=None)
# win_weights = win_count/idx.shape[0]
# # only_win_norms = tf.zeros_like(norms)
# # Now we can divide by 11520 * 32 to get the mean if we want.
# digits_update = tf.scatter_nd(tf.expand_dims(winners, axis=-1), updates, shape=[10,16], name=None)/idx.shape[0]

# # Then we can multiply with win_count 
# # digits_update = digits_update * tf.expand_dims(tf.cast(win_weights, dtype=tf.float32),axis=-1)
# # or...  we can use it as a balance between old and new
# # And lastly, add to the digit caps (in any case).
# digit_caps = digit_caps + digits_update

# output = tf.math.softmax(tf.reduce_mean(similarities, axis=-2), axis=-1) # The output tensor. If multiple iterations, just do without softmax and add means at each iteration or better, consider only the last similarities. Then do softmax outside the loop.


# Alternative way
import tensorflow as tf

W = tf.random.uniform(shape=[1152,8, 160])
x = tf.random.uniform(shape=[32, 1152, 8])

digit_caps = tf.random.uniform(shape=[10,16])

u = tf.einsum('...ji,jik->...jk', x, W)

u = tf.reshape(u,(-1, 6*6*32*10, 16)) # Or just u = tf.reshape(u,(-1, 6*6*32, 10, 16)) but this is different.
b = tf.constant([1,1,10,1], tf.int32)

# Normalize vectors
n = tf.norm(u, axis=-1,keepdims=True)
# u = tf.multiply(n**2/(1+n**2)/(n + tf.keras.backend.epsilon()), u) # squash
u = tf.multiply(tf.math.divide(tf.math.tanh(n),n), u) # tanh for vectors

# To compute similarity, we need to tile the vote tensor 10 times. Of course we can use W to map each vote into 10 separate vectors (like we do now) but not merge them to the vote dimension.
u = tf.expand_dims(u, -2)
u = tf.tile(u,b)

sparse_updates_all = tf.zeros_like(u) # u after tiling

# Used if we enable weights acording to the number of wins.
win_count_all = tf.zeros(shape=[10], dtype=tf.int64)

# Get the difference used in update.
differences = u-digit_caps # Try using only the u

# Create a masked digit caps in case we want multiple iterations to create a sence of neighborhood. Using masked_digit_caps we won't pick the same winners.
masked_digit_caps = tf.ones_like(u)
# Compute final similarities here or at the end, after updating digit_caps?
# similarities_final = tf.reduce_sum(tf.math.multiply(u,digit_caps),axis=-1)

# List of neighboorhood
l_thetas =  [1, 0.2, 0.1]
l_thetas = tf.constant(l_thetas, dtype=tf.float32)
for theta in l_thetas:
    #### IN LOOP ####
    # Get similarities
    similarities = tf.reduce_sum(tf.math.multiply(u*masked_digit_caps,digit_caps),axis=-1)

    # Get for each vote, which digit_caps is the winner.
    winners = tf.math.argmax(similarities, axis=-1, output_type=tf.dtypes.int64, name=None)

    # If you want, you can compute the winner ratio
    # winners_f = tf.reshape(winners,shape=[-1,*winners.shape[2:]])
    # y, idx, count = tf.unique_with_counts(winners_f, out_idx=tf.dtypes.int32, name=None)
    # win_count = tf.scatter_nd(tf.expand_dims(y,axis=-1), count, shape=[10], name=None)
    # win_count_all += win_count

    # Alternatively, we could softmax the similarities instead of having hard winners. If you do so, choose a bigger learning rate. Probably, then you should not use iterations (just one).

    # Convert winners to one hot vectors. Then multiply them with similarities etc. to keep win similarities. Try to multiply it with difference vectors to form the update vector.
    winners_mask = tf.keras.backend.one_hot(indices=winners, num_classes=10)

    lr = tf.constant(1.0,dtype=tf.float32)
    # If you want to take into account the similarity of the pairs in the update, uncomment the two lines below and comment the third one.
    #sparse_similarities = similarities * winners_mask
    #sparse_updates = differences * tf.expand_dims(sparse_similarities,axis=-1) * lr
    sparse_updates = differences * tf.expand_dims(winners_mask,axis=-1) * lr * theta

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


digit_caps = digit_caps + updates

# Take the norm of digit_caps (and repeat).
d_norm = tf.norm(digit_caps, axis=-1,keepdims=True)
digit_caps = digit_caps / d_norm

# Re-compute similarities with the updated vector.
similarities_final = tf.reduce_sum(tf.math.multiply(u,digit_caps),axis=-1)
# Compute the output (if we have multiple iterations, this will be out of the loop)
output = tf.math.softmax(tf.reduce_mean(similarities_final, axis=-2), axis=-1)


#### Multiplication simple
xa = tf.ones(shape=[2,2*2*3,8])
wa = tf.ones(shape=[2*2*3,8,16])
tf.einsum('...ji,jik->...jk', xa, wa) 

b = tf.constant([2,1,1,1], tf.int32)
wa = tf.expand_dims(wa, 0)
wa = tf.tile(wa,b)