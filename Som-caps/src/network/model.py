'''
This is where the model is defined.
We will use the functional API to define our model.
'''

import numpy as np
import tensorflow as tf
from tensorflow.python.eager.context import update_server_def
from utils.tools import squash, tanh_vector

from network.layers import DigitCaps, Length, Mask, PrimaryCaps, Mask2

# tf.compat.v1.disable_eager_execution()
tf.config.run_functions_eagerly(True)

class MyModel(tf.keras.Model):

  def __init__(self, digit_vector_size = 8, num_classes=10, generator=False, reduced_votes=True, small=False, iterations=1, softmax=False, neighboor_thetas=[1.0], transformation_matrices_for_each_capsule=10):
    super(MyModel, self).__init__()
  
    self.first_layer = tf.keras.layers.Conv2D(256, 9, activation='relu')
    self.p_caps = PrimaryCaps()
   
    self.small = small
    if small:
      self.p_caps2 = tf.keras.layers.Conv2D(2, 6, activation='relu')
    self.digit_vector_size = digit_vector_size
    self.num_classes = num_classes
    self.reduced_votes = reduced_votes
    self.small = small
    self.iterations = iterations
    self.softmax = softmax
    self.neighboor_thetas = neighboor_thetas
    self.transformation_matrices_for_each_capsule = transformation_matrices_for_each_capsule
    self.generator = generator
    self.d_caps= DigitCaps(vector_depth=self.digit_vector_size, num_out_caps=self.num_classes , reduced_votes=self.reduced_votes, transf_mat_for_each_caps= self.transformation_matrices_for_each_capsule, iter=self.iterations, softmax=self.softmax, l_thetas=self.neighboor_thetas)


  def call(self, inputs):
    x = self.first_layer(inputs)
    x = self.p_caps(x)
    if self.small:
      x = self.p_caps2(x)

    x , caps=  self.d_caps(x)

    # For more efficiency: define digit_caps inside Digit_Caps layer and...
    #self.add_update
    # or self.digit_caps.assign_add(new_value_to_add) or .assign(new_value)

    return x, caps
  
  def compute_output_shape(self, batch_input_shape):
    return tf.TensorShape((batch_input_shape[0],self.num_classes), (batch_input_shape[0], self.num_classes, self.digit_vector_size))
  
  def get_config(self):
    base_config = super().get_config()
    return {**base_config,
            "digit_vector_size":self.digit_vector_size,
              "num_classes":self.num_classes,
              "generator": self.generator,
              "reduced_votes": self.reduced_votes,
              "small": self.small,
              "iterations": self.iterations,
              "softmax": self.softmax,
              "neighboor_thetas": self.neighboor_thetas,
              "transformation_matrices_for_each_capsule": self.transformation_matrices_for_each_capsule}

    # self._input_shape = input_shape
    # if generator:
    #   self.mask = Mask(trainable=self.trainable)
    #   self.g1 = tf.keras.layers.Dense(512, activation='relu')
    #   self.g2 = tf.keras.layers.Dense(1024, activation='relu')
    #   self.g3 = tf.keras.layers.Dense(tf.math.reduce_prod(input_shape[1:]), activation='sigmoid')
    #   self.g4_reshape = tf.keras.layers.Reshape(tf.concat((-tf.ones(1, dtype=tf.int32),tf.convert_to_tensor(input_shape[1:])), axis=0), name='out_generator')
    # if self.generator:
      # img = self.mask(self.digit_caps, rs)
      # img = self.g1(img)
      # img = self.g2(img)
      # img = self.g3(img)
      # img = self.g4_reshape(img)
      # recon_loss = tf.reduce_mean(tf.square(img-inputs))
      # self.add_loss(recon_loss)

def som_capsnet_graph(input_shape, digit_vector_size = 8, num_classes=10, reduced_votes=True, small=False, iterations=1, softmax=False, neighboor_thetas=[1.0], transformation_matrices_for_each_capsule=10, gen=False, lr_som =1.0, radical=False, normalize_d_in_loop=False,
                      normalize_digit_caps=False, normalize_votes=False, norm_type=0, take_into_account_similarity=False, take_into_account_winner_ratios=False, tanh_like=False):
  inputs = tf.keras.Input(input_shape)
  #print('SHAPEEE: ', input_shape)
  out1 = tf.keras.layers.Conv2D(256, 9, activation='relu', name='conv_layer')(inputs)
  out2 = PrimaryCaps()(out1)
  if small:
    out2 = tf.keras.layers.Conv2D(2, 6, activation='relu', name='conv_mini')(out2)
  out3, caps = DigitCaps(vector_depth=digit_vector_size, num_out_caps=num_classes , reduced_votes=reduced_votes, transf_mat_for_each_caps= transformation_matrices_for_each_capsule, iter=iterations, softmax=softmax, l_thetas=neighboor_thetas, lr_som=lr_som, radical=radical,
                         normalize_d_in_loop=normalize_d_in_loop, normalize_digit_caps=normalize_digit_caps, normalize_votes=normalize_votes, norm_type=norm_type, take_into_account_similarity=take_into_account_similarity, take_into_account_winner_ratios=take_into_account_winner_ratios, tanh_like=tanh_like)(out2)
  
  if gen:
    out = [out3, caps]
  else:
    out = out3
  return tf.keras.Model(inputs=inputs,outputs=out, name='Som_CapsNet_partial')

def create_generator(input_shape, out_shape, deconv=False, cifar=False):
  inputs = tf.keras.Input(input_shape)
  if not deconv:
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(out_shape[0]*out_shape[1]*out_shape[2], activation='sigmoid')(x)
    output = tf.keras.layers.Reshape(out_shape, name='out_generator')(x)
  else:
    if not cifar:
      x = tf.keras.layers.Dense(6**2 * 10, activation='relu', kernel_initializer='he_normal')(inputs)
      x = tf.keras.layers.Reshape(target_shape=[6,6,10])(x)
      x = tf.keras.layers.Conv2DTranspose(32,9,2)(x)
      x = tf.keras.layers.Conv2DTranspose(64,9,1)(x)
      output = tf.keras.layers.Conv2DTranspose(1,2,1)(x)
    else:
      x = tf.keras.layers.Dense(6**2 * 10, activation='relu', kernel_initializer='he_normal')(inputs)
      x = tf.keras.layers.Reshape(target_shape=[6,6,10])(x)
      x = tf.keras.layers.Conv2DTranspose(32,9,2)(x)
      x = tf.keras.layers.Conv2DTranspose(64,9,1)(x)
      output = tf.keras.layers.Conv2DTranspose(3,2,1)(x)
    
  return tf.keras.Model(inputs=inputs,outputs=output, name='Generator')


def build_graph(input_shape, batch_size=16, digit_vector_size = 8, num_classes=10, reduced_votes=True, small=False, iterations=1, softmax=False, neighboor_thetas=[1.0], transformation_matrices_for_each_capsule=10, gen=False, training=True, deco=False, cifar=False, lr_som =1.0, radical=False, normalize_d_in_loop=False,
                      normalize_digit_caps=False, normalize_votes=False, norm_type=0, take_into_account_similarity=False, take_into_account_winner_ratios=False, tanh_like=False):
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.layers.Input(shape=(1,))

    model =som_capsnet_graph(input_shape, digit_vector_size = digit_vector_size, num_classes=num_classes, reduced_votes=reduced_votes, small=small, iterations=iterations, softmax=softmax, neighboor_thetas=neighboor_thetas, transformation_matrices_for_each_capsule=transformation_matrices_for_each_capsule, gen=gen, lr_som=lr_som, radical=radical,
                         normalize_d_in_loop=normalize_d_in_loop, normalize_digit_caps=normalize_digit_caps, normalize_votes=normalize_votes, norm_type=norm_type, take_into_account_similarity=take_into_account_similarity, take_into_account_winner_ratios=take_into_account_winner_ratios, tanh_like=tanh_like)
    if gen:
      similarities, digit_caps = model(inputs)
    else:
      similarities = model(inputs)
    
    if gen:
      if training:
        masked_by_y = Mask(num_classes, batch_size=batch_size)([digit_caps, y_true], training)
      else:
        masked = Mask(num_classes, batch_size=batch_size)([digit_caps, similarities], training)
      
      generator = create_generator(num_classes*digit_vector_size, input_shape, deconv=deco, cifar=cifar)
      if training:
        x_gen = generator(masked_by_y)
      else:
        x_gen = generator(masked)

    if training and gen:   
      return tf.keras.models.Model([inputs, y_true], [similarities, x_gen], name='SOM_capsnet_with_Generator_trained_with_labels')
    elif not gen:
      return tf.keras.models.Model(inputs, similarities, name='SOM_Capsnet_without_Generator')
    else: # test with generator
      return tf.keras.models.Model(inputs, [similarities, x_gen], name='SOM_capsnet_with_Generator_without_labels')

    # self._input_shape = input_shape
    # if generator:
    #   self.mask = Mask(trainable=self.trainable)
    #   self.g1 = tf.keras.layers.Dense(512, activation='relu')
    #   self.g2 = tf.keras.layers.Dense(1024, activation='relu')
    #   self.g3 = tf.keras.layers.Dense(tf.math.reduce_prod(input_shape[1:]), activation='sigmoid')
    #   self.g4_reshape = tf.keras.layers.Reshape(tf.concat((-tf.ones(1, dtype=tf.int32),tf.convert_to_tensor(input_shape[1:])), axis=0), name='out_generator')
    # if self.generator:
      # img = self.mask(self.digit_caps, rs)
      # img = self.g1(img)
      # img = self.g2(img)
      # img = self.g3(img)
      # img = self.g4_reshape(img)
      # recon_loss = tf.reduce_mean(tf.square(img-inputs))
      # self.add_loss(recon_loss)

def SMALLNORB_som_capsnet_graph(input_shape, digit_vector_size = 8, num_classes=5, reduced_votes=True, small=False, iterations=1, softmax=False, neighboor_thetas=[1.0], transformation_matrices_for_each_capsule=5, gen=False, lr_som =1.0, radical=False, normalize_d_in_loop=False,
                      normalize_digit_caps=False, normalize_votes=False, norm_type=0, take_into_account_similarity=False, take_into_account_winner_ratios=False, tanh_like=False):
  inputs = tf.keras.Input(input_shape)
  #print('SHAPEEE: ', input_shape) # should be [48, 48, 2] because we have 2 images for each sample.
  out1 = tf.keras.layers.Conv2D(256, 9, activation='relu', name='conv_layer')(inputs)
  out2 = PrimaryCaps()(out1)
  if small:
    out2 = tf.keras.layers.Conv2D(2, 6, activation='relu', name='conv_mini')(out2)
  out3, caps = DigitCaps(vector_depth=digit_vector_size, num_out_caps=num_classes , reduced_votes=reduced_votes, transf_mat_for_each_caps= transformation_matrices_for_each_capsule, iter=iterations, softmax=softmax, l_thetas=neighboor_thetas, lr_som=lr_som, radical=radical,
                         normalize_d_in_loop=normalize_d_in_loop, normalize_digit_caps=normalize_digit_caps, normalize_votes=normalize_votes, norm_type=norm_type, take_into_account_similarity=take_into_account_similarity, take_into_account_winner_ratios=take_into_account_winner_ratios, tanh_like=tanh_like)(out2)
  
  if gen:
    out = [out3, caps]
  else:
    out = out3
  return tf.keras.Model(inputs=inputs,outputs=out, name='Som_CapsNet_partial')

def SMALLNORB_create_generator(input_shape, out_shape, deconv=False):
  inputs = tf.keras.Input(input_shape)
  if not deconv:
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(out_shape[0]*out_shape[1]*out_shape[2], activation='sigmoid')(x)
    output = tf.keras.layers.Reshape(out_shape, name='out_generator')(x)
  else:
    # In this reconstruction, upsampling is used along with convolution by the original efficent net algorithm. 
    # This is essentially the same as deconvolution.
    # So, we moved the code of "not deconv" under the "else" statement and wrote a simple reconstructioon under the "if" statement.
    x = tf.keras.layers.Dense(64)(inputs)
    x = tf.keras.layers.Reshape(target_shape=(8,8,1))(x)
    x = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="valid", activation=tf.nn.leaky_relu)(x)
    x = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="valid", activation=tf.nn.leaky_relu)(x)
    x = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='bilinear')(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="valid", activation=tf.nn.leaky_relu)(x)
    output = tf.keras.layers.Conv2D(filters=2, kernel_size=(3,3), padding="valid", activation=tf.nn.sigmoid)(x)     
    
  return tf.keras.Model(inputs=inputs,outputs=output, name='Generator')


def SMALLNORB_build_graph(input_shape, batch_size=16, digit_vector_size = 8, num_classes=5, reduced_votes=True, small=False, iterations=1, softmax=False, neighboor_thetas=[1.0], transformation_matrices_for_each_capsule=5, gen=False, training=True, deco=False, cifar=False, lr_som =1.0, radical=False, normalize_d_in_loop=False,
                      normalize_digit_caps=False, normalize_votes=False, norm_type=0, take_into_account_similarity=False, take_into_account_winner_ratios=False, tanh_like=False):
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.layers.Input(shape=(5,)) # Now we use one hot vector.

    model =SMALLNORB_som_capsnet_graph(input_shape, digit_vector_size = digit_vector_size, num_classes=num_classes, reduced_votes=reduced_votes, small=small, iterations=iterations, softmax=softmax, neighboor_thetas=neighboor_thetas, transformation_matrices_for_each_capsule=transformation_matrices_for_each_capsule, gen=gen, lr_som=lr_som, radical=radical,
                         normalize_d_in_loop=normalize_d_in_loop, normalize_digit_caps=normalize_digit_caps, normalize_votes=normalize_votes, norm_type=norm_type, take_into_account_similarity=take_into_account_similarity, take_into_account_winner_ratios=take_into_account_winner_ratios, tanh_like=tanh_like)
    if gen:
      similarities, digit_caps = model(inputs)
    else:
      similarities = model(inputs)
    
    if gen:
      if training:
        masked_by_y = Mask2(num_classes, batch_size=batch_size)([digit_caps, y_true], training=training)
      else:
        masked = Mask2(num_classes, batch_size=batch_size)([digit_caps, similarities], training=training)
      
      generator = SMALLNORB_create_generator(num_classes*digit_vector_size, input_shape)
      if training:
        x_gen = generator(masked_by_y)
      else:
        x_gen = generator(masked)

    if training and gen:   
      return tf.keras.models.Model([inputs, y_true], [similarities, x_gen], name='SOM_capsnet_with_Generator_trained_with_labels')
    elif not gen:
      return tf.keras.models.Model(inputs, similarities, name='SOM_Capsnet_without_Generator')
    else: # test with generator
      return tf.keras.models.Model(inputs, [similarities, x_gen], name='SOM_capsnet_with_Generator_without_labels')


def MULTIMNIST_som_capsnet_graph(input_shape, digit_vector_size = 8, num_classes=10, reduced_votes=True, small=False, iterations=1, softmax=False, neighboor_thetas=[1.0], transformation_matrices_for_each_capsule=10, gen=False, lr_som =1.0, radical=False, normalize_d_in_loop=False,
                      normalize_digit_caps=False, normalize_votes=False, norm_type=0, take_into_account_similarity=False, take_into_account_winner_ratios=False, tanh_like=False):
  inputs = tf.keras.Input(input_shape)
  #print('SHAPEEE Hi!: ', input_shape)
  out1 = tf.keras.layers.Conv2D(256, 9, activation='relu', name='conv_layer')(inputs)
  out2 = PrimaryCaps()(out1)
  if small:
    out2 = tf.keras.layers.Conv2D(2, 6, activation='relu', name='conv_mini')(out2)
  out3, caps = DigitCaps(vector_depth=digit_vector_size, num_out_caps=num_classes , reduced_votes=reduced_votes, transf_mat_for_each_caps= transformation_matrices_for_each_capsule, iter=iterations, softmax=softmax, l_thetas=neighboor_thetas, lr_som=lr_som, radical=radical,
                         normalize_d_in_loop=normalize_d_in_loop, normalize_digit_caps=normalize_digit_caps, normalize_votes=normalize_votes, norm_type=norm_type, take_into_account_similarity=take_into_account_similarity, take_into_account_winner_ratios=take_into_account_winner_ratios, tanh_like=tanh_like)(out2)
  
  if gen:
    out = [out3, caps]
  else:
    out = out3
  return tf.keras.Model(inputs=inputs,outputs=out, name='Som_CapsNet_partial')

def MULTIMNIST_create_generator(input_shape, out_shape, deconv=False):
  inputs = tf.keras.Input(input_shape)
  if not deconv:
    x = tf.keras.layers.Dense(512, activation='relu')(inputs)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dense(out_shape[0]*out_shape[1]*out_shape[2], activation='sigmoid')(x)
    output = tf.keras.layers.Reshape(out_shape, name='out_generator')(x)
  else:
    x = tf.keras.layers.Dense(10**2 * 10, activation='relu', kernel_initializer='he_normal')(inputs)
    x = tf.keras.layers.Reshape(target_shape=[10,10,10])(x)
    x = tf.keras.layers.Conv2DTranspose(32,9,2)(x)
    x = tf.keras.layers.Conv2DTranspose(64,9,1)(x)
    output = tf.keras.layers.Conv2DTranspose(1,2,1)(x)
    
  return tf.keras.Model(inputs=inputs,outputs=output, name='Generator')

def MULTIMNIST_build_graph(input_shape, batch_size=16, digit_vector_size = 8, num_classes=10, reduced_votes=True, small=False, iterations=1, softmax=False, neighboor_thetas=[1.0], transformation_matrices_for_each_capsule=10, gen=False, training=True, deco=False, cifar=False, lr_som =1.0, radical=False, normalize_d_in_loop=False,
                      normalize_digit_caps=False, normalize_votes=False, norm_type=0, take_into_account_similarity=False, take_into_account_winner_ratios=False, tanh_like=False):
    inputs = tf.keras.Input(input_shape)
    y_true1 = tf.keras.layers.Input(shape=(10,)) # We now have two outputs, for the two digits.
    y_true2 = tf.keras.layers.Input(shape=(10,)) # Also, unlike mnist dataset, here we have one hot vector encodings of the true label (y).

    model =MULTIMNIST_som_capsnet_graph(input_shape, digit_vector_size = digit_vector_size, num_classes=num_classes, reduced_votes=reduced_votes, small=small, iterations=iterations, softmax=softmax, neighboor_thetas=neighboor_thetas, transformation_matrices_for_each_capsule=transformation_matrices_for_each_capsule, gen=gen, lr_som=lr_som, radical=radical,
                         normalize_d_in_loop=normalize_d_in_loop, normalize_digit_caps=normalize_digit_caps, normalize_votes=normalize_votes, norm_type=norm_type, take_into_account_similarity=take_into_account_similarity, take_into_account_winner_ratios=take_into_account_winner_ratios, tanh_like=tanh_like)
    if gen:
      similarities, digit_caps = model(inputs)
    else:
      similarities = model(inputs)
    
    if gen:
      if training:
        masked_by_y1, masked_by_y2 = Mask2(num_classes, batch_size=batch_size)([digit_caps, y_true1, y_true2], double_mask=True, training=training)
      else:
        masked1, masked2 = Mask2(num_classes, batch_size=batch_size)([digit_caps, similarities], double_mask=True, training=training)
      
      generator = MULTIMNIST_create_generator(num_classes*digit_vector_size, input_shape)
      if training:
        x_gen1, X_gen2 = generator(masked_by_y1), generator(masked_by_y2)
      else:
        x_gen1, X_gen2 = generator(masked1), generator(masked2)

    if training and gen:   
      return tf.keras.models.Model([inputs, y_true1, y_true2], [similarities, x_gen1, X_gen2], name='SOM_capsnet_with_Generator_trained_with_labels')
    elif not gen:
      return tf.keras.models.Model(inputs, similarities, name='SOM_Capsnet_without_Generator')
    else: # test with generator
      return tf.keras.models.Model(inputs, [similarities, x_gen1, X_gen2], name='SOM_capsnet_with_Generator_without_labels')

