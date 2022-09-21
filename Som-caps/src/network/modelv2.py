'''
This is where the model is defined.
We will use the functional API to define our model.
'''

import tensorflow as tf
from tensorflow.python.eager.context import update_server_def
from network.layers import PrimaryCaps, DigitCaps, Length, Mask
from utils.tools import squash, tanh_vector
import numpy as np


class MyModel(tf.keras.Model):

  def __init__(self, input_shape, digit_vector_size = 16, num_classes=10, generator=False, reduced_votes=False, iterations=1, softmax=False, neighboor_thetas=[1.0], transformation_matrices_for_each_capsule=10):
    super(MyModel, self).__init__()
    #self.digit_caps = self.add_weight(shape=[10, 8], initializer=tf.keras.initializers.RandomUniform(minval=-1.,maxval=1.,seed=None), name='out_caps', trainable=False)
    self.first_layer = tf.keras.layers.Conv2D(256, 9, activation='relu')
    self.p_caps = PrimaryCaps()
    #self.p_caps2 = tf.keras.layers.Conv2D(2, 6, activation='relu')
    self.d_caps = DigitCaps(vector_depth=digit_vector_size, num_out_caps=num_classes , reduced_votes=reduced_votes, transf_mat_for_each_caps= transformation_matrices_for_each_capsule, iter=iterations, softmax=softmax, l_thetas=neighboor_thetas)
    self._input_shape = input_shape
    self.generator = generator
    if generator:
      self.mask = Mask(trainable=self.trainable)
      self.g1 = tf.keras.layers.Dense(512, activation='relu')
      self.g2 = tf.keras.layers.Dense(1024, activation='relu')
      self.g3 = tf.keras.layers.Dense(tf.math.reduce_prod(input_shape), activation='sigmoid')
      self.g4_reshape = tf.keras.layers.Reshape(target_shape=input_shape, name='out_generator')

  def call(self, inputs):
    x = self.first_layer(inputs)
    x = self.p_caps(x)
    #x = self.p_caps2(x)
    rs, d = self.d_caps(x)
    # For more efficiency: define digit_caps inside Digit_Caps layer and...
    #self.add_update
    # or self.digit_caps.assign_add(new_value_to_add) or .assign(new_value)
    if self.generator:
      img = self.mask(d, rs)
      img = self.g1(img)
      img = self.g2(img)
      img = self.g3(img)
      img = self.g4_reshape(img)
      recon_loss = tf.reduce_mean(tf.square(img-inputs))
      self.add_loss(recon_loss)
    return rs

