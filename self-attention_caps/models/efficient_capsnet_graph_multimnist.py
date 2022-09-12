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
from utils.layers import PrimaryCaps, FCCaps, Length, Mask, FCCapsMultihead


def efficient_capsnet_graph(input_shape, multihead=False, original_convs=False):
    """
    Efficient-CapsNet graph architecture.
    
    Parameters
    ----------   
    input_shape: list
        network input shape
    """
    inputs = tf.keras.Input(input_shape)
    
    if not original_convs:
        x = tf.keras.layers.Conv2D(32,5,activation="relu", padding='valid', kernel_initializer='he_normal')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64,3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64,3,2, activation='relu', padding='valid', kernel_initializer='he_normal')(x)   
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(128,3,2, activation='relu', padding='valid', kernel_initializer='he_normal')(x)   
        x = tf.keras.layers.BatchNormalization()(x)
        x = PrimaryCaps(128, 5, 16, 8, 2)(x)

    else:
        x = tf.keras.layers.Conv2D(128, 9, activation="relu", padding='valid')(inputs)
        x = PrimaryCaps(128, 9, 16*10*10, 8, s=2)(x)
    
    if multihead:
        digit_caps, c = FCCapsMultihead(10,16, A=4, VD=4)(x)

    else:
        # Original: no multihead.
        digit_caps, c = FCCaps(10,16)(x)
    
    digit_caps_len = Length(name='length_capsnet_output')(digit_caps)

    return tf.keras.Model(inputs=inputs,outputs=[digit_caps, digit_caps_len], name='Efficient_CapsNet')


def generator_graph(input_shape, deconv):
    """
    Generator graph architecture.
    
    Parameters
    ----------   
    input_shape: list
        network input shape
    """
    inputs = tf.keras.Input(16*10)
    if not deconv:
        x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(inputs)
        x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid', kernel_initializer='glorot_normal')(x)
        x = tf.keras.layers.Reshape(target_shape=input_shape, name='out_generator')(x)
    else:
        x = tf.keras.layers.Dense(10**2 * 10, activation='relu', kernel_initializer='he_normal')(inputs)
        x = tf.keras.layers.Reshape(target_shape=[10,10,10])(x)
        x = tf.keras.layers.Conv2DTranspose(32,9,2)(x)
        x = tf.keras.layers.Conv2DTranspose(64,9,1)(x)
        x = tf.keras.layers.Conv2DTranspose(1,2,1)(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name='Generator')


def build_graph(input_shape, mode, verbose, deconv, multihead, original):
    """
    Efficient-CapsNet graph architecture with reconstruction regularizer. The network can be initialize with different modalities.
    Parameters
    ----------   
    input_shape: list
        network input shape
    mode: str
        working mode ('train', 'test' & 'play')
    verbose: bool
    """
    inputs = tf.keras.Input(input_shape)
    y_true1 = tf.keras.layers.Input(shape=(10,))
    y_true2 = tf.keras.layers.Input(shape=(10,))

    efficient_capsnet = efficient_capsnet_graph(input_shape, multihead, original)

    if verbose:
        efficient_capsnet.summary()
        print("\n\n")
    
    digit_caps, digit_caps_len = efficient_capsnet(inputs)
    
    masked_by_y1,masked_by_y2 = Mask()([digit_caps, y_true1, y_true2],double_mask=True)  
    masked1,masked2 = Mask()(digit_caps,double_mask=True)
    
    generator = generator_graph(input_shape, deconv)

    if verbose:
        generator.summary()
        print("\n\n")

    x_gen_train1,x_gen_train2 = generator(masked_by_y1),generator(masked_by_y2)
    x_gen_eval1,x_gen_eval2 = generator(masked1),generator(masked2)

    if mode == 'train':   
        return tf.keras.models.Model([inputs, y_true1,y_true2], [digit_caps_len, x_gen_train1,x_gen_train2], name='Efficinet_CapsNet_Generator')
    elif mode == 'test':
        return tf.keras.models.Model(inputs, [digit_caps_len, x_gen_eval1,x_gen_eval2], name='Efficinet_CapsNet_Generator')
    else:
        raise RuntimeError('mode not recognized')