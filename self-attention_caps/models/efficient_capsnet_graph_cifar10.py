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


def efficient_capsnet_graph(input_shape, multihead=False, original_convs=False, num_heads=4, Algorithm='RooMAV', scale_the_embedding=True, use_agreement_criterion=True):
    """
    Efficient-CapsNet graph architecture.

    Parameters
    ----------   
    input_shape: list
        network input shape
    """
    if Algorithm == 'RooMAV':
        Algo1 = True
    elif Algorithm == 'RoWSS':
        Algo1 = False
    elif (Algorithm is None) and ((num_heads <=0)or(multihead==False)):
        pass
    else:
        raise ValueError("Invalid argument in efficient_capsnet_graph. Set Algorithm appropriately")
    
    inputs = tf.keras.Input(input_shape)

    if not original_convs:
        # Not the original number of primary caps. With depthwise convolution. One capsule for each filter.
        # Capsules here are not local.
        x = tf.keras.layers.Conv2D(32,5,activation="relu", padding='valid', kernel_initializer='he_normal')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64,3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64,3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)   
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(256,3,2, activation='relu', padding='valid', kernel_initializer='he_normal')(x)   
        x = tf.keras.layers.BatchNormalization()(x)
        x = PrimaryCaps(256, 7, 32, 8)(x)
    else:
        # Like Dynamic Routing
        x = tf.keras.layers.Conv2D(256, 9, activation='relu', padding='valid')(inputs)
        x = PrimaryCaps(64*8, 9, 64*4*4, 8, s=2)(x)

    if multihead:
        # With multihead: You may cange the number of heads through A parameter.
        digit_caps , c= FCCapsMultihead(11,16, A=num_heads, QKD=int(16/num_heads), D_v=int(16/num_heads), Alg1=Algo1, scaled_emb=scale_the_embedding, agreement_scores=use_agreement_criterion)(x)
    else:
        # no multihead.
        digit_caps, c = FCCaps(11,16)(x)
    
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
    inputs = tf.keras.Input(16*11)
    if not deconv:
        x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal')(inputs)
        x = tf.keras.layers.Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid', kernel_initializer='glorot_normal')(x)
        x = tf.keras.layers.Reshape(target_shape=input_shape, name='out_generator')(x)
    else:
        x = tf.keras.layers.Dense(6**2 * 10, activation='relu', kernel_initializer='he_normal')(inputs)
        x = tf.keras.layers.Reshape(target_shape=[6,6,10])(x)
        x = tf.keras.layers.Conv2DTranspose(32,9,2)(x)
        x = tf.keras.layers.Conv2DTranspose(64,5,1)(x)
        x = tf.keras.layers.Conv2DTranspose(3,2,1)(x)

    return tf.keras.Model(inputs=inputs, outputs=x, name='Generator')


def build_graph(input_shape, mode, verbose, deconv, multihead, original, num_heads, algorithm, scale_the_embedding, use_agreement_criterion, no_reconstruct):
    """
    Efficient-CapsNet graph architecture with reconstruction regularizer. The network can be initialized with different modalities.

    Parameters
    ----------
    input_shape: list
        network input shape
    mode: str
        working mode ('train', 'test' & 'play')
    verbose: bool
    """
    inputs = tf.keras.Input(input_shape)
    y_true = tf.keras.layers.Input(shape=(11,))
    noise = tf.keras.layers.Input(shape=(11, 16))

    efficient_capsnet = efficient_capsnet_graph(input_shape, multihead, original, num_heads, algorithm, scale_the_embedding, use_agreement_criterion)

    if verbose:
        efficient_capsnet.summary()
        print("\n\n")
    
    digit_caps, digit_caps_len = efficient_capsnet(inputs)
    noised_digitcaps = tf.keras.layers.Add()([digit_caps, noise]) # only if mode is play
    
    masked_by_y = Mask()([digit_caps, y_true])
    masked = Mask()(digit_caps)
    masked_noised_y = Mask()([noised_digitcaps, y_true])
    
    generator = generator_graph(input_shape, deconv)
    
    if not no_reconstruct:
        if verbose:
            generator.summary()
            print("\n\n")

    x_gen_train = generator(masked_by_y)
    x_gen_eval = generator(masked)
    x_gen_play = generator(masked_noised_y)

    if mode == 'train':   
        if no_reconstruct:
            return tf.keras.models.Model(inputs, digit_caps_len, name='Efficinet_CapsNet_No_Generator')
        else:
            return tf.keras.models.Model([inputs, y_true], [digit_caps_len, x_gen_train], name='Efficinet_CapsNet_Generator')
    elif mode == 'test':
        if no_reconstruct:
            return tf.keras.models.Model(inputs, digit_caps_len, name='Efficinet_CapsNet_No_Generator')
        else:
            return tf.keras.models.Model(inputs, [digit_caps_len, x_gen_eval], name='Efficinet_CapsNet_Generator')
    elif mode == 'play':
        # In play mode you need to have a generator.
        return tf.keras.models.Model([inputs, y_true, noise], [digit_caps_len, x_gen_play], name='Efficinet_CapsNet_Generator')
    else:
        raise RuntimeError('mode not recognized')
