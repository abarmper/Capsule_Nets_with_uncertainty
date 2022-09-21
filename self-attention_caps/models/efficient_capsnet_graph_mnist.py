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


def efficient_capsnet_graph(input_shape, multihead=False, original_convs=False, num_heads=2, Algorithm = 'RooMAV', scale_the_embedding=True, use_agreement_criterion=True):
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
        x = tf.keras.layers.Conv2D(32,5,activation="relu", padding='valid', kernel_initializer='he_normal')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64,3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64,3, activation='relu', padding='valid', kernel_initializer='he_normal')(x)   
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(128,3,2, activation='relu', padding='valid', kernel_initializer='he_normal')(x)   
        x = tf.keras.layers.BatchNormalization()(x)
        x = PrimaryCaps(128, 9, 16, 8)(x) # 128/8 = Only 16 Primary Caps

    else:
        x = tf.keras.layers.Conv2D(256, 9, activation="relu", padding='valid')(inputs)
        #x = tf.keras.layers.Conv2D(256, 9, activation="relu", padding='valid')(x)
        x = PrimaryCaps(256, 9, 32*6*6, 8, s=2)(x) # 256/8 = 32, 6 x 6 are the result of the reshape.
        

    if multihead:
        digit_caps, c = FCCapsMultihead(10,16, A=num_heads, QKD=int(16/num_heads), D_v=int(16/num_heads), Alg1=Algo1, scaled_emb=scale_the_embedding, agreement_scores=use_agreement_criterion)(x)

        # x = tf.keras.layers.Flatten()(x)
        # x = tf.keras.layers.Dense(num_of_primary_caps*10*16, activation=None, use_bias=False)(x)
        # x = tf.keras.layers.Reshape(target_shape=(10,num_of_primary_caps,16))(x)
        # digit_caps, c = tf.keras.layers.MultiHeadAttention(4, 4, value_dim=4, dropout=0.0, use_bias=True, attention_axes=[3], output_shape=[1])(x,x,x, return_attention_scores=True)

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
        x = tf.keras.layers.Dense(6**2 * 10, activation='relu', kernel_initializer='he_normal')(inputs)
        x = tf.keras.layers.Reshape(target_shape=[6,6,10])(x)
        x = tf.keras.layers.Conv2DTranspose(32,9,2)(x)
        x = tf.keras.layers.Conv2DTranspose(64,9,1)(x)
        x = tf.keras.layers.Conv2DTranspose(1,2,1)(x)

    
    return tf.keras.Model(inputs=inputs, outputs=x, name='Generator')


def build_graph(input_shape, mode, verbose, deconv, multihead, original, num_heads, algorithm, scale_the_embedding, use_agreement_criterion, no_reconstruct):
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
    y_true = tf.keras.layers.Input(shape=(10,))
    noise = tf.keras.layers.Input(shape=(10, 16))

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
            return tf.keras.models.Model(inputs=inputs, outputs=digit_caps_len, name='Efficinet_CapsNet_No_Generator')
        else: 
            return tf.keras.models.Model([inputs, y_true], [digit_caps_len, x_gen_train], name='Efficinet_CapsNet_Generator')
    elif mode == 'test':
        if no_reconstruct:
            return tf.keras.models.Model(inputs=inputs, outputs=digit_caps_len, name='Efficinet_CapsNet_No_Generator')
        else:
            return tf.keras.models.Model(inputs, [digit_caps_len, x_gen_eval], name='Efficinet_CapsNet_Generator')
    elif mode == 'play':
        # In play mode you need to have a generator.
        return tf.keras.models.Model([inputs, y_true, noise], [digit_caps_len, x_gen_play], name='Efficinet_CapsNet_Generator')
    else:
        raise RuntimeError('mode not recognized')
