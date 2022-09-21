# Copyright 2021 Adam Byerly & Vittorio Mazzia & Francesco Salvetti. All Rights Reserved.
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
import os
import cv2
tf2 = tf.compat.v2

# constants
CIFAR10_IMG_SIZE = 32
CIFAR10_TRAIN_IMAGE_COUNT = 50000
PARALLEL_INPUT_CALLS = 16
PATCH_CIFAR10 = 24
AUTOTUNE = tf.data.AUTOTUNE

# normalize dataset
def pre_process(image, label):
	return image / 255.0, tf.keras.utils.to_categorical(label, num_classes=11)

def center_patches(image, label):
	return tf.image.central_crop(image, 0.75), label

def generator(image, label):
	return (image, label), (label, image)

def generate_tf_data(X_train, y_train, X_test, y_test, batch_size, patch_size):
	dataset_train = tf.data.Dataset.from_tensor_slices((X_train,y_train))
	dataset_train = dataset_train.map(center_patches,
	   num_parallel_calls=PARALLEL_INPUT_CALLS)
	dataset_train = dataset_train.shuffle(buffer_size=CIFAR10_TRAIN_IMAGE_COUNT)

	dataset_train = dataset_train.map(generator, num_parallel_calls=PARALLEL_INPUT_CALLS)
	dataset_train = dataset_train.batch(batch_size)
	dataset_train = dataset_train.prefetch(AUTOTUNE)

	dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

	dataset_test = dataset_test.cache()
	dataset_test = dataset_test.map(generator,
	num_parallel_calls=PARALLEL_INPUT_CALLS)
	dataset_test = dataset_test.batch(batch_size)
	dataset_test = dataset_test.prefetch(AUTOTUNE)

	return dataset_train, dataset_test

def generate_tf_data_no_reconstructor(X_train, y_train, X_test, y_test, batch_size, patch_size):
	dataset_train = tf.data.Dataset.from_tensor_slices((X_train,y_train))
	dataset_train = dataset_train.map(center_patches,
	   num_parallel_calls=PARALLEL_INPUT_CALLS)
	dataset_train = dataset_train.shuffle(buffer_size=CIFAR10_TRAIN_IMAGE_COUNT)

	dataset_train = dataset_train.batch(batch_size)
	dataset_train = dataset_train.prefetch(AUTOTUNE)

	dataset_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))

	dataset_test = dataset_test.cache()

	dataset_test = dataset_test.batch(batch_size)
	dataset_test = dataset_test.prefetch(AUTOTUNE)

	return dataset_train, dataset_test