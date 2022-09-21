'''
This file is responsible for downloading the datasets used to train the model.
It also calls preprocessing functions so as to prepare the data before the get fed into the model.

Made by Alexandrow Barmperis under CC-BY 4.0 license.
'''

import numpy as np
import tensorflow as tf
from tensorflow._api.v2 import data
from tensorflow.python.keras.utils.tf_utils import dataset_is_infinite
import tensorflow_datasets as tfds
from tqdm.notebook import tqdm

AUTOTUNE = tf.data.AUTOTUNE
PARALLEL_INPUT_CALLS = 16
SMALLNORB_SAMPLES = 24300
SMALLNORB_PATCH_SMALLNORB = 48
SMALLNORB_INPUT_SHAPE = 96
SMALLNORB_N_CLASSES = 5
SMALLNORB_MAX_DELTA = 2.0
SMALLNORB_LOWER_CONTRAST = 0.5
SMALLNORB_UPPER_CONTRAST = 1.5
SMALLNORB_SCALE = 64
SMALLNORB_PATCH = 48

MULTIMNIST_IMG_SIZE = 36
MULTIMNIST_PAD = 4
MULTIMNIST_SHIFT = 6

class Dataset(object):
    '''
    Calss that handles the data (gets the desired data, applyies transformations etc.)
    '''
    def __init__(self, dataset_name, batch_sz, gen):
        self.class_names = None
        self.ds_train = None
        self.ds_test = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.X_val = None
        self.y_val = None
        self.ds_info = None
        self.gen = gen
        self.batch_size = batch_sz
        self.X_test_patch = None
        if dataset_name == 'MNIST':
            self.get_MNIST_data()
        elif dataset_name == 'FASHION-MNIST':
            self.get_FASHION_MNIST_data()
        elif dataset_name == 'CIFAR10':
            self.get_CIFAR10_data()
        elif dataset_name == 'SMALLNORB':
            self.get_SMALLNORB_data()
        elif dataset_name == 'MULTIMNIST':
            self.get_MULTIMNIST_data()
        else:
            raise ValueError("Invalid dataset argument.")

    # def get_MNIST_data(self):
    #     (ds_train, ds_test), ds_info = tfds.load(
    #     'mnist',
    #     split=['train', 'test'],
    #     shuffle_files=True,
    #     as_supervised=True,
    #     with_info=True,
    #     )
    #     def normalize_img(image, label):
    #         """Normalizes images: `uint8` -> `float32`."""
    #         return tf.cast(image, tf.float32) / 255., label
        
    #     ds_train = ds_train.map(
    #     normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    #     ds_train = ds_train.cache()
    #     ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    #     ds_train = ds_train.batch(self.batch_size)
    #     ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
        
    #     ds_test = ds_test.map(
    #     normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    #     ds_test = ds_test.batch(self.batch_size)
    #     ds_test = ds_test.cache()
    #     ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    def generator(self, image, label):
        return (image, label), (label, image)

    def generator_test(self, image, label):
        return (image, ), (label, image)

    def get_MNIST_data(self):
        train, test = tf.keras.datasets.mnist.load_data()
        images, labels = train
        # unsqueeze = add one last dimension (channels)
        images = np.expand_dims(images, axis=-1) # keras convolutional layers require shape of format: (Batch_size, height, width, channels)
        images = (images/255.).astype(np.float32)
        self.ds_train = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(100)
        if self.gen:
            self.ds_train = self.ds_train.map(self.generator, num_parallel_calls=PARALLEL_INPUT_CALLS)
        self.ds_train = self.ds_train.batch(self.batch_size).prefetch(AUTOTUNE)
        self.X_train, self.y_train = tf.convert_to_tensor(images), tf.convert_to_tensor(labels, dtype=tf.int32)

        images, labels = test
        # unsqueeze = add one last dimension (channels)
        images = np.expand_dims(images, axis=-1)
        images = (images/255.).astype(np.float32)
        self.ds_val = tf.data.Dataset.from_tensor_slices((images, labels))
        if self.gen:
            self.ds_val = self.ds_val.map(self.generator, num_parallel_calls=PARALLEL_INPUT_CALLS)
        self.ds_val= self.ds_val.batch(self.batch_size).prefetch(AUTOTUNE)
        self.X_val, self.y_val = tf.convert_to_tensor(images), tf.convert_to_tensor(labels, dtype=tf.int32)
        
        images, labels = test
        # unsqueeze = add one last dimension (channels)
        images = np.expand_dims(images, axis=-1)
        images = (images/255.).astype(np.float32)
        self.ds_test = tf.data.Dataset.from_tensor_slices((images, labels))
        if self.gen:
            self.ds_test = self.ds_test.map(self.generator_test, num_parallel_calls=PARALLEL_INPUT_CALLS)
        self.ds_test= self.ds_test.batch(self.batch_size).prefetch(AUTOTUNE)
        self.X_test, self.y_test = tf.convert_to_tensor(images), tf.convert_to_tensor(labels, dtype=tf.int32)
        self.class_names = list(range(10))

    def get_CIFAR_data(self):
        train, test = tf.keras.datasets.cifar10.load_data()
        images, labels = train
        
        # unsqueeze = add one last dimension (channels)
        images = np.expand_dims(images, axis=-1) # keras convolutional layers require shape of format: (Batch_size, height, width, channels)
        images = (images/255.).astype(np.float32)
        self.ds_train = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(100)
        if self.gen:
            self.ds_train.map(self.generator, num_parallel_calls=PARALLEL_INPUT_CALLS)
        self.ds_train = self.ds_train.batch(self.batch_size).prefetch(AUTOTUNE)
        self.X_train, self.y_train = tf.convert_to_tensor(images), tf.convert_to_tensor(labels, dtype=tf.int32)

        images, labels = test
        # unsqueeze = add one last dimension (channels)
        images = np.expand_dims(images, axis=-1)
        images = (images/255.).astype(np.float32)
        self.ds_val = tf.data.Dataset.from_tensor_slices((images, labels))
        if self.gen:
            self.ds_val = self.ds_val.map(self.generator, num_parallel_calls=PARALLEL_INPUT_CALLS)
        self.ds_val= self.ds_val.batch(self.batch_size).prefetch(AUTOTUNE)
        self.X_val, self.y_val = tf.convert_to_tensor(images), tf.convert_to_tensor(labels, dtype=tf.int32)
        
        images, labels = test
        # unsqueeze = add one last dimension (channels)
        images = np.expand_dims(images, axis=-1)
        images = (images/255.).astype(np.float32)
        self.ds_test = tf.data.Dataset.from_tensor_slices((images, labels))
        if self.gen:
            self.ds_test = self.ds_test.map(self.generator_test, num_parallel_calls=PARALLEL_INPUT_CALLS)
        self.ds_test= self.ds_test.batch(self.batch_size).prefetch(AUTOTUNE)
        self.X_test, self.y_test = tf.convert_to_tensor(images), tf.convert_to_tensor(labels, dtype=tf.int32)
        self.class_names = list(range(10))

    def get_FASHION_MNIST_data(self):
        train, test = tf.keras.datasets.fashion_mnist.load_data()
        images, labels = train
        # unsqueeze = add one last dimension (channels)
        images = np.expand_dims(images, axis=-1) # keras convolutional layers require shape of format: (Batch_size, height, width, channels)
        images = (images/255.).astype(np.float32)
        self.ds_train = self.ds_train = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(100)
        if self.gen:
            self.ds_train.map(self.generator, num_parallel_calls=PARALLEL_INPUT_CALLS)
        self.ds_train.batch(self.batch_size).prefetch(AUTOTUNE)
        self.X_train, self.y_train = tf.convert_to_tensor(images), tf.convert_to_tensor(labels, dtype=tf.int32)

        images, labels = test
        # unsqueeze = add one last dimension (channels)
        images = np.expand_dims(images, axis=-1)
        images = (images/255.).astype(np.float32)
        self.ds_val = tf.data.Dataset.from_tensor_slices((images, labels))
        if self.gen:
            self.ds_val = self.ds_val.map(self.generator, num_parallel_calls=PARALLEL_INPUT_CALLS)
        self.ds_val= self.ds_val.batch(self.batch_size).prefetch(AUTOTUNE)
        self.X_val, self.y_val = tf.convert_to_tensor(images), tf.convert_to_tensor(labels, dtype=tf.int32)

        images, labels = test
        # unsqueeze = add one last dimension (channels)
        images = np.expand_dims(images, axis=-1)
        images = (images/255.).astype(np.float32)
        self.ds_test = tf.data.Dataset.from_tensor_slices((images, labels))
        if self.gen:
            self.ds_test = self.ds_test.map(self.generator_test, num_parallel_calls=PARALLEL_INPUT_CALLS)
        self.ds_test= self.ds_test.batch(self.batch_size).prefetch(AUTOTUNE)
        self.X_test, self.y_test = tf.convert_to_tensor(images), tf.convert_to_tensor(labels, dtype=tf.int32)
        self.class_names = list(range(10))
        
    def get_SMALLNORB_data(self):
                # import the datatset
        (self.ds_train, self.ds_test), self.ds_info = tfds.load(
            'smallnorb',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=False,
            with_info=True)
        self.X_train, self.y_train = self.smallnorb_pre_process(self.ds_train)
        self.X_test, self.y_test = self.smallnorb_pre_process(self.ds_test)

        self.X_train, self.y_train = self.smallnorb_standardize(self.X_train, self.y_train)
        self.X_train, self.y_train = self.smallnorb_rescale(self.X_train, self.y_train)
        self.X_test, self.y_test = self.smallnorb_standardize(self.X_test, self.y_test)
        self.X_test, self.y_test = self.smallnorb_rescale(self.X_test, self.y_test) 
        self.X_test_patch, self.y_test = self.smallnorb_test_patches(self.X_test, self.y_test)
        self.class_names = self.ds_info.features['label_category'].names
        
            
        self.ds_train = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        # self.ds_train = self.ds_train.shuffle(buffer_size=SMALLNORB_SAMPLES) not needed if imported with tfds
        self.ds_train = self.ds_train.map(self.smallnorb_random_patches,
            num_parallel_calls=PARALLEL_INPUT_CALLS)
        self.ds_train = self.ds_train.map(self.smallnorb_random_brightness,
            num_parallel_calls=PARALLEL_INPUT_CALLS)
        self.ds_train = self.ds_train.map(self.smallnorb_random_contrast,
            num_parallel_calls=PARALLEL_INPUT_CALLS)
        if self.gen:
            self.ds_train = self.ds_train.map(self.generator,
                num_parallel_calls=PARALLEL_INPUT_CALLS)
        self.ds_train = self.ds_train.batch(self.batch_size)
        self.ds_train = self.ds_train.prefetch(AUTOTUNE)
        
        self.ds_val = tf.data.Dataset.from_tensor_slices((self.X_test_patch, self.y_test))
        
        if self.gen:
            self.ds_val = self.ds_val.map(self.generator,
                num_parallel_calls=PARALLEL_INPUT_CALLS)
        self.ds_val = self.ds_val.batch(self.batch_size) # Could be set to 1 on testing.
        self.ds_val = self.ds_val.prefetch(AUTOTUNE)

        self.ds_test = tf.data.Dataset.from_tensor_slices((self.X_test_patch, self.y_test))
        
        if self.gen:
            self.ds_test = self.ds_test.map(self.generator_test,
                num_parallel_calls=PARALLEL_INPUT_CALLS)
        self.ds_test = self.ds_test.batch(self.batch_size) # Could be set to 1 on testing.
        self.ds_test = self.ds_test.prefetch(AUTOTUNE)
        
        print("[INFO] SMALLNORB Dataset load Completed!")
        return

                
    def get_MULTIMNIST_data(self):
        (self.X_train, self.y_train), (self.X_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        # prepare the data
        self.X_train = self.multimnist_pad_dataset(self.X_train, MULTIMNIST_PAD)
        self.X_test = self.multimnist_pad_dataset(self.X_test, MULTIMNIST_PAD)
        self.X_train, self.y_train = self.multimnist_pre_process(self.X_train, self.y_train)
        self.X_test, self.y_test = self.multimnist_pre_process(self.X_test, self.y_test)
        self.class_names = list(range(10))


        if self.gen:
            input_shape = (MULTIMNIST_IMG_SIZE,MULTIMNIST_IMG_SIZE,1)
            self.ds_train = tf.data.Dataset.from_generator(self.multimnist_generator(self.X_train, self.y_train, MULTIMNIST_SHIFT),
                                                        output_shapes=((input_shape, (10,),(10,)), ((10,), input_shape, input_shape)),
                                                        output_types=((tf.float32, tf.float32, tf.float32),
                                                                        (tf.float32, tf.float32, tf.float32)))
            self.ds_train = self.ds_train.batch(self.batch_size).prefetch(AUTOTUNE)
            self.ds_val = tf.data.Dataset.from_generator(self.multimnist_generator_validation(self.X_test, self.y_test, MULTIMNIST_SHIFT),
                                                        output_shapes=((input_shape, (10,),(10,)), ((10,), input_shape, input_shape)),
                                                        output_types=((tf.float32, tf.float32, tf.float32),
                                                                    (tf.float32, tf.float32, tf.float32)))
            self.ds_val = self.ds_val.batch(self.batch_size).prefetch(AUTOTUNE)
            
            # Test dataset
            input_shape = (MULTIMNIST_IMG_SIZE,MULTIMNIST_IMG_SIZE,1)
            n_multi = 1000
            np.random.seed(42)
            self.ds_test = tf.data.Dataset.from_generator(self.multimnist_generator_test(self.X_test, self.y_test, MULTIMNIST_SHIFT, n_multi),
                                                        output_shapes=((n_multi,)+input_shape,(n_multi,10,)),
                                                        output_types=(tf.float32,tf.float32))
            self.ds_test = self.ds_test.prefetch(AUTOTUNE)
            
        else:
            input_shape = (MULTIMNIST_IMG_SIZE,MULTIMNIST_IMG_SIZE,1)
            self.ds_train = tf.data.Dataset.from_generator(self.multimnist_generator_no_reconstructor(self.X_train,self.y_train,MULTIMNIST_SHIFT),
                                                           output_signature=(tf.TensorSpec(shape=input_shape,dtype=tf.float32),
                                                                             tf.TensorSpec(shape=[10], dtype=tf.float32)))
            self.ds_train = self.ds_train.batch(self.batch_size).prefetch(AUTOTUNE)
            self.ds_val = tf.data.Dataset.from_generator(self.multimnist_generator_validation_no_reconstructor(self.X_test, self.y_test, MULTIMNIST_SHIFT),
                                                        output_shapes=((input_shape),(None, 10)),
                                                        output_types=((tf.float32), (tf.float32)))
            self.ds_val = self.ds_val.batch(self.batch_size).prefetch(AUTOTUNE)
            
            # Test dataset
            input_shape = (MULTIMNIST_IMG_SIZE,MULTIMNIST_IMG_SIZE,1)
            n_multi = 1000
            np.random.seed(42)
            self.ds_test = tf.data.Dataset.from_generator(self.multimnist_generator_test_no_reconstructor(self.X_test, self.y_test, MULTIMNIST_SHIFT, n_multi),
                                                        output_shapes=((n_multi,)+input_shape,(n_multi,10,)),
                                                        output_types=(tf.float32,tf.float32))
            self.ds_test = self.ds_test.prefetch(AUTOTUNE)
            
        print("[INFO] MULTIMNIST Dataset Load Completed!")
        return
            
        
    def multimnist_pad_dataset(self, images,pad):
        return np.pad(images,[(0,0),(pad,pad),(pad,pad)])

    def multimnist_pre_process(self, image, label):
        return (image / 255)[...,None].astype('float32'), tf.keras.utils.to_categorical(label, num_classes=10)

    def multimnist_shift_images(self, images, shifts, max_shift):
        l = images.shape[1]
        images_sh = np.pad(images,((0,0),(max_shift,max_shift),(max_shift,max_shift),(0,0)))
        shifts = max_shift - shifts
        batches = np.arange(len(images))[:,None,None]
        images_sh = images_sh[batches,np.arange(l+max_shift*2)[None,:,None],(shifts[:,0,None]+np.arange(0,l))[:,None,:]]
        images_sh = images_sh[batches,(shifts[:,1,None]+np.arange(0,l))[...,None],np.arange(l)[None,None]]
        return images_sh

    def multimnist_merge_with_image(self, images,labels,i,shift,n_multi=1000): #for an image i, generate n_multi merged images
        base_image = images[i]
        base_label = labels[i]
        indexes = np.arange(len(images))[np.bitwise_not((labels==base_label).all(axis=-1))]
        indexes = np.random.choice(indexes,n_multi,replace=False)
        top_images = images[indexes]
        top_labels = labels[indexes]
        shifts = np.random.randint(-shift,shift+1,(n_multi+1,2))
        images_sh = self.multimnist_shift_images(np.concatenate((base_image[None],top_images),axis=0),shifts,shift)
        base_sh = images_sh[0]
        top_sh = images_sh[1:]
        merged = np.clip(base_sh+top_sh,0,1)
        merged_labels = base_label+top_labels
        return merged,merged_labels

    def multimnist_generator(self, images,labels,shift):
        def multi_mnist():
            while True:
                i = np.random.randint(len(images))
                j = np.random.randint(len(images))
                while np.all(images[i]==images[j]):
                    j = np.random.randint(len(images))
                base = self.multimnist_shift_images(images[i:i+1],np.random.randint(-shift,shift+1,(1,2)),shift)[0]
                top = self.multimnist_shift_images(images[j:j+1],np.random.randint(-shift,shift+1,(1,2)),shift)[0]
                merged = tf.clip_by_value(tf.add(base, top),0,1)
                yield (merged,labels[i],labels[j]),(labels[i]+labels[j],base,top)
        return multi_mnist
        
    def multimnist_generator_validation(self, images,labels,shift):
        def multi_mnist_val():
            for i in range(len(images)):
                j = np.random.randint(len(images))
                while np.all(labels[i]==labels[j]):
                    j = np.random.randint(len(images))
                base = self.multimnist_shift_images(images[i:i+1],np.random.randint(-shift,shift+1,(1,2)),shift)[0]
                top = self.multimnist_shift_images(images[j:j+1],np.random.randint(-shift,shift+1,(1,2)),shift)[0]
                merged = tf.clip_by_value(tf.add(base, top),0,1)
                yield (merged,labels[i],labels[j]),(labels[i]+labels[j],base,top)
        return multi_mnist_val

    def multimnist_generator_no_reconstructor(self, images,labels,shift):
        def multi_mnist():
            while True:
                i = np.random.randint(len(images))
                j = np.random.randint(len(images))
                while np.all(images[i]==images[j]):
                    j = np.random.randint(len(images))
                base = self.multimnist_shift_images(images[i:i+1],np.random.randint(-shift,shift+1,(1,2)),shift)[0]
                top = self.multimnist_shift_images(images[j:j+1],np.random.randint(-shift,shift+1,(1,2)),shift)[0]
                merged = tf.clip_by_value(tf.add(base, top),0,1)
                yield (merged),(labels[i]+labels[j])
        return multi_mnist
        
    def multimnist_generator_validation_no_reconstructor(self, images,labels,shift):
        def multi_mnist_val():
            for i in range(len(images)):
                j = np.random.randint(len(images))
                while np.all(labels[i]==labels[j]):
                    j = np.random.randint(len(images))
                base = self.multimnist_shift_images(images[i:i+1],np.random.randint(-shift,shift+1,(1,2)),shift)[0]
                top = self.multimnist_shift_images(images[j:j+1],np.random.randint(-shift,shift+1,(1,2)),shift)[0]
                merged = tf.clip_by_value(tf.add(base, top),0,1)
                yield (merged),(labels[i]+labels[j])
        return multi_mnist_val

    def multimnist_generator_test(self, images,labels,shift,n_multi=1000):
        def multi_mnist_test():
            for i in range(len(images)):
                X_merged,y_merged = self.multimnist_merge_with_image(images,labels,i,shift,n_multi)
                yield X_merged,y_merged
        return multi_mnist_test 

    def multimnist_generator_test_no_reconstructor(self, images,labels,shift,n_multi=1000):
        def multi_mnist_test():
            for i in range(len(images)):
                X_merged,y_merged = self.multimnist_merge_with_image(images,labels,i,shift,n_multi)
                yield X_merged,y_merged
        return multi_mnist_test 

    # Helper Functions
    def smallnorb_pre_process(self, ds):
        X = np.empty((SMALLNORB_SAMPLES, SMALLNORB_INPUT_SHAPE, SMALLNORB_INPUT_SHAPE, 2))
        y = np.empty((SMALLNORB_SAMPLES,))
            
        for index, d in tqdm(enumerate(ds.batch(1))):
            X[index, :, :, 0:1] = d['image']
            X[index, :, :, 1:2] = d['image2']
            y[index] = d['label_category']
        return X, y
    
    def smallnorb_standardize(self, x, y):
        x[...,0] = (x[...,0] - x[...,0].mean()) / x[...,0].std()
        x[...,1] = (x[...,1] - x[...,1].mean()) / x[...,1].std()
        return x, tf.one_hot(y, SMALLNORB_N_CLASSES)

    def smallnorb_rescale(self, x, y):
        with tf.device("/cpu:0"):
            x = tf.image.resize(x , [SMALLNORB_SCALE, SMALLNORB_SCALE])
        return x, y

    def smallnorb_test_patches(self, x, y):
        res = (SMALLNORB_SCALE - SMALLNORB_PATCH) // 2
        return x[:,res:-res,res:-res,:], y
    
    def smallnorb_random_patches(self, x, y):
        return tf.image.random_crop(x, [SMALLNORB_PATCH, SMALLNORB_PATCH, 2]), y

    def smallnorb_random_brightness(self, x, y):
        return tf.image.random_brightness(x, max_delta=SMALLNORB_MAX_DELTA), y

    def smallnorb_random_contrast(self, x, y):
        return tf.image.random_contrast(x, lower=SMALLNORB_LOWER_CONTRAST, upper=SMALLNORB_UPPER_CONTRAST), y
# from numba import cuda 
# device = cuda.get_current_device()
# device.reset()
