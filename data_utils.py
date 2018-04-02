from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from PIL import Image
import skimage.io as skio

from slic import slic_gaussian

def read_image(address):
    #image = Image.open(add).resize((im_size, im_size))
    image = Image.open(address)
    image = image.convert("RGB")
    image = np.array(image)/255.0
    #image = image.reshape((1, im_size, im_size, 3))
    return image

def load_cifar(cache_dir):
    ''' Loads and returns CIFAR10 dataset
    Mode = 0 : Load from cached files (saves time) 
    Mode = 1 : Load fresh by reading data from individual files
    '''

    train_size = 40000
    test_size = 10000
    val_size = 10000

    images_train = np.load(cache_dir + "/cifar_X_train.npy")
    images_test  = np.load(cache_dir + "/cifar_X_test.npy")
    images_val   = np.load(cache_dir + "/cifar_X_val.npy")
    labels_train = np.load(cache_dir + "/cifar_Y_train.npy")
    labels_test  = np.load(cache_dir + "/cifar_Y_test.npy")
    labels_val   = np.load(cache_dir + "/cifar_Y_val.npy")

    return images_train, images_test, images_val, labels_train, labels_test, labels_val

def load_fashion_mnist(cache_dir):
    ''' Loads and returns Fashion MNIST dataset
    Mode = 0 : Load from cached files (saves time) 
    Mode = 1 : Load fresh by reading data from individual files
    '''

    train_size = 50000
    test_size = 10000
    val_size = 10000

    images_train = np.load(cache_dir + "/fashion_X_train.npy")
    images_test  = np.load(cache_dir + "/fashion_X_test.npy")
    images_val   = np.load(cache_dir + "/fashion_X_val.npy")
    labels_train = np.load(cache_dir + "/fashion_Y_train.npy")
    labels_test  = np.load(cache_dir + "/fashion_Y_test.npy")
    labels_val   = np.load(cache_dir + "/fashion_Y_val.npy")

    return images_train, images_test, images_val, labels_train, labels_test, labels_val

def load_cifar_sp(cache_dir):
    ''' Loads and returns CIFAR10 dataset
    Mode = 0 : Load from cached files (saves time) 
    Mode = 1 : Load fresh by reading data from individual files
    '''

    train_size = 40000
    test_size = 10000
    val_size = 10000

    images_train = np.load(cache_dir + "/cifar_sp_X_train.npy")
    images_test  = np.load(cache_dir + "/cifar_sp_X_test.npy")
    images_val   = np.load(cache_dir + "/cifar_sp_X_val.npy")
    labels_train = np.load(cache_dir + "/cifar_sp_Y_train.npy")
    labels_test  = np.load(cache_dir + "/cifar_sp_Y_test.npy")
    labels_val   = np.load(cache_dir + "/cifar_sp_Y_val.npy")

    return images_train, images_test, images_val, labels_train, labels_test, labels_val

def load_fashion_mnist_sp(cache_dir):
    ''' Loads and returns Fashion MNIST dataset
    Mode = 0 : Load from cached files (saves time) 
    Mode = 1 : Load fresh by reading data from individual files
    '''

    train_size = 50000
    test_size = 10000
    val_size = 10000

    images_train = np.load(cache_dir + "/fashion_sp_X_train.npy")
    images_test  = np.load(cache_dir + "/fashion_sp_X_test.npy")
    images_val   = np.load(cache_dir + "/fashion_sp_X_val.npy")
    labels_train = np.load(cache_dir + "/fashion_sp_Y_train.npy")
    labels_test  = np.load(cache_dir + "/fashion_sp_Y_test.npy")
    labels_val   = np.load(cache_dir + "/fashion_sp_Y_val.npy")

    return images_train, images_test, images_val, labels_train, labels_test, labels_val

def cache_cifar(base_dir, cache_dir):

    # Metadata
    num_classes = 10
    train_size = 40000
    test_size = 10000
    val_size = 10000
    im_chan = 3
    im_size = 32

    clas_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    clas_dict = {'airplane':0, 'automobile':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    all_train_img_add = []
    all_train_labels = []

    train_img_add = []
    train_labels = []

    test_img_add = []
    test_labels = []

    val_img_add = []
    val_labels = []

    # Reading file lists
    im_dir = base_dir + "train/"
    
    for root, dirs, files in os.walk(im_dir):
        for name in files:
            add = im_dir + name
            all_train_img_add.append(add)

            for clas in clas_names:
                if clas in add:
                    all_train_labels.append(clas_dict[clas])

    im_dir = base_dir + "test/"
    
    for root, dirs, files in os.walk(im_dir):
        for name in files:
            add = im_dir + name
            test_img_add.append(add)

            for clas in clas_names:
                if clas in add:
                    test_labels.append(clas_dict[clas])

    a = list(range(len(all_train_img_add)))
    np.random.shuffle(a)
    
    for i in range(train_size):
        train_img_add.append(all_train_img_add[a[i]]) 
    for i in range(train_size):
        train_labels.append(all_train_labels[a[i]])
    for i in range(train_size, train_size + val_size):
        val_img_add.append(all_train_img_add[a[i]])
    for i in range(train_size, train_size + val_size):
        val_labels.append(all_train_labels[a[i]])

    # Loop to load the data in Numpy array

    # Reading the training data ==========================================
    images_train = np.ndarray(shape=(train_size, im_size, im_size, im_chan))
    labels_train = np.zeros(shape=(train_size, num_classes))

    start = 0
    end = train_size
    
    for ind, im_index in enumerate(range(start, end)):
        a = im_index
        images_train[ind, :, :, :] = read_image(train_img_add[im_index])
        lab = train_labels[im_index]
        labels_train[ind, lab] = 1
    # ====================================================================

    # Reading the testing data ===========================================
    images_test = np.ndarray(shape=(test_size, im_size, im_size, im_chan))
    labels_test = np.zeros(shape=(test_size, num_classes))

    start = 0
    end = test_size
    
    for ind, im_index in enumerate(range(start, end)):
        a =im_index
        images_test[ind, :, :, :] = read_image(test_img_add[im_index])
        lab = test_labels[im_index]
        labels_test[ind, lab] = 1
    # ====================================================================

    # Reading the validation data ========================================
    images_val = np.ndarray(shape=(val_size, im_size, im_size, im_chan))
    labels_val = np.zeros(shape=(val_size, num_classes))

    start = 0
    end = val_size
    
    for ind, im_index in enumerate(range(start, end)):
        a =im_index
        images_val[ind, :, :, :] = read_image(val_img_add[im_index])
        lab = val_labels[im_index]
        labels_val[ind, lab] = 1
    # ====================================================================

    # Save as numpy array ================================================
    np.save(cache_dir + "/cifar_X_train.npy", images_train)
    np.save(cache_dir + "/cifar_X_test.npy" , images_test)
    np.save(cache_dir + "/cifar_X_val.npy"  , images_val)
    np.save(cache_dir + "/cifar_Y_train.npy", labels_train)
    np.save(cache_dir + "/cifar_Y_test.npy" , labels_test)
    np.save(cache_dir + "/cifar_Y_val.npy"  , labels_val)


def cache_fashion_mnist(base_dir, cache_dir):

    # Metadata
    num_classes = 10
    train_size = 50000
    test_size = 10000
    val_size = 10000
    im_chan = 3
    im_size = 28

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Reading file lists
    im_dir = base_dir + "/train/"

    train_img_add = []
    train_labels = []

    test_img_add = []
    test_labels = []

    val_img_add = []
    val_labels = []
    
    for ind, clas in enumerate(classes):
        im_dir = base_dir + "/train/" + clas
        for root, dirs, files in os.walk(im_dir):
            for name in files:
                add = im_dir + "/" + name
                train_img_add.append(add)
                train_labels.append(ind)

        im_dir = base_dir + "/test/" + clas
        for root, dirs, files in os.walk(im_dir):
            for name in files:
                add = im_dir + "/" + name
                test_img_add.append(add)
                test_labels.append(ind)

        im_dir = base_dir + "/val/" + clas
        for root, dirs, files in os.walk(im_dir):
            for name in files:
                add = im_dir + "/" + name
                val_img_add.append(add)
                val_labels.append(ind)

    # Loop to load the data in Numpy array

    # Reading the training data ==========================================
    images_train = np.ndarray(shape=(train_size, im_size, im_size, im_chan))
    labels_train = np.zeros(shape=(train_size, num_classes))

    start = 0
    end = train_size
    
    for ind, im_index in enumerate(range(start, end)):
        a = im_index
        images_train[ind, :, :, :] = read_image(train_img_add[im_index])
        lab = train_labels[im_index]
        labels_train[ind, lab] = 1
    # ====================================================================

    # Reading the testing data ===========================================
    images_test = np.ndarray(shape=(test_size, im_size, im_size, im_chan))
    labels_test = np.zeros(shape=(test_size, num_classes))

    start = 0
    end = test_size
    
    for ind, im_index in enumerate(range(start, end)):
        a =im_index
        images_test[ind, :, :, :] = read_image(test_img_add[im_index])
        lab = test_labels[im_index]
        labels_test[ind, lab] = 1
    # ====================================================================

    # Reading the validation data ========================================
    images_val = np.ndarray(shape=(val_size, im_size, im_size, im_chan))
    labels_val = np.zeros(shape=(val_size, num_classes))

    start = 0
    end = val_size
    
    for ind, im_index in enumerate(range(start, end)):
        a =im_index
        images_val[ind, :, :, :] = read_image(val_img_add[im_index])
        lab = val_labels[im_index]
        labels_val[ind, lab] = 1
    # ====================================================================

    # Save as numpy array ================================================
    np.save(cache_dir + "/fashion_X_train.npy", images_train)
    np.save(cache_dir + "/fashion_X_test.npy" , images_test)
    np.save(cache_dir + "/fashion_X_val.npy"  , images_val)
    np.save(cache_dir + "/fashion_Y_train.npy", labels_train)
    np.save(cache_dir + "/fashion_Y_test.npy" , labels_test)
    np.save(cache_dir + "/fashion_Y_val.npy"  , labels_val)

def cache_cifar_sp(X_train, X_test, X_val, Y_train, Y_test, Y_val, cache_dir):

    slic_args = {
        'n_segments' : 256,
        'compactness' : 5,
        'sigma' : 1,
        'gaussian_filter' : 0
    }

    images_train = np.ndarray(shape=X_train.shape)
    images_test = np.ndarray(shape=X_test.shape)
    images_val = np.ndarray(shape=X_val.shape)

    l = X_train.shape[0]
    for i in range(l):
        # Generating SLIC images
        images_train[i, :, :, :] = slic_gaussian(image=X_train[i, :, :, :], args=slic_args)
    print("Loop1")

    l = X_test.shape[0]
    for i in range(l):
        # Generating SLIC images
        images_test[i, :, :, :] = slic_gaussian(image=X_test[i, :, :, :], args=slic_args)
        ima = images_test[i, :, :, :]
        skio.imsave("./sample/slic/cifar/" + str(i) + ".png", ima)
    print("Loop2")

    l = X_val.shape[0]
    for i in range(l):
        # Generating SLIC images
        images_val[i, :, :, :] = slic_gaussian(image=X_val[i, :, :, :], args=slic_args)
    print("Loop3")

    labels_train = np.copy(Y_train)
    labels_test = np.copy(Y_test)
    labels_val = np.copy(Y_val)

    # Save as numpy array ================================================
    np.save(cache_dir + "/cifar_sp_X_train.npy", images_train)
    np.save(cache_dir + "/cifar_sp_X_test.npy" , images_test)
    np.save(cache_dir + "/cifar_sp_X_val.npy"  , images_val)
    np.save(cache_dir + "/cifar_sp_Y_train.npy", labels_train)
    np.save(cache_dir + "/cifar_sp_Y_test.npy" , labels_test)
    np.save(cache_dir + "/cifar_sp_Y_val.npy"  , labels_val)
    
def cache_fashion_sp(X_train, X_test, X_val, Y_train, Y_test, Y_val, cache_dir):

    slic_args = {
        'n_segments' : 256,
        'compactness' : 5,
        'sigma' : 1,
        'gaussian_filter' : 0
    }

    images_train = np.ndarray(shape=X_train.shape)
    images_test = np.ndarray(shape=X_test.shape)
    images_val = np.ndarray(shape=X_val.shape)

    l = X_train.shape[0]
    for i in range(l):
        # Generating SLIC images
        images_train[i, :, :, :] = slic_gaussian(image=X_train[i, :, :, :], args=slic_args)
    print("Loop1")

    l = X_test.shape[0]
    for i in range(l):
        # Generating SLIC images
        images_test[i, :, :, :] = slic_gaussian(image=X_test[i, :, :, :], args=slic_args)
        ima = images_test[i, :, :, :]
        skio.imsave("./sample/slic/fashion/" + str(i) + ".png", ima)
    print("Loop2")

    l = X_val.shape[0]
    for i in range(l):
        # Generating SLIC images
        images_val[i, :, :, :] = slic_gaussian(image=X_val[i, :, :, :], args=slic_args)
    print("Loop3")

    labels_train = np.copy(Y_train)
    labels_test = np.copy(Y_test)
    labels_val = np.copy(Y_val)

    # Save as numpy array ================================================
    np.save(cache_dir + "/fashion_sp_X_train.npy", images_train)
    np.save(cache_dir + "/fashion_sp_X_test.npy" , images_test)
    np.save(cache_dir + "/fashion_sp_X_val.npy"  , images_val)
    np.save(cache_dir + "/fashion_sp_Y_train.npy", labels_train)
    np.save(cache_dir + "/fashion_sp_Y_test.npy" , labels_test)
    np.save(cache_dir + "/fashion_sp_Y_val.npy"  , labels_val)

''' 
if __name__ == "__main__":

    print("Started!")
    X_train, X_test, X_val, Y_train, Y_test, Y_val = load_cifar("./cache")
    print("Loaded!")
    cache_cifar_sp(X_train, X_test, X_val, Y_train, Y_test, Y_val, "./cache")
    print("Done!")
'''