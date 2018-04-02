from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
import scipy.misc
from PIL import Image
from skimage import io

import tensorflow as tf
import logging
from tensorflow.python.platform import flags

from cleverhans.attacks import FastGradientMethod
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf import model_train, model_eval, tf_model_load
from cleverhans_tutorials.tutorial_models import make_basic_cnn, MLP
from cleverhans_tutorials.tutorial_models import Flatten, Linear, ReLU, Softmax, Conv2D

from data_utils import load_cifar, load_fashion_mnist
from data_utils import load_cifar_sp, load_fashion_mnist_sp

#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Training Parameters
learning_rate = 0.001
batch_size = 128
nb_epochs = 40

im_chan = 3

num_classes = 10

base_dir = "./"
cache_dir = "./cache"

CIFAR = 1
FASHION = 0

TRAIN = 0

tf.set_random_seed(1234)
sess = tf.Session()


def cifar_net(n_classes):

    input_shape=((None, im_size, im_size, im_chan))
    layers = [Conv2D(128, (5, 5), (1, 1), "SAME"), ReLU(),
              Conv2D(128, (3, 3), (1, 1), "SAME"), ReLU(),
              Conv2D(64, (3, 3), (2, 2), "SAME"),  ReLU(),
              Conv2D(64, (3, 3), (1, 1), "SAME"),  ReLU(),
              Conv2D(64, (3, 3), (1, 1), "SAME"),  ReLU(),
              Conv2D(32, (3, 3), (2, 2), "SAME"),  ReLU(),
              Conv2D(12, (3, 3), (1, 1), "SAME"),  ReLU(),
              Flatten(),
              Linear(256), ReLU(),
              Linear(n_classes)]

    model = MLP(layers, input_shape)
    return model

def fashion_net(n_classes):

    input_shape=((None, im_size, im_size, im_chan))
    layers = [Conv2D(64, (8, 8), (2, 2), "SAME"), ReLU(),
              Conv2D(64 * 2, (6, 6), (2, 2), "VALID"), ReLU(),
              Conv2D(64 * 2, (5, 5), (1, 1), "VALID"), ReLU(),
              Flatten(),
              Linear(n_classes),
              Softmax()]

    model = MLP(layers, input_shape)
    return model


if CIFAR:
    '''
    Loads model and declares data related to CIFAR-10. 
    '''
    im_size = 32
    train_size = 40000
    test_size = 10000
    val_size = 10000

    X_train, X_test, X_val, Y_train, Y_test, Y_val = load_cifar(cache_dir)
    _, X_slic, _, _, Y_slic, _ = load_cifar_sp(cache_dir)

    # tf Graph input
    x = tf.placeholder(tf.float32, shape=(None, im_size, im_size, im_chan))
    y = tf.placeholder(tf.float32, shape=(None, num_classes))

    model = cifar_net(num_classes)
    preds = model.get_probs(x)

    MODEL_NAME = "cifar/cifar_model"
    MODEL_NAME_SP = "cifar_sp/cifar_model_sp"

elif FASHION:
    '''
    Loads model and declares data related to Fashion MNIST. 
    '''
    im_size = 28
    train_size = 50000
    test_size = 10000
    val_size = 10000

    X_train, X_test, X_val, Y_train, Y_test, Y_val = load_fashion_mnist(cache_dir)
    _, X_slic, _, _, Y_slic, _ = load_fashion_mnist_sp(cache_dir)

    # tf Graph input
    x = tf.placeholder(tf.float32, shape=(None, im_size, im_size, im_chan))
    y = tf.placeholder(tf.float32, shape=(None, num_classes))

    model = fashion_net(num_classes)
    preds = model.get_probs(x)

    MODEL_NAME = "fashion/fashion_model"
    MODEL_NAME_SP = "fashion_sp/fashion_model_sp"

else:
    print("No data option chosen! Exiting!")
    exit()

init = tf.global_variables_initializer()
sess.run(init)

rng = np.random.RandomState([2017, 8, 30])

train_params = {
    'nb_epochs': nb_epochs,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'train_dir': "./",
    'filename': cache_dir + "/" + MODEL_NAME
}

def evaluate():
    # Evaluate the accuracy on legitimate validaition examples
    eval_params = {'batch_size': batch_size}
    acc = model_eval(
          sess, x, y, preds, X_val, Y_val, args=eval_params)
    print('Validation accuracy on legitimate examples: %0.4f' % acc)

if TRAIN:
    model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                        args=train_params, rng=rng, save=True)

    eval_par = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds, X_train, Y_train, args=eval_par)
    print('Train accuracy on legitimate examples: %0.4f\n' % acc)

    acc = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_par)
    print('Test accuracy on legitimate examples: %0.4f\n' % acc)

if not TRAIN:
    tf_model_load(sess, cache_dir + "/" + MODEL_NAME)

fgsm_params = {
    'eps': 0.25,
    'clip_min': 0.,
    'clip_max': 1.
}

fgsm = FastGradientMethod(model, sess=sess)
adversarial_sample = fgsm.generate(x, **fgsm_params)

# Predictions and class of the adversarial examples.
adversarial_preds = model.get_probs(adversarial_sample)
adversarial_class = tf.argmax(adversarial_preds, axis=-1)

X_adv = np.ndarray(shape=X_test.shape)
Y_adv = np.copy(Y_test)

X_slic = np.ndarray(shape=X_test.shape)
Y_slic = np.copy(Y_test)

X_slic_adv = np.ndarray(shape=X_test.shape)
Y_slic_adv = np.copy(Y_test)

slic_args = {
    'n_segments' : 256,
    'compactness' : 5,
    'sigma' : 1,
    'gaussian_filter' : 0
}

from slic import slic_gaussian

for i in range(test_size):
    # Generating avdersarial images
    x_fordict = X_test[i, :, :, :].reshape(1, im_size, im_size, im_chan)
    y_fordict = Y_test[i, :].reshape(1, num_classes)

    feed_dict_test = {x: x_fordict, y: y_fordict}
    
    adv_image, probs, clas = sess.run([adversarial_sample, 
                                       adversarial_preds, 
                                       adversarial_class], 
                                       feed_dict=feed_dict_test)

    X_adv[i, :, :, :] = adv_image[0, :, :, :]

print("Done Adv!")
'''
for i in range(test_size):
    # Generating SLIC image of adversarial images
    X_slic_adv[i, :, :, :] = slic_gaussian(image=X_adv[i, :, :, :], args=slic_args)

print("Done Adv + SLIC!")
'''
if not TRAIN:
    tf_model_load(sess, cache_dir + "/" + MODEL_NAME_SP)

eval_par = {'batch_size': batch_size}
acc = model_eval(sess, x, y, preds, X_adv, Y_adv, args=eval_par)
print('Test accuracy on adversarial examples: %0.4f\n' % acc)

exit()

eval_par = {'batch_size': batch_size}
acc = model_eval(sess, x, y, preds, X_slic_adv, Y_slic_adv, args=eval_par)
print('Test accuracy on adversarial + slic examples: %0.4f\n' % acc)

# Once all the code is running, we can start drawing inferences
# and the images required to the added in the report.

