import time
import math
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import dataset
import cv2

from sklearn.metrics import confusion_matrix
from datetime import timedelta

#%matplotlib inline


#Hyperparameter aside for easy tuning and not searching

#Convolution Layer 1
filter_size1 = 3 
num_filters1 = 16

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 16

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 32

# Fully-connected layer.
fc_size = 64            # Number of neurons in fully-connected layer.

# Number of color channels for the images (RBG)
num_channels = 3

# image dimensions
img_size = 64

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info
classes = ['Dog', 'Fish', 'Snake', 'Scorpion', 'Duck', 'Koala', 'Snail', 'Bear', 'Cat', 'Lion']
num_classes = len(classes)

#load data : create training, validation, test set

#define a conv layer

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters):        # Number of filters.

    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = tf.Variable(tf.truncated_normal(shape = shape, stddev=0.01))
    biases = tf.Variable(tf.constant(0.05, shape=[length]))

    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases
    layer = tf.nn.relu(layer)

    # Pooling to down-sample the image resolution

    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements() 
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs):    # Num. outputs.
    shape=[num_inputs, num_outputs]
    weights = tf.Variable(tf.truncated_normal(shape = shape, stddev=0.01))
    biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))

    layer = tf.matmul(input, weights) + biases
    layer = tf.nn.relu(layer)

    return layer

#preparaing placeholders
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

#actual code begins
#3 layers of convulation each containing pooling
layerc1, weightsc1 = new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1)

layerc2, weightsc1 = new_conv_layer(input=layerc1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2)

layerc3, weightsc3 = new_conv_layer(input=layerc2, num_input_channels=num_filters2, filter_size=filter_size3, num_filters=num_filters3)

#flattening step
layer_flat, num_features = flatten_layer(layerc3)

#fully-connected step
layer_fc = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs= num_classes)

#prediction
y_pred = tf.nn.softmax(layer_fc)

#class_prediction
y_pred_cls = tf.argmax(y_pred, dimension=1)

#loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc, labels=y_true)
cost = tf.reduce_mean(cross_entropy)

#optimization
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

#performance
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




#tensorflow session
session = tf.Session()
session.run(tf.initialize_all_variables())