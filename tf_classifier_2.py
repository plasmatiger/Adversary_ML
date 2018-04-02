
# coding: utf-8

# In[54]:

import time
import math
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import skimage
from skimage.util import img_as_float
from skimage import io
from scipy import misc
#import dataset
import cv2
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from sklearn.metrics import confusion_matrix
from datetime import timedelta


# In[26]:

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

num_iterations = 1000
early_stopping = None


# In[74]:


def read_labeled_image_list(image_list_file):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line.split(' ')
        filenames.append(filename)
        labels.append(int(label))
    return filenames, labels


# In[72]:

def read_images_from_disk(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    label = input_queue[1]
    file_contents = tf.read_file(input_queue[0])
    example = tf.image.decode_jpeg(file_contents, channels=3)
    return example, label


# In[76]:

#load data : create training, validation, test set
image_list, label_list = read_labeled_image_list('/home/chhotabheem/Desktop/MLT_Dataset/all_files/new_dataset.txt')

image =  np.ndarray((8500, img_size, img_size, num_channels), dtype = 'float32')
label = np.zeros((8500, num_classes), dtype = 'int')

for i in range(8500):
    img = img_as_float(io.imread('/home/chhotabheem/Desktop/MLT_Dataset/all_files/'+ image_list[i]))
    if img.shape == (64,64):
        img = skimage.color.grey2rgb(img)
    #img = cv.LoadImageM('/home/chhotabheem/Desktop/MLT_Dataset/all_files/'+ image_list[i])
    #img = misc.imread('/home/chhotabheem/Desktop/MLT_Dataset/all_files/'+ image_list[i])
    print(img.shape, 'and', i, label_list[i])
    image[i] = np.asarray(img)
    label[i][label_list[i]] = 1
    

# Makes an input queue
# input_queue = tf.train.slice_input_producer([images, labels],
#                                             num_epochs=10,
#                                             shuffle=True)

# image, label = read_images_from_disk(input_queue)


# In[77]:

#define a conv layer

def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters):        # Number of filters.

    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = tf.Variable(tf.truncated_normal(shape = shape, stddev=0.01))
    biases = tf.Variable(tf.constant(0.05, shape=[num_filters]))

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


# In[78]:

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements() 
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features


# In[79]:

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs):    # Num. outputs.
    shape=[num_inputs, num_outputs]
    weights = tf.Variable(tf.truncated_normal(shape = shape, stddev=0.01))
    biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))

    layer = tf.matmul(input, weights) + biases
    layer = tf.nn.relu(layer)

    return layer


# In[80]:

#preparaing placeholders
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')

y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, axis =0)


# Actual code will begin now

# In[81]:

#actual code begins
#3 layers of convulation each containing pooling
layerc1, weightsc1 = new_conv_layer(input=x, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1)

layerc2, weightsc1 = new_conv_layer(input=layerc1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2)

layerc3, weightsc3 = new_conv_layer(input=layerc2, num_input_channels=num_filters2, filter_size=filter_size3, num_filters=num_filters3)


# In[82]:

#flattening step
layer_flat, num_features = flatten_layer(layerc3)

#fully-connected step
layer_fc = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs= num_classes)


# In[83]:

#prediction
y_pred = tf.nn.softmax(layer_fc)

y_pred_cls = tf.argmax(y_pred, axis =0)


# In[84]:

#loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc, labels=y_true)
cost = tf.reduce_mean(cross_entropy)


# In[85]:

#optimization
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)


# In[86]:

#performance
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[87]:

#tensorflow session
session = tf.Session()
session.run(tf.global_variables_initializer())


# In[92]:

def optimize(num_it):
    # Start-time used for printing time-usage below.
    start_time = time.time()
    
#     best_val_loss = float("inf")
#     patience = 0

    for i in range(num_it):
        x = image
        y = label

#         x_valid = 
#         y_valid = 

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]

        #tf.reshape(x, [8499, img_size_flat], 'x')
        #x_valid = x_valid.reshape(num_valid_examples, img_size_flat)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x, y_true: y}
        
        #feed_dict_validate = {x: x_valid,
        #                     y_true: y_valid}

        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)
        print(session.run(accuracy, feed_dict = feed_dict_train))
        #print(session.run(accuracy, feed_dict = feed_dict_validate))
        

        # Print status at end of each epoch (defined as full pass through training dataset).
    # Update the total number of iterations performed.
    #total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))


# In[93]:

optimize(num_iterations)


# In[ ]:



