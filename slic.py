
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import numpy as np
from skimage import io
from skimage.filters import gaussian
from skimage.util import img_as_float
from skimage.segmentation import slic, mark_boundaries


def slic_gaussian(image=None, channels=3, args=None):
	''' 
	Input: Image (Numpy array) (3 channels)
		   If an image is single channeled, pass by converting
		   to 3 channel.

	Output: Image (Numpy array)
	
	This function takes an image in the form of a numpy array
	and performs SLIC (pixel aggregation).
	'''

	if image is None:
		return image

	n_segments_ = args['n_segments']
	compactness_ = args['compactness']
	sigma_ = args['sigma']
	perform_gaussian = args['gaussian_filter']

	image_out = np.zeros(image.shape, dtype = np.float32)

	segments = slic(image, n_segments = n_segments_, 
						   compactness = compactness_, 
						   sigma = sigma_)

	for seg in np.unique(segments):
		mask = np.zeros(image.shape[:2], dtype = np.float32)
		mask[segments == seg] = 1
	
		n = np.sum(mask)

		for ch in range(3): 
			image_temp = np.multiply(image[:, :, ch], mask)
			cavg = np.sum(image_temp)/n
			image_out[:, :, ch] = image_out[:, :, ch] + cavg*0.99*mask

	if perform_gaussian:
		image_out = gaussian(image_out, multichannel=True)

	return image_out
