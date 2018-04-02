import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
#from skimage.segmentation import get_mean_colors
from skimage.util import img_as_float
from skimage import io
import numpy as np

#pylab inline
#matplotlib inline 

def get_mean(img, indic):
	indica = img*indic
	n = indic.sum()
	return int(indica.sum()*1.0/n)

img = img_as_float(io.imread('f1.JPEG'))
io.imshow(img);
imga = np.zeros(img.shape, dtype = 'uint8')

count =0



segments = slic(img, n_segments = 10, compactness = 10, sigma = 1)
for v in np.unique(segments):
	print(count)
	count += 1
	# construct a mask for the segment
	# print "[x] inspecting segment %d" % (i)
	indic = np.zeros(img.shape, dtype = "uint8")
	indic[segments == v] = 1
	imga[segments == v] = get_mean(img, indic)

io.imshow(imga);
fig = plt.figure(figsize=(12,4))
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(mark_boundaries(img, segments)) 
plt.show()

#50, 100, 300, 500, 1000