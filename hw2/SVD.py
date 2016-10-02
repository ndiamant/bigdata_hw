
# Import libraries
import PIL
import scipy.linalg as splin
import numpy as np
import matplotlib.pylab as plt
get_ipython().magic('matplotlib inline')


# Import Image
from PIL import Image
im = Image.open("clown.jpg")
im = np.array(im)

# SVD using numpy
X = im[:,:,0]
U, s, V = np.linalg.svd(X, full_matrices=True)
S = np.zeros((U.shape[1], V.shape[0]))
S[:len(s), :len(s)] = np.diag(s)


# Generate the various compressed images
Scompress20 = np.copy(S)
Scompress10 = np.copy(S)
Scompress2 = np.copy(S)
Scompress20[20:,20:] = 0
Scompress10[10:,10:] = 0
Scompress2[2:,2:] = 0
Xcompress = [U @ Sc @ V for Sc in [Scompress20,
			Scompress10,Scompress2]]

# Plot the compressed images
plt.figure(figsize = (10,10))
plt.subplot(221)
plt.imshow(im)
plt.title('Original Image')
plt.axis('off')
plt.subplot(222)
plt.imshow(Image.fromarray(Xcompress[0]))
plt.title('20 Singular Values')
plt.axis('off')
plt.subplot(223)
plt.imshow(Image.fromarray(Xcompress[1]))
plt.title('10 Singular Values')
plt.axis('off')
plt.subplot(224)
plt.imshow(Image.fromarray(Xcompress[2]))
plt.title('2 Singular Values')
plt.axis('off')
plt.tight_layout()


# Plot the singular values
plt.figure(figsize = (15,10))
plt.plot(s[:100], label = "Original Image")
X2 = X.flatten()
np.random.shuffle(X2)
X2 = X2.reshape(X.shape[0], X.shape[1])
U, s2, V = np.linalg.svd(X2, full_matrices=True)
plt.plot(s2[:100], label = "Randomized Image")
plt.ylim([0,10000])
plt.title("First 100 Singular Values")
plt.legend()
