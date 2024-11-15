# uts-ersha
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale mode
image = cv2.imread('image.jpg', 0)  # Replace 'image.jpg' with the path to your image

# Step 1: Calculate the histogram of the image
hist, bins = np.histogram(image.flatten(), 256, [0, 256])

# Step 2: Calculate the cumulative distribution function (CDF)
cdf = hist.cumsum()
cdf_normalized = cdf * hist.max() / cdf.max()

# Step 3: Normalize the CDF
cdf_m = np.ma.masked_equal(cdf, 0)
cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
cdf = np.ma.filled(cdf_m, 0).astype('uint8')

# Step 4: Apply the histogram equalization to get the new image
equalized_image = cdf[image]

# Display the original and equalized images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Equalized Image")
plt.imshow(equalized_image, cmap='gray')

plt.show()

# Step 5: Plot histogram for both original and equalized images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Histogram (Original Image)")
plt.hist(image.flatten(), 256, [0, 256], color='r')

plt.subplot(1, 2, 2)
plt.title("Histogram (Equalized Image)")
plt.hist(equalized_image.flatten(), 256, [0, 256], color='g')

plt.show()
