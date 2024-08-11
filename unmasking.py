import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image as grayscale
img_path=str(input("Enter the image "))
image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Threshold the image to create a binary mask
ret, mask = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Find the contours in the mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contours with largest area - the ring is the largest
max_contour = max(contours, key=cv2.contourArea)

# Create a mask for the largest contour
mask = np.zeros_like(mask)
cv2.drawContours(mask, [max_contour], -1, 255, -1)

# Invert the mask
inverted_mask = cv2.bitwise_not(mask)

# Apply the inverted mask to the original image
completed_image = cv2.bitwise_and(image, image, mask=inverted_mask)
cv2.imshow("completed", completed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
