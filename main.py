import cv2
import numpy as np
import matplotlib.pyplot as plt

def iris_segmentation(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error loading image")
        return

    # Apply GaussianBlur to reduce noise and improve circle detection
    blurred_image = cv2.GaussianBlur(image, (9, 9), 2)

    # Detect the iris using the HoughCircles method
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                               param1=100, param2=30, minRadius=20, maxRadius=80)

    if circles is not None:
        circles = np.uint8(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]

            # Draw the outer circle
            cv2.circle(image, center, radius, (255, 0, 0), 2)
            # Draw the center of the circle
            cv2.circle(image, center, 2, (0, 255, 0), 3)
    else:
        print("No circles were found")

    # Display the result
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.title('Iris Segmentation')
    plt.axis('off')
    plt.show()

# Path to the input image
image_path = 'images 1.jpeg'

# Perform iris segmentation
iris_segmentation(image_path)
