import cv2
import numpy as np
import matplotlib.pyplot as plt

def iris_pupil_segmentation(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image '{image_path}'")
        return

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply median blur to reduce noise
    blurred_image = cv2.medianBlur(gray_image, 5)

    # Use adaptive histogram equalization to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(blurred_image)

    # Detect edges using the Canny edge detector
    edges = cv2.Canny(enhanced_image, threshold1=50, threshold2=150)

    # Use HoughCircles to detect the outer boundary of the iris
    iris_circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=10,
                                    param1=100, param2=30, minRadius=20, maxRadius=40)

    if iris_circles is not None:
        iris_circles = np.uint16(np.around(iris_circles))

        # Assuming the first detected circle is the iris
        iris_circle = iris_circles[0][0]
        iris_center = (int(iris_circle[0]), int(iris_circle[1]))
        iris_radius = int(iris_circle[2])

        # Create a mask for the iris
        iris_mask = np.zeros_like(gray_image)
        cv2.circle(iris_mask, iris_center, iris_radius, 255, thickness=-1)

        # Refine the pupil detection within the iris region
        pupil_edges = cv2.bitwise_and(edges, edges, mask=iris_mask)
        pupil_circles = cv2.HoughCircles(pupil_edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                                         param1=100, param2=20, minRadius=10, maxRadius=iris_radius // 2)

        if pupil_circles is not None:
            pupil_circles = np.uint16(np.around(pupil_circles))
            # Assuming the first detected circle within the iris is the pupil
            pupil_circle = pupil_circles[0][0]
            pupil_center = (int(pupil_circle[0]), int(pupil_circle[1]))
            pupil_radius = int(pupil_circle[2])
        else:
            # If pupil detection fails, use iris center and smaller radius
            pupil_center = iris_center
            pupil_radius = iris_radius // 3

        # Draw the iris
        cv2.circle(image, iris_center, iris_radius, (255, 0, 0), 2)

        # Ensure text position is within the image bounds for iris
        text_x_iris = max(0, min(image.shape[1] - 30, iris_center[0] - 30))
        text_y_iris = max(0, min(image.shape[0] - 10, iris_center[1] - iris_radius - 10))
        cv2.putText(image, 'Iris', (text_x_iris, text_y_iris),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw the pupil
        cv2.circle(image, pupil_center, pupil_radius, (0, 0, 255), 2)

        # Ensure text position is within the image bounds for pupil
        text_x_pupil = max(0, min(image.shape[1] - 30, pupil_center[0] - 30))
        text_y_pupil = max(0, min(image.shape[0] - 10, pupil_center[1] - pupil_radius - 10))
        cv2.putText(image, 'Pupil', (text_x_pupil, text_y_pupil),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        print("Error: No iris circles were found")

    # Display the result
    plt.figure(figsize=(8, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Iris and Pupil Segmentation')
    plt.axis('off')
    plt.show()


# Path to the input image
image_path = 'images 1.jpeg'  # Replace with the path to your image

# Perform iris and pupil segmentation
iris_pupil_segmentation(image_path)
