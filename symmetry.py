import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
    return binary

def detect_contours(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def reflect_points(points, axis, center):
    reflected_points = np.copy(points)
    if axis == 'vertical':
        reflected_points[:, 0] = 2 * center[0] - points[:, 0]
    elif axis == 'horizontal':
        reflected_points[:, 1] = 2 * center[1] - points[:, 1]
    else:
        raise ValueError("Axis must be 'vertical' or 'horizontal'")
    return reflected_points

def is_symmetric(points, axis):

    center = np.mean(points, axis=0)
    reflected_points = reflect_points(points, axis, center)
    return np.allclose(points, reflected_points, atol=1e-2)

def plot_image_with_symmetry(image_path, contours, axis, is_symmetric):
    img = cv2.imread(image_path)
    plt.figure()
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for contour in contours:
        plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'r-')
    center = np.mean(contours[0], axis=0)
    if axis == 'vertical':
        plt.axvline(x=center[0][0], color='g' if is_symmetric else 'b', linestyle='--')
    elif axis == 'horizontal':
        plt.axhline(y=center[0][1], color='g' if is_symmetric else 'b', linestyle='--')
    plt.title('Symmetry Detection')
    plt.show()

def main(image_path, axis):
    binary_image = preprocess_image(image_path)
    contours = detect_contours(binary_image)
    all_points = np.vstack(contours).squeeze()
    symmetric = is_symmetric(all_points, axis)
    plot_image_with_symmetry(image_path, contours, axis, symmetric)
    print(f'Symmetry along {axis}: {symmetric}')

image_path = '/path/to/image/'
axis = 'vertical'  # or 'horizontal'
main(image_path, axis)
