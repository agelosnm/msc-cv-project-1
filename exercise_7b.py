import cv2
import numpy as np
import matplotlib.pyplot as plt

# Διαβάζουμε την εικόνα
image = cv2.imread("images-project-1/butterfly_g.jpg")

# Υλοποίηση φίλτρου LoG
def LoG_filter(image, sigma):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.Laplacian(blurred, cv2.CV_64F)

# Υλοποίηση συνάρτησης για την εύρεση βέλτιστων παραμέτρων
def find_optimal_parameters(image):
    best_variance = 0
    best_threshold = 0
    max_edges = 0

    for variance in range(1, 10, 1):
        for threshold in range(1, 100, 5):
            edges = np.uint8(LoG_filter(image, variance))
            edges = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)[1]

            num_edges = np.sum(edges) / 255
            print(num_edges)

            if num_edges > max_edges:
                max_edges = num_edges
                best_variance = variance
                best_threshold = threshold

    return max_edges, best_variance, best_threshold

# Εύρεση βέλτιστων παραμέτρων
max_edges, best_variance, best_threshold = find_optimal_parameters(image)

# Εφαρμογή του LoG με τις βέλτιστες παραμέτρους
edges = np.uint8(LoG_filter(image, best_variance))
edges = cv2.threshold(edges, best_threshold, 255, cv2.THRESH_BINARY)[1]

# Εκτύπωση των βέλτιστων παραμέτρων
print(f"Best Variance: {best_variance}")
print(f"Best Threshold: {best_threshold}")
print(f"Edges found: {int(max_edges)}")

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(edges)
plt.title('Edges (LoG)')

plt.show()