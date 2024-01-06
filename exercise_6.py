import cv2
import numpy as np
import matplotlib.pyplot as plt

def mean_shift_segmentation(image, r=20, d=40, T=50):
    # Φόρτωση εικόνας
    image_cielab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    height, width, _ = image_cielab.shape

    # Δημιουργία ομοιόμορφου πλέγματος
    grid_x, grid_y = np.meshgrid(np.arange(r, width, r), np.arange(r, height, r))
    centers = np.vstack((grid_x.flatten(), grid_y.flatten())).T

    # Αρχικοποίηση μέσων διανυσμάτων mean-shift
    mean_shift_centers = np.zeros((len(centers), 2), dtype=np.float32)
    mean_shift_centers_values = []

    # Εκτέλεση αλγορίθμου mean-shift
    for iteration in range(T):
        for i, center in enumerate(centers):
            x, y = center
            roi = image_cielab[y - r:y + r, x - r:x + r].reshape(-1, 3)

            # Υπολογισμός μέσου διανύσματος mean-shift
            mean_shift_vector = np.mean(roi, axis=0)

            # Ενημέρωση μέσου διανύσματος mean-shift
            mean_shift_centers[i] = center + mean_shift_vector[:2]  # Προσθήκη των πρώτων δύο στοιχείων

            mean_shift_centers_values.append(mean_shift_centers[i])

    # Συνέχιση με τη δημιουργία μιας μάσκας τμηματοποίησης
    segmentation_mask = np.zeros_like(image_cielab, dtype=np.uint8)

    # Σημεία που ανήκουν σε διαφορετικές κλάσεις ανακατεύονται με διάφορα χρώματα
    for i, center in enumerate(mean_shift_centers_values):
        x, y = center.astype(int)  # Μετατροπή των δεικτών σε ακέραιους
        index = int((y // r) * (width // r) + (x // r))
        color_value = [i % 255, (2 * i) % 255, (3 * i) % 255]  # Διάφορα χρώματα
        segmentation_mask[y - r:y + r, x - r:x + r] = color_value

    return segmentation_mask

# Παράδειγμα χρήσης
image = cv2.imread('images-project-1/butterfly.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

segmentation_result = mean_shift_segmentation(image)
masked_image = cv2.bitwise_and(image, segmentation_result)

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(masked_image)
plt.title('Segmented Image')

plt.show()

