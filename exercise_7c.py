import cv2
import matplotlib.pyplot as plt

# Διαβάζουμε την εικόνα
image_path = "images-project-1/butterfly_g.jpg"
original_image = cv2.imread(image_path)

# Εφαρμόζουμε τη μέθοδο Canny
edges = cv2.Canny(original_image, 100, 200)  # Οι τιμές 100 και 200 είναι οι κατώτατη και ανώτατη παράμετροι του Canny

# Εμφανίζουμε την αρχική εικόνα
plt.subplot(121), plt.imshow(original_image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

# Εμφανίζουμε τις ακμές που προέκυψαν από τη μέθοδο Canny
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Canny Edges'), plt.xticks([]), plt.yticks([])

plt.show()
