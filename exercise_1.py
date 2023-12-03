import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def uniform_quantizer(image, levels):
    quantized_image = np.floor(image / (256 / levels)) * (256 / levels)
    return quantized_image

def plot_transform_function(levels):
    x = np.arange(0, 256, 1)
    y = uniform_quantizer(x, levels)

    plt.plot(x, y, label=f'Levels = {levels}')

def mean_square_error(original, quantized):
    return np.mean((original - quantized) ** 2)

# Φορτώνουμε την εικόνα αποχρώσεων του γκρι
image_path = "images-project-1/cameraman.bmp"
image = np.array(Image.open(image_path).convert("L"))

# Σχεδιάζουμε τη συνάρτηση μετασχηματισμού για διάφορους κβαντιστές
plt.figure(figsize=(10, 6))

quantizer_levels = [7, 11, 15, 19]
for levels in quantizer_levels:
    plot_transform_function(levels)

plt.xlabel('Input Intensity')
plt.ylabel('Quantized Intensity')
plt.title('Quantization Transform Function for Uniform Quantizer')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(15, 10))

for i, levels in enumerate(quantizer_levels, 1):
    # Εφαρμόζουμε τον κβαντιστή
    quantized_image = uniform_quantizer(image, levels)
    
    # Υπολογίζουμε το μέσο τετραγωνικό σφάλμα κβάντισης
    mse = mean_square_error(image, quantized_image)
    
    # Εμφανίζουμε την εικόνα και τα αποτελέσματα
    plt.subplot(2, 2, i)
    plt.imshow(quantized_image, cmap='gray')
    plt.title(f'Quantizer Levels = {levels}\nMSE = {mse:.2f}')
    plt.axis('off')

plt.show()
