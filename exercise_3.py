import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Α. Υλοποίηση DCT και IDCT
def dct2(block):
    return np.fft.fft2(block, norm="ortho")

def idct2(block):
    return np.fft.ifft2(block, norm="ortho")

# Β. Υπολογισμός και εκτύπωση φάσματος DCT
def plot_dct_spectrum(image):
    dct_image = dct2(image)
    spectrum = np.log(np.abs(dct_image) + 1)  # Λογαριθμική κλίμακα για καλύτερη ορατότητα

    plt.figure(figsize=(8, 8))
    plt.imshow(spectrum, cmap='gray')
    plt.title('DCT Spectrum')
    plt.colorbar()
    plt.show()

# Γ. Ανακατασκευή εικόνας από το 20%, 40%, 60%, 80% των συντελεστών
def reconstruct_image(image, percentage):
    dct_image = dct2(image)
    threshold = np.percentile(np.abs(dct_image), 100 - percentage)
    mask = np.abs(dct_image) > threshold
    dct_image_low_freq = dct_image * mask
    reconstructed_image = idct2(dct_image_low_freq).real
    return reconstructed_image

# Ανάκτηση της εικόνας "cameraman.bmp"
image_path = "images-project-1/cameraman.bmp"
image = np.array(Image.open(image_path).convert("L"))

# Βήμα Β: Υπολογισμός και εκτύπωση του φάσματος DCT
plot_dct_spectrum(image)

# Δ. Ανακατασκευή και εκτύπωση αποτελεσμάτων για το 20%, 40%, 60%, 80% των συντελεστών
percentages = [20, 40, 60, 80]

plt.figure(figsize=(15, 10))
for i, percentage in enumerate(percentages, 1):
    # Reconstruct the image using the specified percentage of coefficients
    reconstructed_image = reconstruct_image(image, percentage)
    
    # Calculate and display the Mean Squared Error (MSE)
    mse = np.mean((image - reconstructed_image) ** 2)
    
    plt.subplot(2, 2, i)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f'Reconstruction with {percentage}% Coefficients\nMSE = {mse:.2f}')
    plt.axis('off')

plt.show()
