import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('images-project-1/cameraman.bmp', cv2.IMREAD_GRAYSCALE)

dft = np.fft.fft2(image)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = np.log(np.abs(dft_shift))
phase_spectrum = np.angle(dft_shift)

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")

plt.subplot(1, 3, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("Magnitude Spectrum")

plt.subplot(1, 3, 3)
plt.imshow(phase_spectrum, cmap='gray')
plt.title("Phase Spectrum")

plt.show()

percentages = [0.2, 0.4, 0.6, 0.8]
reconstructed_images = []
reconstructed_images_mse = []

for i, percentage in enumerate(percentages, 1):
    rows, cols = image.shape
    mask_rows = int(rows * percentage)
    mask_cols = int(cols * percentage)
    mask = np.zeros((rows, cols), dtype=np.uint8)
    mask[rows // 2 - mask_rows // 2: rows // 2 + mask_rows // 2,
         cols // 2 - mask_cols // 2: cols // 2 + mask_cols // 2] = 1

    masked_dft_shift = dft_shift * mask

    inverse_shifted = np.fft.ifftshift(masked_dft_shift)
    inverse_result = np.fft.ifft2(inverse_shifted)
    reconstructed_image = np.abs(inverse_result)

    mse = np.mean((image - reconstructed_image) ** 2)

    reconstructed_images.append(reconstructed_image)
    reconstructed_images_mse.append(mse)

plt.subplot(2, 2, 1)
plt.imshow(reconstructed_images[0], cmap='gray')
plt.title(f"{"Image with 20% Coefficients"}\nMSE: {reconstructed_images_mse[0]:.2f}")

plt.subplot(2, 2, 2)
plt.imshow(reconstructed_images[1], cmap='gray')
plt.title(f"{"Image with 40% Coefficients"}\nMSE: {reconstructed_images_mse[1]:.2f}")

plt.subplot(2, 2, 3)
plt.imshow(reconstructed_images[2], cmap='gray')
plt.title(f"{"Image with 60% Coefficients"}\nMSE: {reconstructed_images_mse[2]:.2f}")

plt.subplot(2, 2, 4)
plt.imshow(reconstructed_images[3], cmap='gray')
plt.title(f"{"Image with 80% Coefficients"}\nMSE: {reconstructed_images_mse[3]:.2f}")

plt.show()