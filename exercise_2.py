import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def dft_coefficients(image_fourier_transform):
    # Shift the spectrum to center the low frequencies
    shifted_spectrum = np.fft.fftshift(np.abs(image_fourier_transform))
    
    # Extract the phase spectrum
    phase_spectrum = np.angle(image_fourier_transform)
    
    # Display the original image, magnitude spectrum, and phase
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.log(1 + shifted_spectrum), cmap='gray')
    plt.title('Magnitude Spectrum')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(phase_spectrum, cmap='gray')
    plt.title('Phase Spectrum')
    plt.axis('off')
    
    plt.show()

def reconstruct_image(fourier_transform, percentage):
    # Calculate the number of coefficients to keep based on the given percentage
    num_coeffs_x = int(percentage * fourier_transform.shape[0] / 100)
    num_coeffs_y = int(percentage * fourier_transform.shape[1] / 100)

    # Crop the Fourier transform based on the calculated coefficients
    cropped_transform = fourier_transform[:num_coeffs_x, :num_coeffs_y]

    # Pad the cropped transform to the original size
    padded_transform = np.pad(cropped_transform, ((0, fourier_transform.shape[0] - num_coeffs_x), 
                                                 (0, fourier_transform.shape[1] - num_coeffs_y)),
                             mode='constant', constant_values=0)

    # Reconstruct the image using inverse Fourier transform
    reconstructed_image = np.fft.ifft2(np.fft.ifftshift(padded_transform)).real

    return reconstructed_image

# Load the image and convert it to grayscale
image_path = "images-project-1/cameraman.bmp"
image = np.array(Image.open(image_path).convert("L"))

# Compute the 2D Fourier transform of the image
fourier_transform = np.fft.fft2(image)

# Display the DFT coefficients (original image, magnitude spectrum,     and phase spectrum)
dft_coefficients(fourier_transform)

# Define different percentages of coefficients to keep for reconstruction
percentages = [20, 40, 60, 80]

# Display reconstructed images for different percentages
plt.figure(figsize=(15, 10))
for i, percentage in enumerate(percentages, 1):
    # Reconstruct the image using the specified percentage of coefficients
    reconstructed_image = reconstruct_image(fourier_transform, percentage)
    
    # Calculate and display the Mean Squared Error (MSE)
    mse = np.mean((image - reconstructed_image) ** 2)
    
    plt.subplot(2, 2, i)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f'Reconstruction with {percentage}% Coefficients\nMSE = {mse:.2f}')
    plt.axis('off')

plt.show()