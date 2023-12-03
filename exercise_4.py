import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

def noise_image(image):
    original_image_array = np.array(image)

    # Create a random array with the same dimensions as the image
    noise = np.random.rand(*original_image_array.shape)

    noise_density = 0.05

    # Create a mask for the positions where the noise will be added
    salt_and_pepper = noise < noise_density / 2

    # Add salt & pepper noise
    original_image_array[salt_and_pepper] = 0  # Salt
    original_image_array[noise > 1 - noise_density / 2] = 255  # Pepper

    # Convert the numpy array back to a PIL image
    noisy_image = Image.fromarray(original_image_array)

    return noisy_image

def filter_image(noisy_image, filter_size):    
    
    filtered_image = noisy_image.filter(ImageFilter.BoxBlur(filter_size))

    return filtered_image

# Image path
image_path = 'images-project-1/lenna.bmp'

# Read the original image
image = Image.open(image_path).convert('L')  # 'L' for grayscale

noisy_image = noise_image(image)

# Display the images using matplotlib
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')

plt.show()

filter_sizes = [3, 5, 7]

for i, filter_size in enumerate(filter_sizes):
    filtered = filter_image(noisy_image, filter_size)
    plt.subplot(1, len(filter_sizes), i + 1)
    plt.imshow(filtered, cmap='gray')
    plt.title(f'Filtered Image (Size {filter_size}x{filter_size})')

plt.show()