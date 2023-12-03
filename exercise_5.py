import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

def noise_image(image, mean=0, stg=0.01):
    img_array = np.array(image)
    noise = np.random.normal(mean, stg, img_array.shape)
    noisy_image = img_array + noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Clip values to the valid range [0, 255]
    noisy_image = noisy_image.astype(np.uint8)
    return Image.fromarray(noisy_image)

def butterworth_low_pass_filter(image_fft, order, cutoff_freq):
    M,N = image_fft.shape
    H = np.zeros((M,N), dtype=np.float32)
    D0 = cutoff_freq
    n = order

    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            H[u,v] = 1 / (1 + (D/D0)**n)
    
    filtered_image_fft = image_fft * H
    filtered_image = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_image_fft)))
    return filtered_image

image_path = 'images-project-1/lenna.bmp'
image = Image.open(image_path).convert('L')  # 'L' for grayscale

noisy_image = noise_image(image)

image_fft = np.fft.fftshift(np.fft.fft2(image))
# image_fft = np.log(np.abs()))
noisy_image_fft = np.fft.fftshift(np.fft.fft2(noisy_image))

filter_orders = [3, 5, 7]

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Noisy Image')

plt.show()

for i, filter_order in enumerate(filter_orders):
    filtered = butterworth_low_pass_filter(image_fft, filter_order, 10)
    plt.subplot(1, len(filter_orders), i + 1)
    plt.imshow(filtered, cmap='gray')
    plt.title(f'Filtered Image (Size {filter_order}x{filter_order})')

plt.show()