import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import csv

os.chdir('exercise_8')

# Συνάρτηση για υπολογισμό κανονικοποιημένου ιστογράμματος φωτεινότητας
def compute_normalized_intensity_histogram(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    normalized_hist = hist / float(np.sum(hist))
    return normalized_hist

# Συνάρτηση για υπολογισμό κανονικοποιημένου ιστογράμματος χαρακτηριστικού υφής LBP
def compute_normalized_lbp_histogram(image):
    lbp = cv2.calcHist([image], [0], None, [256], [0, 256], accumulate=False)
    normalized_hist = lbp / float(np.sum(lbp))
    return normalized_hist

# Υπολογισμός L1 και L2 αποστάσεων μεταξύ δύο εικόνων
def calculate_distances(image1, image2):
    histogram1 = compute_normalized_intensity_histogram(image1)
    histogram2 = compute_normalized_intensity_histogram(image2)
    
    l1_distance = np.sum(np.abs(histogram1 - histogram2))
    l2_distance = np.sqrt(np.sum((histogram1 - histogram2)**2))
    
    return l1_distance, l2_distance

# Διαδρομή προς τον φάκελο που περιέχει τις εικόνες
folder_path = 'flowers'

# Λίστα για τις διαδρομές όλων των αρχείων εικόνων στον φάκελο
images_path = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if filename.endswith('.jpg')]

# A
for image_path in images_path:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    intensity_histogram = compute_normalized_intensity_histogram(image)
    lbp_histogram = compute_normalized_lbp_histogram(image)

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(intensity_histogram, color='gray')
    plt.title(f'Normalized Intensity Histogram - {os.path.basename(image_path)}')
    plt.subplot(2, 1, 2)
    plt.plot(lbp_histogram, color='gray')
    plt.title(f'Normalized LBP Histogram - {os.path.basename(image_path)}')

    subdirectory_name = 'a_histograms'
    subdirectory_path = os.path.join(os.getcwd(), subdirectory_name)
    os.makedirs(subdirectory_path, exist_ok=True)
    histogram_filename = os.path.splitext(os.path.basename(image_path))[0] + '_histograms.png'
    histogram_filepath = os.path.join(subdirectory_path, histogram_filename)
    plt.savefig(histogram_filepath)
    print(f"Histograms for {image_path} saved ({subdirectory_name}/{histogram_filename})")

# B
csv_filename = 'b_images_distances.csv'
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    csv_writer.writerow(['Image1', 'Image2', 'L1 Distance', 'L2 Distance'])
    
    for i, image_path1 in enumerate(images_path):
        for j, image_path2 in enumerate(images_path):
            if i != j:
                image1 = cv2.imread(os.path.join(image_path1), cv2.IMREAD_GRAYSCALE)
                image2 = cv2.imread(os.path.join(image_path2), cv2.IMREAD_GRAYSCALE)
                
                l1, l2 = calculate_distances(image1, image2)                
                csv_writer.writerow([f"{image_path1}", f"{image_path2}", l1, l2])
                print(f"Distance between Image {image_path1} and Image {image_path2} - L1: {l1}, L2: {l2}")