import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import random
from sklearn.metrics.pairwise import cosine_similarity

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
# for image_path in images_path:
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     intensity_histogram = compute_normalized_intensity_histogram(image)
#     lbp_histogram = compute_normalized_lbp_histogram(image)

#     plt.figure(1)
#     plt.subplot(2, 1, 1)
#     plt.plot(intensity_histogram, color='gray')
#     plt.title(f'Normalized Intensity Histogram - {os.path.basename(image_path)}')
#     plt.subplot(2, 1, 2)
#     plt.plot(lbp_histogram, color='gray')
#     plt.title(f'Normalized LBP Histogram - {os.path.basename(image_path)}')

#     subdirectory_name = 'a_histograms'
#     subdirectory_path = os.path.join(os.getcwd(), subdirectory_name)
#     os.makedirs(subdirectory_path, exist_ok=True)
#     histogram_filename = os.path.splitext(os.path.basename(image_path))[0] + '_histograms.png'
#     histogram_filepath = os.path.join(subdirectory_path, histogram_filename)
#     plt.savefig(histogram_filepath)
#     print(f"Histograms for {image_path} saved ({subdirectory_name}/{histogram_filename})")

# B
# csv_filename = 'b_images_distances.csv'
# with open(csv_filename, 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
    
#     csv_writer.writerow(['Image1', 'Image2', 'L1 Distance', 'L2 Distance'])
    
#     for i, image_path1 in enumerate(images_path):
#         for j, image_path2 in enumerate(images_path):
#             if i != j:
#                 image1 = cv2.imread(os.path.join(image_path1), cv2.IMREAD_GRAYSCALE)
#                 image2 = cv2.imread(os.path.join(image_path2), cv2.IMREAD_GRAYSCALE)
                
#                 l1, l2 = calculate_distances(image1, image2)                
#                 csv_writer.writerow([f"{image_path1}", f"{image_path2}", l1, l2])
#                 print(f"Distance between Image {image_path1} and Image {image_path2} - L1: {l1}, L2: {l2}")

# C
# Ορίστε τυχαία 5 εικόνες από διαφορετικές κατηγορίες
images = os.listdir(folder_path)
categories = set()
random_images = []

for image in images:
    category = image.split('_')[0]
    categories.add(category)

categories = list(categories)

for category in categories:
    category_images = [image for image in images if image.startswith(category + '_')]
    selected_images = random.sample(category_images, min(5, len(category_images)))
    random_images.extend(selected_images)

random_images = ['orchids_00033.jpg', 'orchids_00029.jpg', 'orchids_00048.jpg', 'orchids_00037.jpg', 'orchids_00045.jpg', 'hydrangeas_00028.jpg', 'hydrangeas_00017.jpg', 'hydrangeas_00043.jpg', 'hydrangeas_00005.jpg', 'hydrangeas_00013.jpg', 'daisies_00080.jpg', 'daisies_00025.jpg', 'daisies_00088.jpg', 'daisies_00036.jpg', 'daisies_00043.jpg', 'lilies_00002.jpg', 'lilies_00017.jpg', 'lilies_00066.jpg', 'lilies_00059.jpg', 'lilies_00054.jpg', 'gardenias_00011.jpg', 'gardenias_00010.jpg', 'gardenias_00069.jpg', 'gardenias_00067.jpg', 'gardenias_00071.jpg', 'hibiscus_00018.jpg', 'hibiscus_00007.jpg', 'hibiscus_00077.jpg', 'hibiscus_00067.jpg', 'hibiscus_00047.jpg', 'tulip_00049.jpg', 'tulip_00079.jpg', 'tulip_00017.jpg', 'tulip_00012.jpg', 'tulip_00063.jpg', 'bougainvillea_00003.jpg', 'bougainvillea_00021.jpg', 'bougainvillea_00057.jpg', 
'bougainvillea_00019.jpg', 'bougainvillea_00009.jpg', 'peonies_00013.jpg', 'peonies_00018.jpg', 'peonies_00009.jpg', 'peonies_00024.jpg', 'peonies_00043.jpg', 'garden_roses_00065.jpg', 'garden_roses_00077.jpg', 'garden_roses_00051.jpg', 'garden_roses_00030.jpg', 'garden_roses_00012.jpg']

# Εκτύπωση των τυχαίων εικόνων
for i, image_path in enumerate(random_images):
    print(f"Random Image {i+1}: {image_path}")

# # Εκτύπωση των top-10 αποτελεσμάτων για όλους τους δυνατούς συνδυασμούς
# for i, query_image_path in enumerate(random_images):
#     print(f"\nQuery Image {i+1}: {query_image_path}")

#     # Φόρτωση της query image
#     query_image = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)

#     # Υπολογισμός των εικόνων που δεν είναι η query image
#     train_images = [image_path for image_path in random_images if image_path != query_image_path]

#     # Κατασκευή των features για την εκπαίδευση (χρησιμοποιήστε τα κατάλληλα features)
#     features = []
#     for image_path in train_images:
#         train_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#         feature = compute_normalized_intensity_histogram(train_image).flatten()
#         features.append(feature)

#     # Κατασκευή του feature για την query image
#     query_feature = compute_normalized_intensity_histogram(query_image).flatten()

#     # Εκπαίδευση του μοντέλου Nearest Neighbors
#     neighbors_model = NearestNeighbors(n_neighbors=10, metric='cosine')  # Μετρική cosine similarity
#     neighbors_model.fit(features)

#     # Υπολογισμός των top-10 αποτελεσμάτων
#     distances, indices = neighbors_model.kneighbors([query_feature])

#     # Εκτύπωση των top-10 αποτελεσμάτων
#     print("Top-10 Retrieval Results:")
#     for j, index in enumerate(indices.flatten()):
#         result_image_path = train_images[index]
#         print(f"{j+1}. {result_image_path} - Cosine Similarity: {1 - distances.flatten()[j]}")