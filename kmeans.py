from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from PIL import Image
import numpy as np
import os
import random
from collections import defaultdict

# Load images from Dataset1
image_dir = '/Dataset1/Image/train/'
image_files = os.listdir(image_dir)
images = []

for image_file in image_files:
    image = Image.open(os.path.join(image_dir, image_file))
    image = image.resize((128, 128))  # Resize the image
    image = np.array(image)
    images.append(image.flatten())  # Flatten the image

# Convert list to numpy array
images = np.array(images)

# Standardize features to have zero mean and unit variance
scaler = StandardScaler()
images = scaler.fit_transform(images)

# Initialize the KMeans model
kmeans = KMeans(n_clusters=2, random_state=42)

# Fit the model
kmeans.fit(images)

# Get the cluster assignments for each image
cluster_assignments = kmeans.labels_

# Now, `cluster_assignments[i]` is the cluster that `images[i]` belongs to.

images_by_cluster = defaultdict(list)
for i, cluster in enumerate(cluster_assignments):
    images_by_cluster[cluster].append(image_files[i])

# Determine the number of images to sample from each cluster
sample_size = min(len(images) for images in images_by_cluster.values())

# Sample an equal number of images from each cluster
sampled_images = {cluster: random.sample(images, sample_size) for cluster, images in images_by_cluster.items()}

def matchingMask(imageName):
    return imageName[:-4] + '.png'

img_directory = 'Dataset1/Image/train/'
msk_direcotry = 'Dataset1/Mask/train/'


# Write the sampled image paths to a file
with open('sampled_images.txt', 'w') as f:
    for cluster, images in sampled_images.items():
        for image in images:
            f.write(f'{img_directory}{image} {msk_direcotry}{matchingMask(image)}\n')