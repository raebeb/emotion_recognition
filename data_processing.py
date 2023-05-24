import csv
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_images(image_paths, target_size):
    print("Loading and preprocessing images...")
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
        image = cv2.resize(image, (target_size, target_size))  # Resize image
        image = image.reshape(target_size, target_size, 1)  # Reshape to add channel dimension
        image = image.astype(np.float32)  # Convert pixel values to float32
        image /= 255.0  # Normalize pixel values to the range [0, 1]
        images.append(image)
    return np.array(images)

def read_csv_file(file_path):
    image_paths = []
    labels = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        # print(reader)
        next(reader)  # Skip the header row
        for row in reader:
            image_paths.append(row[0])
            labels.append(row[1])
    return image_paths, labels

# Example usage
train_data_file = 'train_data.csv'
test_data_file = 'test_data.csv'

train_image_paths, train_labels = read_csv_file(train_data_file)
test_image_paths, test_labels = read_csv_file(test_data_file)

# Convert labels to numerical format using LabelEncoder
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)

target_size = 128

train_images = load_and_preprocess_images(train_image_paths, target_size)
test_images = load_and_preprocess_images(test_image_paths, target_size)

import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        print(f"Images: {self.images[index]}")
        print(f"Labels: {self.labels[index]}")
        image = self.images[index]
        label = self.labels[index]
        if isinstance(image, tuple):
            image = image[0]
        if self.transform:
            image = self.transform(image)
        return image, label

# Convert labels to numerical format if necessary

# Convert images and labels to PyTorch tensors
train_images = torch.from_numpy(train_images)
train_labels = torch.tensor(train_labels)
test_images = torch.from_numpy(test_images)
test_labels = torch.tensor(test_labels)

# Create dataset objects
train_dataset = EmotionDataset(train_images, train_labels)
test_dataset = EmotionDataset(test_images, test_labels)
