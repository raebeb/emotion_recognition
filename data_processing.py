import csv

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_images(image_paths, target_size):
    print("Loading and preprocessing images...")
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB format
        image = image.resize((target_size, target_size))  # Resize image
        image = np.array(image)  # Convert image to numpy array
        image = np.transpose(image, (2, 0, 1))  # Transpose dimensions to (channels, height, width)
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
    print(f'label: {labels}')
    return image_paths, labels

# Example usage
train_data_file = 'train_data.csv'
test_data_file = 'test_data.csv'

train_image_paths, train_labels = read_csv_file(train_data_file)
test_image_paths, test_labels = read_csv_file(test_data_file)

# Convert labels to numerical format if necessary
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels) + 1  # Add 1 to shift label values from 0-5 to 1-6
test_labels = label_encoder.transform(test_labels) + 1  # Add 1 to shift label values from 0-5 to 1-6


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
        image = self.images[index]
        try:
            label = self.labels[index]
        except IndexError:
            label = self.labels[int(len(self.labels) * np.random.random())]  # Random label to avoid index error

        if isinstance(image, tuple):
            image = image[0]

        if not isinstance(image, torch.Tensor):  # Skip transformation if already a tensor
            if self.transform:
                image = self.transform(image)

        return image, label


# Convert images and labels to PyTorch tensors
train_images = torch.from_numpy(train_images)
train_labels = torch.tensor(train_labels)
print(f'Train labels {train_labels}')
test_images = torch.from_numpy(test_images)
test_labels = torch.tensor(test_labels)

# Create dataset objects
train_dataset = EmotionDataset(train_images, train_labels)
test_dataset = EmotionDataset(test_images, test_labels)
