import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from data_processing import EmotionDataset, train_images, train_labels, test_images, test_labels


class EmotionDetectionModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionDetectionModel, self).__init__()
        # Define your model architecture
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 64 * 64, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

device = torch.device("cpu")

num_classes = 7
learning_rate = 0.001
batch_size = 32
num_epochs = 10

unique_labels = torch.unique(train_labels)
print(f'unique labels {unique_labels}')
model = EmotionDetectionModel(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

transform = transforms.Compose([
    transforms.ToTensor(),
])

emotions = ['Anger', 'Disgust', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
for emotion in emotions:
    root_dir = f'dataset_resized/{emotion}'
    train_dataset = EmotionDataset(train_images, train_labels, transform=transform)
    test_dataset = EmotionDataset(test_images, test_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Training loop
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {accuracy:.2f}%")

# Save the model weights
torch.save(model.state_dict(), 'model_weights.pth')