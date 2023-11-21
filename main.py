import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# Set random seed for reproducibility
torch.manual_seed(42)

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# Load the training set
training_set = ImageFolder('/path/to/training_set', transform=transform)
train_loader = DataLoader(training_set, batch_size=32, shuffle=True)

# Load the test set
testing_set = ImageFolder('/path/to/test_set', transform=transform)
test_loader = DataLoader(testing_set, batch_size=32, shuffle=False)

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = nn.functional.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Initialize the model, loss function, and optimizer
cnn = CNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

# Train the model
num_epochs = 25
for epoch in range(num_epochs):
    cnn.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels.float().view(-1, 1))
        loss.backward()
        optimizer.step()

# Test the model
cnn.eval()
all_predictions = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = cnn(images)
        predictions = (outputs >= 0.5).float()
        all_predictions.extend(predictions.numpy())
        all_labels.extend(labels.numpy())

# Convert predictions and labels to numpy arrays
all_predictions = np.array(all_predictions).flatten()
all_labels = np.array(all_labels).flatten()

# Confusion Matrix using PyTorch
conf_matrix_torch = torch.tensor(confusion_matrix(all_labels, all_predictions))
print(conf_matrix_torch)

# Accuracy Score using PyTorch
accuracy_torch = accuracy_score(all_labels, all_predictions)
print("Accuracy:", accuracy_torch)

# Plotting using Matplotlib and Seaborn
plt.figure(figsize=(6, 6))
sns.heatmap(conf_matrix_torch, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
