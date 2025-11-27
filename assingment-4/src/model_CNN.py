import os
import cv2
import data_loader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from skimage.feature import local_binary_pattern, hog

# -------------------------------------------------------
# Constants
# -------------------------------------------------------
CHANNELS = 3
NUM_CLASSES = 8
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")







class CNN(nn.Module):
        def __init__(self, num_classes=NUM_CLASSES):
            super(CNN, self).__init__()
            
            self.model = nn.Sequential(
                # Conv Block 1
                nn.Conv2d(CHANNELS, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Conv Block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                # Conv Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Flatten(),
                nn.Linear(128 * 16 * 16, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes),
                nn.Softmax(dim=1)
            )
        
        def forward(self, x):
            return self.model(x)








class DataFinder:
    def load_data_fer(data_dir, self):
        X = []
        y = []

        for i in range(0, 19):
            folder = os.path.join(data_dir, str(i))
            for filename in os.listdir(folder):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(folder, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    task = 0
                    while task != 1 and task != 2:
                        task = int(input("1 for HOG, 2 for HLBP, 3 for flattened: "))
                    match task:
                        case 1:     features = self.extract_hog(img)
                        case 2:     features = self.extract_hlbp(img)
                        case 3:     features = img.flatten()
                    X.append(features)
                    label = os.path.splitext(filename)[0]
                    y.append(label)

        return np.array(X), np.array(y)


    def load_data_vtr(data_dir, self):
        X = []
        y = []

        for vehicle_type in os.listdir(data_dir):
            folder = os.path.join(data_dir, vehicle_type)
            for filename in os.listdir(folder):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(folder, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    task = 0
                    while task != 1 and task != 2:
                        task = int(input("1 for HOG, 2 for HLBP, 3 for flattened: "))
                    match task:
                        case 1:     features = self.extract_hog(img)
                        case 2:     features = self.extract_hlbp(img)
                        case 3:     features = img.flatten()
                    X.append(features)
                    y.append(vehicle_type)
        
        return np.array(X), np.array(y)



    def extract_hog(img):
        """
        Extract HOG (Histogram of Oriented Gradients) features.
        Input:  grayscale image (numpy array)
        Output: 1D feature vector
        """
        features = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            transform_sqrt=True
        )
        return features


    def extract_hlbp(img, P=8, R=1, bins=256):
        """
        Extract HLBP (Histogram of Local Binary Patterns)
        Input:  grayscale image (numpy array)
        Output: 1D histogram feature vector
        """
        lbp = local_binary_pattern(img, P=P, R=R, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
        return hist












def main():
    
    data = DataFinder()
    task = 0
    while task != 1 and task != 2:
        task = int(input("1 to train model on facial emotion data set, 2 for vehicle type data set: "))
    if task == 1:
        X, y = data.load_data_fer("../data/preprocessed_CNN_vtr/r7bthvstxw-1")
    else:
        X, y = data.load_data_vtr("../data/preprocessed_CNN_vtr/r7bthvstxw-1")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Data loaders
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test_tensor, y_test_tensor),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Initialize model
    model = CNN().to(DEVICE)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, criterion, optimizer)
    evaluate(model, test_loader)


# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------
def train(model, train_loader, criterion, optimizer):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)

            # Convert one-hot → class indices
            y_labels = torch.argmax(y_batch, dim=1)

            loss = criterion(outputs, y_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")


# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)
            predicted = torch.argmax(outputs, dim=1)
            labels = torch.argmax(y_batch, dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"\nTest Accuracy: {100 * correct / total:.2f}%")
    

# -------------------------------------------------------
# Run training and evaluation
# -------------------------------------------------------
if __name__ == "__main__":
    main()