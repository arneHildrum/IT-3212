import data_finder
import class_CNN
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")                                   # Const for whether to use GPU is available


def main():
    data = data_finder.DataFinder()
    task = 0


    while task not in [1, 2]:
        task = int(input("1 = Facial emotion dataset, 2 = Vehicle dataset: "))

    # Load dataset
    if task == 1:
        X, y = data.load_data_fer("../data/preprocessed_CNN_fer")
        EPOCHS = 20
        BATCH_SIZE = 32
    else:    
        X, y = data.load_data_vtr("../data/preprocessed_CNN_vtr/r7bthvstxw-1")
        EPOCHS = 30
        BATCH_SIZE = 128

#    scaler = StandardScaler()
#    X = scaler.fit_transform(X)
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)

    if X.ndim == 2:
        N, D = X.shape
        if int(np.sqrt(D)) ** 2 == D:
            H = W = int(np.sqrt(D))
            X = X.reshape(N, 1, H, W)
        else:
            print("X contains HOG/LBP/non-square features — cannot reshape into images.")
    elif X.ndim == 3:
        # Grayscale images (N, H, W) → (N, 1, H, W)
        N, H, W = X.shape
        X = X.reshape(N, 1, H, W)
    elif X.ndim == 4:
        # Color images already in (N, H, W, C) → transpose to (N, C, H, W)
        N, H, W, C = X.shape
        X = X.transpose(0, 3, 1, 2)
        
    channels = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)

    # Convert labels → class indices
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y_train_tensor = torch.tensor([class_to_idx[i] for i in y_train], dtype=torch.long)
    y_test_tensor  = torch.tensor([class_to_idx[i] for i in y_test], dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                    batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test_tensor, y_test_tensor),
                    batch_size=BATCH_SIZE, shuffle=False)

    model = class_CNN.CNN(input_shape=(channels, H, W), num_classes=num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, criterion, optimizer, EPOCHS)
    evaluate(model, test_loader)


def train(model, train_loader, criterion, optimizer, EPOCHS):
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X_batch)

            # Convert one-hot → class indices
            y_labels = y_batch.long()

            loss = criterion(outputs, y_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")


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
            labels = y_batch.long()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"\nTest Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()