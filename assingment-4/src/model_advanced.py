import os
import cv2
from skimage.feature import hog
from skimage.feature import local_binary_pattern
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
# The base models used in the slides' ensemble example:
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# ----------------------------------------------------------------------
# 1. Setup: Create Dummy Data for demonstration
#    (Replace this section with the output from your preprocessing function)
# ----------------------------------------------------------------------
# A dataset of 144 total samples (18 persons * 8 expressions).
# Assuming a feature vector size of 100 for each image.
def load_data_fer():
    X = []
    y = []
    data_dir = "../data/preprocessed_advanced_fer"
    for i in range(0, 19):
        folder = os.path.join(data_dir, str(i))
        for filename in os.listdir(folder):
            if filename.endswith(".jpg"):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                features = img.flatten()
#                features = extract_hlbp(img)
                features = extract_hog(img)
                X.append(features)
                label = os.path.splitext(filename)[0]
                y.append(label)

    return np.array(X), np.array(y)


def load_data_vtr():
    X = []
    y = []

    data_dir = "../data/preprocessed_advanced_vtr/r7bthvstxw-1"
    for vehicle_type in os.listdir(data_dir):
        folder = os.path.join(data_dir, vehicle_type)
        for filename in os.listdir(folder):
            if filename.endswith(".jpg"):
                img_path = os.path.join(folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#                features = img.flatten()
#                features = extract_hlbp(img)
                features = extract_hog(img)
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


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# 2. Model Script (Code from Slide 8)
# ----------------------------------------------------------------------

# Define the two base models used in the ensemble
# model1 = LogisticRegression(random_state=1) [cite: 906]
def train_voting_classifier():

    task = 0
    while task != 1 and task != 2:
        task = int(input("1 to train model on facial emotion data set, 2 for vehicle type data set: "))
    if task == 1:
        X, y = load_data_fer()
    else:
        X, y = load_data_vtr()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model1 = LogisticRegression(random_state=1, max_iter=1000)

    # model2 = tree.DecisionTreeClassifier(random_state=1) [cite: 906]
    model2 = DecisionTreeClassifier(random_state=1)

    # Create the Voting Classifier model
    # voting='hard' means Max Voting (winner-takes-all) 
    model = VotingClassifier(
        estimators=[('lr', model1), ('dt', model2)], 
        voting='hard'
    )

    # Train the ensemble model
    # model.fit(x_train,y_train) [cite: 908]
    print("\nFitting Voting Classifier...")
    model.fit(X_train, y_train)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # Evaluate the model
    # model.score(x_test,y_test) [cite: 909]
    accuracy = model.score(X_test, y_test)
    print(f"Model Score (Accuracy): {accuracy:.4f}")

# You can also make predictions:
# predictions = model.predict(X_test)
# print(f"First 5 predictions: {predictions[:5]}")

if __name__ == "__main__":
    train_voting_classifier()