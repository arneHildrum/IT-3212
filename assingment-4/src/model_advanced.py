import os
import cv2
from skimage.feature import hog
from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
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
def train_voting_classifier(n_components=0.95):

    task = 0
    while task != 1 and task != 2:
        task = int(input("1 to train model on facial emotion data set, 2 for vehicle type data set: "))
    if task == 1:
        X, y = load_data_fer()
    else:
        X, y = load_data_vtr()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if n_components != 0: pca = PCA(n_components=n_components, random_state=42)
    else:                 pca = PCA(random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    model1 = LogisticRegression(random_state=1, max_iter=1000)

    model2 = DecisionTreeClassifier(random_state=1)

    model3 = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        random_state=1
    )

    model4 = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=1
    )

    model5 = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance"
    )

    model6 = MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=500,
        random_state=1
    )

    model = VotingClassifier(
        estimators=[
#            ('lr', model1), 
#            ('dt', model2), 
            ('svm', model3),
            ('rf', model4),
            ('knn', model5),
#            ('mlp', model6)
        ], 
        voting='soft'
    )
    
    print("\nFitting Voting Classifier...")
    model.fit(X_train_pca, y_train)

    print(f"Training samples: {len(X_train_pca)}")
    print(f"Testing samples: {len(X_test_pca)}")

    #print(classification_report(y_test, y_pred))
    #print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

    accuracy = model.score(X_test_pca, y_test)
    print(f"Model Score (Accuracy): {accuracy * 100:.4f}%")

# You can also make predictions:
# predictions = model.predict(X_test)
# print(f"First 5 predictions: {predictions[:5]}")

def main():
    component = int(input("PCA components? 0 for none, write percentage (e.g., 95): "))
    train_voting_classifier(component / 100 if component != 0 else 0)


if __name__ == "__main__":
    main()