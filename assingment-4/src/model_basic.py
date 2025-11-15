import os
import cv2
import numpy as np
from sklearn.svm import SVC
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score


def load_data(data_dir):
    X = []
    y = []

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


def train_svm(kernel="rbf", gamma="scale", n_components=0.95):
    data_dir = "../data/preprocessed_basic"
    X, y = load_data(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if n_components != 0: pca = PCA(n_components=n_components, random_state=42)
    else:                     pca = PCA(random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    svm = SVC(
        kernel=kernel,
        C=0.1,
        gamma=gamma
    )
    param_grid = {
        'C': [0.1, 1, 10, 100],            # Regularization parameter
        'gamma': ['scale', 'auto', 0.1, 1], # Kernel coefficient
        'kernel': ['rbf', 'linear', 'poly']   # Kernel type
    }
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1, 
        verbose=2
    )
    grid_search.fit(X_train_pca, y_train)
#    svm.fit(X_train_pca, y_train)
    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test_pca)
#    y_pred = svm.predict(X_test_pca)

    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")


def main():
    component = int(input("PCA components? 0 for none, write percentage (e.g., 95): "))
    train_svm("rbf", "scale", component / 100 if component != 0 else 0)


if __name__ == "__main__":
    main()