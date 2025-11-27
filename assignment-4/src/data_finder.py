import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog


class DataFinder:
    def load_data_fer(self, data_dir):
        X = []
        y = []
        task = 0
        while task not in [1, 3]:
            task = int(input("1 for HOG, 2 for HLBP, 3 for flattened: "))

        data_dir = os.path.normpath(data_dir)
        for i in range(0, 19):
            folder = os.path.join(data_dir, str(i))
            for filename in os.listdir(folder):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(folder, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    match task:
                        case 1:     features = self.extract_hog(img)
                        case 2:     features = self.extract_hlbp(img)
                        case 3:     features = img.flatten()
                    X.append(features)
                    label = os.path.splitext(filename)[0]
                    y.append(label)

        return np.array(X), np.array(y)


    def load_data_vtr(self, data_dir):
        X = []
        y = []
        task = 0
        while task not in [1, 3]:
            task = int(input("1 for HOG, 2 for HLBP, 3 for flattened: "))

        classes = sorted(os.listdir(data_dir))
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        print("Class mapping:", class_to_idx)

        for vehicle_type in os.listdir(data_dir):
            folder = os.path.join(data_dir, vehicle_type)
            for filename in os.listdir(folder):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(folder, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    match task:
                        case 1:     features = self.extract_hog(img)
                        case 2:     features = self.extract_hlbp(img)
                        case 3:     features = img
                    X.append(features)
                    y.append(class_to_idx[vehicle_type])
        
        return np.array(X), np.array(y)



    def extract_hog(self, img):
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
            transform_sqrt=True#,channel_axis=-1
        )
        return features


    def extract_hlbp(self, img, P=8, R=1, bins=256):
        """
        Extract HLBP (Histogram of Local Binary Patterns)
        Input:  grayscale image (numpy array)
        Output: 1D histogram feature vector
        """
        lbp = local_binary_pattern(img, P=P, R=R, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
        return hist