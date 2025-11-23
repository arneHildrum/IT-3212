import os
import cv2
import numpy as np


def preprocess_basic(image_path, save_dir, filename):
    """
    Preprocessing for a basic ML model
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (64, 64))                                     # Resize image to standardize image dimension, try different sizes during testing
    img_normalized = img_resized.astype('float32') / 255.0
    mean, std = img_normalized.mean(), img_normalized.std()
    if std > 0:
        img_standardized = (img_normalized - mean) / std
    else:
        img_standardized = img_normalized
    

    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, (img_standardized * 255).astype(np.uint8))



def preprocess_advanced(image_path, save_dir, filename):
    """
    Preprocessing for an advanced ML model
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (128, 128))                                   # Resize image to standardize image dimension, try different sizes during testing
    img_normalized = img_resized.astype('float32') / 255.0
    mean, std = img_normalized.mean(), img_normalized.std()
    if std > 0:
        img_standardized = (img_normalized - mean) / std
    else:
        img_standardized = img_normalized

    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, (img_standardized * 255).astype(np.uint8))


def preprocess_CNN(image_path, save_dir, filename):
    """
    Preprocessing for a CNN model
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img, (128, 128))                                   # Resize image to standardize image dimension, try different sizes during testing
    img_normalized = img_resized.astype('float32') / 255.0

    if np.random.rand() > 0.5:
        img_normalized = cv2.flip(img_normalized, 1)                            # horizontal flip
    angle = np.random.uniform(-10, 10)
    h, w = img_normalized.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    img_augmented = cv2.warpAffine(img_normalized, M, (w, h))
 
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, (img_augmented * 255).astype(np.uint8))


def main():
    task = 100
    while task != 0:
        task = int(input("0 to stop, 1 for basic fer, 2 for advanced fer, 3 for CNN fer, 4 for basic vtr, 5 for advanced vtr, 6 for CNN vtr...\nEnter task number (0-6): "))
        if task == 0:
            print("Shutting down program.......")
        #else:
        #    print("\nProcessing...\n")
        #    for i in range(0, 18):
        #        path = f"..\\data\\images\\{i}"
        #        if not os.path.exists(path):
        #            continue
        #        for filename in os.listdir(path):
        #            if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
        #                image_path = os.path.join(path, filename)
        #                match task:
        #                    case 1:
        #                        preprocess_basic(image_path)
        #                    case 2:
        #                        preprocess_advanced(image_path)
        #                    case 3:
        #                        preprocess_CNN(image_path)
        #                    case _:
        #                        print("Invalid task number.")
        #    print("\nProcessing completed.\n")
        else:
            match task:
                case 1:
                    path = "..\\data\\preprocessed_basic_fer"
                    if not os.path.exists(path):
                            os.makedirs(path, exist_ok=True)
                    for i in range(0, 19):
                        subpath = f"..\\data\\preprocessed_basic_fer\\{i}"
                        if not os.path.exists(subpath):
                            os.makedirs(subpath, exist_ok=True)
                        source_path = f"..\\data\\images\\{i}"
                        for filename in os.listdir(source_path):
                            if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
                                image_path = os.path.join(source_path, filename)
                                print(image_path)
                                preprocess_basic(image_path, subpath, filename)
                case 2:
                    path = "..\\data\\preprocessed_advanced_fer"
                    if not os.path.exists(path):
                            os.makedirs(path, exist_ok=True)
                    for i in range(0, 19):
                        subpath = f"..\\data\\preprocessed_advanced_fer\\{i}"
                        if not os.path.exists(subpath):
                            os.makedirs(subpath, exist_ok=True)
                        source_path = f"..\\data\\images\\{i}"
                        for filename in os.listdir(source_path):
                            if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
                                image_path = os.path.join(source_path, filename)
                                preprocess_advanced(image_path, subpath, filename)
                case 3:
                    path = "..\\data\\preprocessed_CNN_fer"
                    if not os.path.exists(path):
                            os.makedirs(path, exist_ok=True)
                    for i in range(0, 19):
                        subpath = f"..\\data\\preprocessed_CNN_fer\\{i}"
                        if not os.path.exists(subpath):
                            os.makedirs(subpath, exist_ok=True)
                        source_path = f"..\\data\\images\\{i}"
                        for filename in os.listdir(source_path):
                            if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
                                image_path = os.path.join(source_path, filename)
                                preprocess_CNN(image_path, subpath, filename)
                case 4:
                    path = "..\\data\\preprocessed_basic_vtr"
                    if not os.path.exists(path):
                        os.makedirs(path, exist_ok=True)
                        
                    subpath = os.path.join(path, "r7bthvstxw-1")
                    if not os.path.exists(subpath):
                        os.makedirs(subpath, exist_ok=True)
                        
                    source_path = f"..\\data\\r7bthvstxw-1"
                    if not os.path.exists(source_path):
                        print(f"Source directory not found: {source_path}")
                    else:
                        for folder in os.listdir(source_path):
                            sub_source_path = os.path.join(source_path, folder)
                            subsubpath = os.path.join(subpath, folder)
                            if not os.path.exists(subsubpath):
                                os.makedirs(subsubpath, exist_ok=True)
                            for filename in os.listdir(sub_source_path):
                                if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
                                    image_path = os.path.join(sub_source_path, filename)
                                    preprocess_basic(image_path, subsubpath, filename)

                case 5:
                    path = "..\\data\\preprocessed_advanced_vtr"
                    if not os.path.exists(path):
                        os.makedirs(path, exist_ok=True)
                        
                    subpath = os.path.join(path, "r7bthvstxw-1")
                    if not os.path.exists(subpath):
                        os.makedirs(subpath, exist_ok=True)
                        
                    source_path = f"..\\data\\r7bthvstxw-1"
                    if not os.path.exists(source_path):
                        print(f"Source directory not found: {source_path}")
                    else:
                        for folder in os.listdir(source_path):
                            sub_source_path = os.path.join(source_path, folder)
                            subsubpath = os.path.join(subpath, folder)
                            if not os.path.exists(subsubpath):
                                os.makedirs(subsubpath, exist_ok=True)
                            for filename in os.listdir(sub_source_path):
                                if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
                                    image_path = os.path.join(sub_source_path, filename)
                                    preprocess_advanced(image_path, subsubpath, filename)
                            
                case 6:
                    path = "..\\data\\preprocessed_CNN_vtr"
                    if not os.path.exists(path):
                        os.makedirs(path, exist_ok=True)
                        
                    subpath = os.path.join(path, "r7bthvstxw-1")
                    if not os.path.exists(subpath):
                        os.makedirs(subpath, exist_ok=True)
                        
                    source_path = f"..\\data\\r7bthvstxw-1"
                    if not os.path.exists(source_path):
                        print(f"Source directory not found: {source_path}")
                    else:
                        for folder in os.listdir(source_path):
                            sub_source_path = os.path.join(source_path, folder)
                            subsubpath = os.path.join(subpath, folder)
                            if not os.path.exists(subsubpath):
                                os.makedirs(subsubpath, exist_ok=True)
                            for filename in os.listdir(sub_source_path):
                                if filename.lower().endswith(".jpg") or filename.lower().endswith(".png"):
                                    image_path = os.path.join(sub_source_path, filename)
                                    preprocess_CNN(image_path, subsubpath, filename)
                
                case _:
                    print("Invalid task number.")

                    

if __name__ == "__main__":
    main()
