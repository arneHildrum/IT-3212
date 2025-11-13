import os
import cv2
import numpy as np
import pandas as pd

def preprocess_basic(path):
    print(path, "was found")
    # gray = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)


def preprocess_advanced(path):
    print(path, "was found")
    # gray = cv2.cvtColor(path, cv2.COLOR_BGR2GRAY)


def preprocess_CNN(path):
    print(path, "was found")


def main():
    task = 100
    while(task != 0):
        task = int(input("0 to stop, 1 for basic model preprocessing, 2 for advanced, 3 for CNN...\n\nEnter task number (0-3): "))
        if task == 0:
            print("Shutting down program.......")
        else:
            print("\n\n")
            match task:
                case 1:
                    for i in range(0, 18):
                        path = f"..\\data\\images\\{i}"
                        for filename in os.listdir(path):
                            if filename.endswith(".jpg") or filename.endswith(".jpg"):
                                image_path = os.path.join(path, filename)
                                preprocess_basic(image_path)
                
                case 2:
                    for i in range(0, 18):
                        path = f"..\\data\\images\\{i}"
                        for filename in os.listdir(path):
                            if filename.endswith(".jpg") or filename.endswith(".jpg"):
                                image_path = os.path.join(path, filename)
                                preprocess_advanced(image_path)
                
                case 3:
                    for i in range(0, 18):
                        path = f"..\\data\\images\\{i}"
                        for filename in os.listdir(path):
                            if filename.endswith(".jpg") or filename.endswith(".jpg"):
                                image_path = os.path.join(path, filename)
                                preprocess_CNN(image_path)
                
                case _:
                    print("Invalid task number. Please enter a number between 0 and 3.")

if __name__ == "__main__":
    main()