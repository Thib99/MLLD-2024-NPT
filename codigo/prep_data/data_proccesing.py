import os
import numpy as np
import glob
import cv2

path_No = 'Dataset/no/*'
path_Yes = 'Dataset/yes/*'

def load_data():
    tumor = []
    no_tumor = []

    for file in glob.iglob(path_Yes):
        img = cv2.imread(file)      #Reading the images from the path
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        #Changing the color from BGR to RGB
        img = cv2.resize(img, (128, 128)) 
        tumor.append((img, 1))  # Appending tuple with image and label 1 (indicating presence of tumor)

    for file in glob.iglob(path_No):
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))
        no_tumor.append((img, 0))  # Appending tuple with image and label 0 (indicating absence of tumor)

    # Concatenating the two lists and shuffle the data
    all_data = tumor + no_tumor


    # Splitting data and labels
    data = np.array([item[0] for item in all_data])
    labels = np.array([item[1] for item in all_data])

    return data, labels


def load_data_flatten() :

    data, labels = [], []
    
    # loop trough all the images in the dataset
    for file in glob.iglob('Dataset/*/*.jpg'):
        
        # get the image, resize it, make it grey scale, and flatten it to a 1D array
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = img.flatten()
        data.append(img)
        
        # put the corect label
        if 'no' in file:
            labels.append(0)
        else:
            labels.append(1)
    
    return np.array(data), np.array(labels)



# print some information about the data, when script executed
if __name__ == '__main__':
    pass
    