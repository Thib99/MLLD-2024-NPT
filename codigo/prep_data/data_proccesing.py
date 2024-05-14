import numpy as np
import glob
import cv2


path_No = 'Dataset/no/*'
path_Yes = 'Dataset/yes/*'


def load_data():
    data, labels = [], []
    
    # Load 'yes' images
    for file in glob.iglob('Dataset/yes/*.jpg'):
        print("Processing file:", file)  # Add this line for debugging
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))  # Resize to (128, 128)
        data.append(img)
        labels.append(1)  # Label 1 for 'yes'
    
    # Load 'no' images
    for file in glob.iglob('Dataset/no/*.jpg'):
        print("Processing file:", file)  # Add this line for debugging
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))  # Resize to (128, 128)
        data.append(img)
        labels.append(0)  # Label 0 for 'no'
    
    # Convert lists to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    # Reshape data to have shape (num_samples, height, width, channels)
    data = np.expand_dims(data, axis=-1)  # Add a single channel dimension
    
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
    