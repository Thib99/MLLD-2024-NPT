# take all the image in the dolder Dataset/no and Dataset/yes and rename them to no_1.jpg, no_2.jpg, yes_1.jpg, yes_2.jpg, etc.

import os
import shutil

import cv2
import numpy as np



def process_folder(folder_path, pattern_new_name, folder_output):
    """
    Process the images in the specified folder by renaming them and resizing them based on the bounding box of the brain.
    
    Args:
    - folder_path: Path of the folder containing the images to be processed.
    - pattern_new_name: Pattern for the new names of the images.
    - folder_output: Path of the folder where the processed images will be saved.
    """
    
    
    
    # delete the folder if it already exists and create a new one
    if os.path.exists(folder_output):
        shutil.rmtree(folder_output)
    os.makedirs(folder_output)
    
    # List all files in the folder
    files = os.listdir(folder_path)
    
    # Initialize counters for naming
    count = 1
    
    # Iterate through each file
    for file in files:
        # Get the full path of the file
        file_path = os.path.join(folder_path, file)
        
        # Check if the file is a regular file
        if os.path.isfile(file_path):
            # Rename the file
            new_path  = os.path.join(folder_output, pattern_new_name + str(count) + ".jpg")
            
            # rezise the image based on the outbouding of the brain
            resize_images_based_on_outbound(file_path, new_path) 
            count += 1


def resize_images_based_on_outbound(file_name, file_name_output):
    """
    Resize the image based on the bounding box of the brain.
    The function detect the outbounds in the image thank's to the Sobel filter and only keep the part of the image inside the outbounds.
    The image is then resized to the specified size_output, to have a consistent size for all images.
    
    Args:
    - file_name: Path of the image to be resized.
    - file_name_output: Path of the resized image.
    """
    
    # load the image in gray scale
    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    # Convolution Matrix (Sobel Filter)
    kernel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    kernel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    
    # aply the filter to the image
    edges_x = cv2.filter2D(image, -1, kernel_x)
    edges_y = cv2.filter2D(image, -1, kernel_y)
    
    # combined the axis
    edges = cv2.addWeighted(edges_x, 0.5, edges_y, 0.5, 0)

    # only keep the edges that are above a certain threshold
    _, thresholded = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)

    # find all the contours taht can be found in the image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    all_contours_points = []

    # get all the points of the contours
    for contour in contours:
        for point in contour:
            all_contours_points.append(point[0])

    # convert the list of points to a numpy array
    all_contours_points = np.array(all_contours_points)

    # Find the bounding rectangle of al the countours points 
    x, y, w, h = cv2.boundingRect(all_contours_points)

    # keep this code to draw the rectangle on the image, to show how we clean up the image
    # # Dessiner le rectangle englobant sur l'image d'origine
    # cv2.rectangle(image, (x, y), (x+w, y+h), color=(0, 0, 0), thickness=10)  # Le rouge est représenté par (0, 0, 255)
    
    
    image = image[y:y+h, x:x+w] #only keep the part of the image that contains the brain
    
    
    # the size of the output image
    background_w = size_output[1]
    background_h = size_output[0]
    
    new_w = background_w
    new_h = background_h
    
    # caculate the resized image of the brain, with the same ratio has the output image
    if w/h > background_w/background_h:
        new_h = int(h * background_w / w)
    else:
        new_w = int(w * background_h / h)
    
    # resize the image
    image = cv2.resize(image, (new_w, new_h))
    
    
    # put the image in the center of the output image
    image = put_in_center(image, (background_h, background_w))
    
    # save the new image 
    cv2.imwrite(file_name_output, image)
  
    
    
def put_in_center(data, array_shape):
    """
    Put data in the center of a NumPy array.

    Args:
    - data: Data to be placed in the center.
    - array_shape: Shape of the NumPy array.

    Returns:
    - centered_array: NumPy array with data placed in the center.
    """
    # get the average color of the background
    bg_color = int(np.mean([data[0][0], data[0][-1], data[-1][0], data[-1][-1]] ))
    
    # Create an array filled with the background image of the specified shape
    centered_array = np.full(array_shape, bg_color, dtype=data.dtype)

    # Calculate the starting indices for placing the data in the center
    start_indices = tuple((np.array(array_shape) - np.array(data.shape)) // 2)

    # Put the data in the center of the array
    centered_array[start_indices[0]:start_indices[0] + data.shape[0],
                   start_indices[1]:start_indices[1] + data.shape[1]] = data

    return centered_array


def run(force_run=False):
    if force_run or not os.path.exists("Dataset/no") or not os.path.exists("Dataset/yes"):
        # Specify the folder paths of input images
        no_folder_path = "Dataset/original_images/no"
        yes_folder_path = "Dataset/original_images/yes"
        no_folder_output = "Dataset/no"
        yes_folder_output = "Dataset/yes"
        
        # Process the folders
        process_folder(no_folder_path, "no_", no_folder_output)
        process_folder(yes_folder_path, "yes_", yes_folder_output)
    
################# LOCAL VARIABLES #################
# size of the output image 
size_output = (150, 128)
###################################################


if __name__ == "__main__":
   
   run(True)
