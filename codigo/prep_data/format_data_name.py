# take all the image in the dolder Dataset/no and Dataset/yes and rename them to no_1.jpg, no_2.jpg, yes_1.jpg, yes_2.jpg, etc.

import os
import shutil

def rename_images(folder_path, pattern_new_name, folder_output):
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
            shutil.move(file_path, new_path)
            count += 1

# Specify the folder paths
no_folder_path = "MLLD-2024-NPT/Dataset/no"
yes_folder_path = "MLLD-2024-NPT/Dataset/yes"

# Rename images in the 'no' folder
rename_images(no_folder_path, "no_", "MLLD-2024-NPT/Dataset/no")

# Rename images in the 'yes' folder
rename_images(yes_folder_path, "yes_", "MLLD-2024-NPT/Dataset/yes")
