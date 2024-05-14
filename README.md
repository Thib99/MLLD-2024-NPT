# MLLD-2024-NPT

**Authors:**
- Pablo Garcia
- Thibault Poux
- Nabil Kaci

**Summary:**
In this project, our aim is to develop a robust machine learning model to predict whether a person has brain cancer or not, based on MRI images. We start by preprocessing the images using the Sobel filter to remove non-meaningful information. We experiment with various machine learning models taught in class and optimize their parameters to achieve the best possible results. Additionally, we implement a Convolutional Neural Network (CNN) model using TensorFlow to further enhance our predictive performance.

**Database Information:**
- [Link to the database]
- Number of images with tumor: 155
- Number of images without tumor: 98

**Installation:**
To install all the required dependencies needed to run the project, execute the following command:

        pip install -r requirements.txt

**How to Use the Project:**
- Different parts of the project, such as preprocessing and model setup, are organized into separate packages.
- To get an overview of our project, execute and check the `main.ipynb` file, which provides insights into all the models used along with graphical representations to help identify the best model for our problem.

**Attention:**
- Execute the first cell of the project initially to load all dependencies and images from our database.
- Some cells require data from others to run, so if you encounter any issues while running the code, try running the cell above.
- Note that the cells calculating the heatmaps are computationally expensive and may take about 30 minutes to run.

Feel free to reach out if you have any questions or need further clarification!
