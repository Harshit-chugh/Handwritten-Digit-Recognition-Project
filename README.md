# Handwritten Digit Recognition

This is a machine learning based web application built using Python and Streamlit that allows users to upload images of handwritten digits and have them recognized by a pre-trained deep learning model. The model is trained on the MNIST dataset, which consists of 70,000 images of handwritten digits (60,000 for training and 10,000 for testing), with each image being 28x28 pixels in grayscale. The project includes two main components: training the CNN model and deploying the web application.

## Project Structure

The project consists of the following files:

- `app.py`: The main Streamlit application file that handles the user interface, file upload, image preprocessing, and digit prediction.
- `train_model.py`: A Python script for training the Convolutional Neural Network (CNN) model and saving it to a file (mnist_cnn_model.h5).
- `handwritten_digit_recognition.ipynb`: A Jupyter Notebook containing the code for experimenting with loading the MNIST dataset, data preprocessing, model building, training, and evaluation.
- `requirements.txt`: A text file listing the required Python packages and their versions.
