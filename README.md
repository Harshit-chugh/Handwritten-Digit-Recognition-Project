# Handwritten Digit Recognition

This is a machine learning based application built using Python and Gradio that allows users to draw images of handwritten digits and have them recognized by a pre-trained deep learning model. The model is trained on the MNIST dataset, which consists of 70,000 images of handwritten digits (60,000 for training and 10,000 for testing), with each image being 28x28 pixels in grayscale. The project includes two main components: training the CNN model and deploying the application.

## Project Structure

The project consists of the following files:

- `app.py`: The main VS code python file that handles the gradio interface, local URL, sketchpad input and digit prediction.
- `train_model.py`: A Python script for training the Convolutional Neural Network (CNN) model and saving it to a file (model.h5).
- `handwritten_digit_recognition.ipynb`: A Jupyter Notebook containing the code for experimenting with loading the MNIST dataset, data preprocessing, model building, training, and evaluation.
- `requirements.txt`: A text file listing the required Python packages and their versions.

## Training the Model
The train_model.py script performs the following tasks:

1. Data Loading and Preprocessing: Loads the MNIST dataset and preprocesses it for training.
2. Data Augmentation: Applies random transformations to the training data to improve the model's robustness.
3. Model Building: Constructs a CNN model using Keras 
4. Model Training: Trains the model using the Adam optimizer, with early stopping and learning rate reduction callbacks.
5. Model Saving: Saves the trained model to model.h5.

We need to train the model from scratch and run the `train_model.py` script. This script will train the CNN on the MNIST dataset and save the model to `model.h5`.
```
python train_model.py

```

## Running the Application

The app.py script sets up a VS Code application with the following features:
1. Gradio Sketchpad(Img): Allows users to draw any digit using brush.
2. Submit Button: Allows to upload image.
3. Image Preprocessing: Converts the image to grayscale, resizes it to 28x28 pixels, and normalizes the pixel values.
4. Prediction: Uses the pre-trained model to predict the digit in the submitted image.
5. Result Display: Shows top three predictions of digit.
6. Flag Option: Provides a link to download the predicted result as a images.
7. Clear Button: Allows to clear sketchpad & output and draw to upload or predict other digit.

To run the application, execute the app.py script:
```
python app.py

```

Gradio also ceates a local URL `Running on local URL:  http://127.0.0.1:7860` on which interface runs and we get ouputs

## Example Workflow
1. Draw Digit: open url and draw any digit on sketchpad using brush.
2. Submit the Digit: The drawn digit will be submitted to model.
3. Get Prediction: The model will predict the top 3 digits and display the predictions result in percentage.
4. Download Result: Click the flag to save the input and prediction to a folder.
