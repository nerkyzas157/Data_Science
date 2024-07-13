# Handwritten Digit Recognition

## Overview
This project demonstrates how to build a Handwritten Digit Recognition system using TensorFlow/Keras. The goal is to teach a computer to recognize handwritten digits (0-9) from images.

## Features
- Loads the MNIST dataset, which contains a large collection of handwritten digit images.
- Preprocesses the data by scaling pixel values to be between 0 and 1.
- Builds and trains a Convolutional Neural Network model using TensorFlow/Keras.
- Evaluates the model's accuracy on a test dataset.
- Visualizes sample images along with their predicted labels to assess the model's performance.

## Setup
1. Install Python (if not already installed).
2. Create a virtual environment for this project:
```
python -m venv .venv
```
3. Activate the virtual environment:
  ```
  .\.venv\Scripts\activate
  ```
4. Install required packages:
```
pip install -r requirements.txt
```
5. Clone or download this repository to your local machine.

## Usage
1. Open the `main.py` file in an IDE.
2. Train the model using the provided dataset.
3. Evaluate the model's accuracy on the test dataset.
4. Visualize sample images and their predicted labels to assess the model's performance.

## Visualization
After training the model, you can visualize the results by plotting sample images along with their predicted labels. The visualization helps in understanding how well the model is performing on unseen data. Already generated visual is added and named: Figure_1.png.

## Requirements
- Python
- TensorFlow
- Keras
- Jupyter Notebook
- Matplotlib (optional)
- NumPy (optional)

## Credits
- This project was created as part of a learning exercise by nerkyzas157.
