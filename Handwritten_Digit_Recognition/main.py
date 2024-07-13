# Used "pip install -r requirments.txt" command in terminal

# Imported libraries for the project
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

# Loaded built-in MNIST handwriting dataset
mnist = tensorflow.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Pixel values have to be normalized by division
train_images = train_images / 255.0
test_images = test_images / 255.0

# Built a neural network model from Conv2D and Dense
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax"),
    ]
)

# Created compiler to tell the model how to learn
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Training time!
model.fit(train_images[..., tensorflow.newaxis], train_labels, epochs=5, batch_size=64)

# Evaluation of the results
test_loss, test_accuracy = model.evaluate(
    test_images[..., tensorflow.newaxis], test_labels
)
print("Test accuracy:", test_accuracy)

# Finally we can make predictions with on new images
predictions = model.predict(test_images[..., tensorflow.newaxis])

# Choose 5 random images for vizualization of the model results
num_images = 5
random_indices = np.random.choice(len(test_images), num_images, replace=False)
sample_images = test_images[random_indices]
sample_labels = test_labels[random_indices]

# Use trained model to predict labels for the sample images
sample_predictions = model.predict(sample_images[..., tensorflow.newaxis])
predicted_labels = np.argmax(sample_predictions, axis=1)

# Created visuals for the sample images along with their predicted labels
plt.figure(figsize=(10, 5))
for i in range(num_images):
    plt.subplot(1, num_images, i + 1)
    plt.imshow(sample_images[i], cmap="gray")
    plt.title(f"Predicted: {predicted_labels[i]}\nActual: {sample_labels[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

# Test with random digit images from the web
from PIL import Image

# Defined path
image_paths = [
    "Handwritten_Digit_Recognition\\test1.png",
    "Handwritten_Digit_Recognition\\test0.png",
    "Handwritten_Digit_Recognition\\test2.png",
    "Handwritten_Digit_Recognition\\test3.jpg",
]


# Set preprocessing settings
def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image = np.array(image) / 255.0  # Normalize pixel values
    return image


# Preprocessed images and set labels
images = [preprocess_image(image_path) for image_path in image_paths]
labels = [7, 1, 7, 6]

# Converted into arrays
images = np.array(images)
labels = np.array(labels)

# Expanded dimentions
images = np.expand_dims(images, axis=-1)

# Made predictions
predictions = model.predict(images)

# Vizualized the results
plt.figure(figsize=(12, 6))
for i in range(len(image_paths)):
    plt.subplot(1, len(image_paths), i + 1)
    plt.imshow(images[i].squeeze(), cmap="gray")
    plt.title(f"Predicted: {np.argmax(predictions[i])}\nActual: {labels[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
