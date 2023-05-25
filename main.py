import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import cv2

def load_data():
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Preprocess the data
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Reshape the data
    X_train = np.reshape(X_train, (-1, 28, 28, 1))
    X_test = np.reshape(X_test, (-1, 28, 28, 1))

    return X_test, Y_test

def load_model(model_file):
    # Load the model from the file
    model = tf.keras.models.load_model(model_file)

    return model

def test_model(model, X_test, Y_test):
    # Evaluate the model on the testing dataset
    test_loss, test_acc = model.evaluate(X_test, Y_test)

    # Print the test loss and accuracy
    print("Test Loss: ", test_loss)
    print("Test Accuracy: ", test_acc)

    # Perform predictions on the testing dataset
    predictions = model.predict(X_test)

    # Convert predicted probabilities to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Calculate and print classification report
    print("Classification Report:")
    print(classification_report(Y_test, predicted_labels))

    # Plot a confusion matrix
    cm = confusion_matrix(Y_test, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

# Load the testing data
X_test, Y_test = load_data()

# Specify the path to your model file
model_file = "MNIST_Model.h5"

# Load the model
model = load_model(model_file)

# Test the model
test_model(model, X_test, Y_test)

# Read the image
# You can Try another digit by selecting from digits folder
image = cv2.imread('digits/five.png', cv2.IMREAD_GRAYSCALE)


# Resize the image to match the input shape of your model
image = cv2.resize(image, (28, 28), interpolation= cv2.INTER_AREA)
image.shape

# Normalize the image
image = image / 255.0

# Reshape the image to match the input shape of your model
image = np.reshape(image, (1, 28, 28, 1))

# show the images
plt.imshow(image.reshape(28, 28), cmap="Greys")

# Perform prediction on the external image
prediction = model.predict(image)

# Convert predicted probabilities to class label
predicted_label = np.argmax(prediction[0])

# Print the predicted label
print("Predicted Label:", predicted_label)

